# ---------------------------------------------------------------
# Copyright (c) 2022 BIT-DA. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# The ema model update and the domain-mixing are based on:
# https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License.
# A copy of the license is available at resources/license_dacs

import math
import os
import random
from copy import deepcopy

import numpy as np
import torch
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd

from mmseg.core import add_prefix
from mmseg.models import UDA, build_segmentor
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks, get_diverse_crack_mask, get_strategic_crack_mask, get_mean_std, strong_transform, get_morphology_aware_crack_mask, get_width_adaptive_weight)
from mmseg.models.utils.visualization import subplotimg
from mmseg.models.utils.ours_transforms import RandomCrop, RandomCropNoProd

from mmseg.models.utils.proto_estimator import ProtoEstimator
from mmseg.models.losses.contrastive_loss import contrast_preparations
from mmseg.models.utils.dacs_transforms import extract_crack_morphology

def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm



def _componentwise_ori_bin_skel_pca(
    crack_mask: torch.Tensor,   # [B,1,H,W] float/bool
    num_bins: int,
    min_pixels: int = 30,       # component 최소 픽셀
    ignore_val: int = 255,
    do_close: bool = False,
    close_ksize: int = 3,
    skel_min_pixels: int = 10,  # skeleton 픽셀 너무 적으면 fallback
    linearity_thr: float = 3.0, # λ1/λ2 < thr 이면 방향 불명확 -> ignore
):
    """
    return: ori_bin_map [B,H,W] (torch.long)
            crack 픽셀: 0 ~ num_bins-1
            non-crack: ignore_val
    """
    import numpy as np
    import cv2

    B, _, H, W = crack_mask.shape
    device = crack_mask.device

    out = torch.full((B, H, W), ignore_val, dtype=torch.long, device=device)

    m = (crack_mask > 0.5).detach().cpu().numpy().astype(np.uint8)

    # connected components (same as your code)
    try:
        from scipy.ndimage import label as ndi_label
        label_fn = lambda x: ndi_label(x, structure=np.ones((3,3)))[0]
    except Exception:
        from skimage.measure import label as sk_label
        label_fn = lambda x: sk_label(x, connectivity=2)

    # skeletonize
    try:
        from skimage.morphology import skeletonize
        skel_fn = lambda x: skeletonize(x > 0).astype(np.uint8)
    except Exception:
        # fallback: Zhang-Suen thinning via OpenCV ximgproc if available
        def skel_fn(x):
            if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
                return cv2.ximgproc.thinning((x > 0).astype(np.uint8) * 255) // 255
            # very last fallback: no skeleton => use original mask (less ideal)
            return (x > 0).astype(np.uint8)

    # optional gap closing (direction estimation stability)
    if do_close:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        for b in range(B):
            m[b, 0] = cv2.morphologyEx(m[b, 0], cv2.MORPH_CLOSE, k)

    for b in range(B):
        comp = label_fn(m[b, 0])
        if comp.max() == 0:
            continue

        bin_map = np.full((H, W), ignore_val, dtype=np.int64)

        for cid in range(1, int(comp.max()) + 1):
            ys, xs = np.where(comp == cid)
            if len(xs) < min_pixels:
                continue

            comp_mask = (comp == cid).astype(np.uint8)

            # skeleton of this component
            skel = skel_fn(comp_mask)
            yk, xk = np.where(skel > 0)

            # fallback: if skeleton too small, use component pixels for PCA
            if len(xk) < skel_min_pixels:
                xk, yk = xs, ys

            # PCA on points (x, y)
            P = np.stack([xk, yk], axis=1).astype(np.float32)  # [N,2]
            if P.shape[0] < 2:
                continue

            Pm = P - P.mean(axis=0, keepdims=True)
            C = (Pm.T @ Pm) / (Pm.shape[0] + 1e-6)  # [2,2]

            # eigen-decomposition
            evals, evecs = np.linalg.eigh(C)  # evals asc
            # principal axis = eigenvector of largest eigenvalue
            v = evecs[:, np.argmax(evals)]  # [2], corresponds to (vx, vy)

            # linearity check (optional but recommended)
            lam1 = float(np.max(evals))
            lam2 = float(np.min(evals))
            if lam2 <= 1e-12:
                ratio = 1e12
            else:
                ratio = lam1 / (lam2 + 1e-12)

            if ratio < linearity_thr:
                # component is too blob-like / branching -> ignore to reduce label noise
                continue

            angle = np.arctan2(v[1], v[0])  # [-pi, pi]
            # undirected: fold to [0, pi)
            angle = angle % np.pi

            bin_id = int(np.floor(angle / (np.pi / num_bins)))
            bin_id = min(max(bin_id, 0), num_bins - 1)

            bin_map[ys, xs] = bin_id

        out[b] = torch.from_numpy(bin_map).to(device)

    return out
import torch
import torch.nn.functional as F

def compute_boundary_mask(crack_mask, kernel_size=3):
    """
    crack_mask: [B,1,H,W] (0/1)
    return:
        bd_mask: [B,1,H,W] (1=boundary, 0=interior)
    """
    pad = kernel_size // 2
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=crack_mask.device)

    # erosion
    eroded = F.conv2d(crack_mask.float(), kernel, padding=pad)
    eroded = (eroded == kernel.numel()).float()

    boundary = crack_mask.float() - eroded
    boundary = (boundary > 0).float()

    return boundary
from torch.autograd import Function


class GradReverse(Function):

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)
import torch
import torch.nn as nn
import torch.nn.functional as F

def classwise_adv_loss(
        src_feat,
        src_mask,
        tgt_feat,
        tgt_mask,
        discriminator,
        crack_class=1,
        bg_class=0,
        ignore_index=255,
        grl_lambda=0.1,
        crack_weight=1.0,
        bg_weight=0.3,
        min_samples=16):

    device = src_feat.device

    # -----------------------------
    # 1) mask size -> feat size
    # -----------------------------
    if src_mask.dim() == 3:
        src_mask = src_mask.unsqueeze(1)   # [B,1,H,W]
    if tgt_mask.dim() == 3:
        tgt_mask = tgt_mask.unsqueeze(1)

    src_mask = F.interpolate(
        src_mask.float(),
        size=src_feat.shape[2:],
        mode='nearest'
    ).long()

    tgt_mask = F.interpolate(
        tgt_mask.float(),
        size=tgt_feat.shape[2:],
        mode='nearest'
    ).long()

    # -----------------------------
    # 2) flatten
    # -----------------------------
    B, C, H, W = src_feat.shape
    src_feat = src_feat.permute(0, 2, 3, 1).reshape(-1, C)
    tgt_feat = tgt_feat.permute(0, 2, 3, 1).reshape(-1, C)

    src_mask = src_mask.squeeze(1).reshape(-1)
    tgt_mask = tgt_mask.squeeze(1).reshape(-1)

    def adv_one_class(cls_id):
        src_sel = (src_mask != ignore_index) & (src_mask == cls_id)
        tgt_sel = (tgt_mask != ignore_index) & (tgt_mask == cls_id)

        src_f = src_feat[src_sel]
        tgt_f = tgt_feat[tgt_sel]

        if src_f.shape[0] < min_samples or tgt_f.shape[0] < min_samples:
            return torch.tensor(0., device=device)

        src_f = grad_reverse(src_f, grl_lambda)
        tgt_f = grad_reverse(tgt_f, grl_lambda)

        logit_src = discriminator(src_f)
        logit_tgt = discriminator(tgt_f)

        dom_src = torch.zeros_like(logit_src)
        dom_tgt = torch.ones_like(logit_tgt)

        loss_src = F.binary_cross_entropy_with_logits(logit_src, dom_src)
        loss_tgt = F.binary_cross_entropy_with_logits(logit_tgt, dom_tgt)

        return 0.5 * (loss_src + loss_tgt)

    loss_crack = adv_one_class(crack_class)
    loss_bg = adv_one_class(bg_class)

    loss = crack_weight * loss_crack + bg_weight * loss_bg
    return loss
class FeatureDomainDiscriminator(nn.Module):
    """
    Domain discriminator for adversarial domain adaptation.
    Input: pixel feature [N, C]
    Output: domain logit [N, 1]
    """

    def __init__(self, in_dim=256, hidden_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)
@UDA.register_module()
class SePiCo(UDADecorator):

    def __init__(self, **cfg):
        super(SePiCo, self).__init__(**cfg)
        # basic setup
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.alpha = cfg['alpha']
        self.compute_boundary_mask = compute_boundary_mask
        # for ssl
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.mix = cfg['mix']
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.debug_img_interval = cfg['debug_img_interval']
        assert self.mix == 'class'
        self.enable_self_training = cfg['enable_self_training']
        self.enable_strong_aug = cfg['enable_strong_aug']
        self.push_off_self_training = cfg.get('push_off_self_training', False)
        self.num_ori_bins = 8
        # configs for contrastive
        self.proj_dim = cfg['model']['auxiliary_head']['channels']
        self.contrast_mode = cfg['model']['auxiliary_head']['input_transform']
        self.calc_layers = cfg['model']['auxiliary_head']['in_index']
        self.num_classes = cfg['model']['decode_head']['num_classes']
        self.enable_avg_pool = cfg['model']['auxiliary_head']['loss_decode']['use_avg_pool']
        self.scale_min_ratio = cfg['model']['auxiliary_head']['loss_decode']['scale_min_ratio']
        
        # iter to start cl
        self.start_distribution_iter = cfg['start_distribution_iter']
        self.start_bank_iter = cfg.get('start_bank_iter', 24000)
        self.start_graph_iter = cfg.get('start_graph_iter', self.start_distribution_iter)
        # for prod strategy (CBC)
        self.pseudo_random_crop = cfg.get('pseudo_random_crop', False)
        self.crop_size = cfg.get('crop_size', (640, 640))
        self.cat_max_ratio = cfg.get('cat_max_ratio', 0.75)
        self.regen_pseudo = cfg.get('regen_pseudo', False)
        self.prod = cfg.get('prod', True)
        self.crack_idx = 1  # 이 줄을 추가하세요
        # feature storage for contrastive
        self.feat_distributions = None
        self.ignore_index = 255
        self.enable_bridge_bank = cfg.get('enable_bridge_bank', False)
        self.start_bridge_iter = cfg.get('start_bridge_iter', self.start_distribution_iter)
        self.bridge_update_prob = cfg.get('bridge_update_prob', 1.0)
        self.bridge_pseudo_threshold = cfg.get('bridge_pseudo_threshold', 0.7)
        self.domain_disc = FeatureDomainDiscriminator(
            in_dim=256,
            hidden_dim=128
        )
        self.classwise_adv_loss = classwise_adv_loss
        # BankCL memory length
        self.memory_length = cfg.get('memory_length', 0)  # 0 means no memory bank
        # prototype bank 초기화
        self.crack_proto_bank    = None
        self.proto_reliability   = 0.0
        # init distribution
        if self.contrast_mode == 'multiple_select':
            self.feat_distributions = {}
            for idx in range(len(self.calc_layers)):
                self.feat_distributions[idx] = ProtoEstimator(dim=self.proj_dim, class_num=self.num_classes,
                                                              memory_length=self.memory_length)
        else:  # 'resize_concat' or None
            self.feat_distributions = ProtoEstimator(dim=self.proj_dim, class_num=self.num_classes,
                                                     memory_length=self.memory_length)
        # bridge bank init (source bank와 동일 스펙)
        if self.enable_bridge_bank:
            if self.contrast_mode == 'multiple_select':
                self.bridge_distributions = {}
                for idx in range(len(self.calc_layers)):
                    self.bridge_distributions[idx] = ProtoEstimator(
                        dim=self.proj_dim, class_num=self.num_classes, memory_length=self.memory_length
                    )
            else:
                self.bridge_distributions = ProtoEstimator(
                    dim=self.proj_dim, class_num=self.num_classes, memory_length=self.memory_length
                )
        # ema model
        ema_cfg = deepcopy(cfg['model'])
        self.ema_model = build_segmentor(ema_cfg)

    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    def _init_ema_weights(self):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        #optimizer.zero_grad()
        #log_vars = self(**data_batch)
        #optimizer.step()

        #log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        #outputs = dict(
        #    log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        outputs = self(**data_batch)
        #optimizer.step()
        return outputs

        #return outputs
    from mmseg.models.utils.dacs_transforms import extract_crack_morphology
    
    def get_thin_wide_masks(self, label, crack_class=1, thin_thresh=3.0):
        B = label.shape[0]
        thin_masks = torch.zeros(B, *label.shape[-2:], device=label.device)
        wide_masks = torch.zeros(B, *label.shape[-2:], device=label.device)
    
        for i in range(B):
            skel, width_map, _ = extract_crack_morphology(label[i], crack_class)
            crack_np = (label[i].squeeze().cpu().numpy() == crack_class)
    
            width_on_crack = np.zeros_like(width_map)
            width_on_crack[crack_np] = width_map[crack_np]
    
            thin = crack_np & (width_on_crack < thin_thresh)
            wide = crack_np & (width_on_crack >= thin_thresh)
    
            thin_masks[i] = torch.from_numpy(thin.astype(np.float32))
            wide_masks[i] = torch.from_numpy(wide.astype(np.float32))
    
        return thin_masks, wide_masks
    def random_crop(self, image, gt_seg, prod=True):
        if prod:
            RC = RandomCrop(crop_size=self.crop_size, cat_max_ratio=self.cat_max_ratio)
        else:
            RC = RandomCropNoProd(crop_size=self.crop_size, cat_max_ratio=self.cat_max_ratio)
        assert self.pseudo_random_crop
        image = image.permute(0, 2, 3, 1).contiguous()
        gt_seg = gt_seg
        res_img, res_gt = [], []
        for img, gt in zip(image, gt_seg):
            results = {'img': img, 'gt_semantic_seg': gt, 'seg_fields': ['gt_semantic_seg']}
            results = RC(results)
            img, gt = results['img'], results['gt_semantic_seg']
            res_img.append(img.unsqueeze(0))
            res_gt.append(gt.unsqueeze(0))
        image = torch.cat(res_img, dim=0).permute(0, 3, 1, 2).contiguous()
        gt_seg = torch.cat(res_gt, dim=0).long()
        return image, gt_seg
    import math
    
    def ramp_hold_linear(self, cur, start, hold, warm, maxv):
        if cur < start + hold:
            return 0.0
        t = (cur - (start + hold)) / max(1, warm)
        t = max(0.0, min(1.0, t))
        return maxv * t
    
    def ramp_hold_cosine(self, cur, start, hold, warm, maxv):
        if cur < start + hold:
            return 0.0
        t = (cur - (start + hold)) / max(1, warm)
        t = max(0.0, min(1.0, t))
        return maxv * 0.5 * (1 - math.cos(math.pi * t))

    def compute_skeleton_mask(self, crack_mask_01: torch.Tensor):
        """
        crack_mask_01: [B,1,H,W] float/byte (0/1)
        return:        [B,H,W]   uint8 (0/1)
        """
        from skimage.morphology import skeletonize
        B, _, H, W = crack_mask_01.shape
        crack_np = (crack_mask_01[:, 0] > 0.5).detach().cpu().numpy().astype("uint8")
    
        skels = []
        for b in range(B):
            sk = skeletonize(crack_np[b]).astype("uint8")  # 0/1
            skels.append(torch.from_numpy(sk))
        return torch.stack(skels, dim=0).to(crack_mask_01.device)  # [B,H,W]
    def normalize_dist_map(self, dist_map, crack_mask, eps=1e-6):
        # dist_map: [B,1,H,W], crack_mask: [B,1,H,W]
        masked = dist_map * (crack_mask > 0.5).float()
        mean = masked.sum(dim=[2,3], keepdim=True) / ((crack_mask > 0.5).float().sum(dim=[2,3], keepdim=True).clamp_min(1.0))
        return dist_map / (mean.clamp_min(eps))

    
    def compute_dist_map_from_crack_and_bd(self, crack_mask, bd_mask):
        """
        crack_mask: torch float/bool [B,1,H,W] (1=crack)
        bd_mask:    torch long/int  [B,1,H,W] (1=boundary, 0=interior or bg)
        return:     torch float     [B,1,H,W] (bg=0, boundary~0, center larger)
        """
        import numpy as np
        import torch
        from scipy.ndimage import distance_transform_edt
        device = crack_mask.device
        if crack_mask.dim() == 4 and crack_mask.size(1) == 1:
            crack_mask_ = crack_mask[:, 0]   # [B,H,W]
        else:
            crack_mask_ = crack_mask
    
        if bd_mask.dim() == 4 and bd_mask.size(1) == 1:
            bd_mask_ = bd_mask[:, 0]
        else:
            bd_mask_ = bd_mask
    
        B, H, W = crack_mask_.shape
        out = np.zeros((B, H, W), dtype=np.float32)
    
        for b in range(B):
            cr = (crack_mask_[b].detach().cpu().numpy() > 0.5)          # bool
            bd = (bd_mask_[b].detach().cpu().numpy() == 1)              # bool (boundary)
            if cr.sum() == 0:
                continue
            interior = cr & (~bd)
    
            # interior에서 boundary까지 거리:
            # dist_transform은 "0인 위치까지 거리"이므로,
            # interior를 1로 두고 boundary를 0으로 만드는 형태로 구성
            # 가장 간단히: interior에 대해 distance_transform_edt(interior)
            # -> interior 영역 내부에서 가장 가까운 0(=boundary or bg)까지 거리
            dist = distance_transform_edt(interior).astype(np.float32)
    
            # crack 내부만 남기고 bg는 0
            dist *= cr.astype(np.float32)
            out[b] = dist
    
        dist_map = torch.from_numpy(out).unsqueeze(1).to(device)  # [B,1,H,W]
        return dist_map
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    def visualize_ori_map(self, ori_bin, crack_mask, save_path):
    
        ori = ori_bin[0].detach().cpu().numpy()
        mask = crack_mask[0,0].detach().cpu().numpy()
    
        vis = np.zeros((ori.shape[0], ori.shape[1], 3), dtype=np.uint8)
    
        # 색상 팔레트 (bin 수에 맞게)
        colors = np.array([
            [255,0,0],
            [255,128,0],
            [255,255,0],
            [0,255,0],
            [0,255,255],
            [0,0,255],
            [128,0,255],
            [255,0,255],
        ], dtype=np.uint8)
    
        num_bins = len(colors)
    
        for b in range(num_bins):
            vis[ori == b] = colors[b]
    
        vis[mask == 0] = 0
    
        plt.figure(figsize=(6,6))
        plt.imshow(vis)
        plt.title("orientation bins")
        plt.axis("off")
    
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    def refine_pseudo_crack_mask(
        self,
        crack_mask: torch.Tensor,   # [B,1,H,W]
        close_ksize: int = 10,
        min_area: int = 5,
        line_len: int = 10,
    ):
        import cv2
        import numpy as np
        import torch
    
        device = crack_mask.device
        x = (crack_mask > 0.5).detach().cpu().numpy().astype(np.uint8)   # [B,1,H,W]
        out = np.zeros_like(x, dtype=np.uint8)
    
        def make_line_kernel(length=5, angle=0):
            k = np.zeros((length, length), dtype=np.uint8)
            c = length // 2
            if angle == 0:
                k[c, :] = 1
            elif angle == 90:
                k[:, c] = 1
            elif angle == 45:
                for i in range(length):
                    k[length - 1 - i, i] = 1
            elif angle == 135:
                for i in range(length):
                    k[i, i] = 1
            return k
    
        kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_ksize, close_ksize)
        )
    
        for b in range(x.shape[0]):
            m = x[b, 0].copy()   # [H,W]
    
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel_close)
    
            merged = m.copy()
            for ang in [0, 45, 90, 135]:
                k = make_line_kernel(line_len, ang)
                mc = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
                merged = np.maximum(merged, mc)
    
            m = merged
    
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    
            cleaned = np.zeros_like(m, dtype=np.uint8)
            for cid in range(1, num_labels):
                if stats[cid, cv2.CC_STAT_AREA] >= min_area:
                    cleaned[labels == cid] = 1
    
            out[b, 0] = cleaned
    
        return torch.from_numpy(out).float().to(device)
    '''
    def refine_pseudo_crack_mask_endpoint_bridge(
        self,
        crack_mask: torch.Tensor,   # [B,1,H,W]
        min_area: int = 3,
        max_gap: int = 60,
        max_angle_diff: float = 75.0,
        thickness: int = 2,
        tangent_radius: int = 9,    # endpoint 주변 local tangent 추정 반경
    ):
        import cv2
        import numpy as np
        import torch
    
        device = crack_mask.device
        x = (crack_mask > 0.5).detach().cpu().numpy().astype(np.uint8)
        out = np.zeros_like(x, dtype=np.uint8)
    
        try:
            from skimage.morphology import skeletonize
            skel_fn = lambda z: skeletonize(z > 0).astype(np.uint8)
        except Exception:
            def skel_fn(z):
                if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
                    return (cv2.ximgproc.thinning((z > 0).astype(np.uint8) * 255) > 0).astype(np.uint8)
                return (z > 0).astype(np.uint8)
    
        def _angle_diff_deg(a, b):
            d = abs(a - b) % np.pi
            d = min(d, np.pi - d)
            return np.degrees(d)
    
        def _line_angle(p, q):
            vx = float(q[0] - p[0])
            vy = float(q[1] - p[1])
            return np.arctan2(vy, vx) % np.pi
    
        def _extract_endpoints_and_tangents(comp):
            """
            comp: [H,W] uint8 binary
            return:
                {
                  "mask": comp,
                  "skel": skel,
                  "endpoints": [N,2] (x,y),
                  "tangents":  [N]   angle in [0, pi)
                }
            """
            skel = skel_fn(comp)
    
            kernel = np.array([[1, 1, 1],
                               [1, 0, 1],
                               [1, 1, 1]], dtype=np.uint8)
            deg = cv2.filter2D(skel.astype(np.uint8), -1, kernel)
            ey, ex = np.where((skel > 0) & (deg == 1))
    
            ys_all, xs_all = np.where(skel > 0)
            if len(xs_all) < 2:
                return None
    
            skel_pts = np.stack([xs_all, ys_all], axis=1).astype(np.float32)
    
            endpoints = []
            tangents = []
    
            def _local_tangent(ep_xy):
                # endpoint 주변 skeleton 점들만 모아서 PCA
                d2 = np.sum((skel_pts - ep_xy[None, :]) ** 2, axis=1)
                idx = np.where(d2 <= tangent_radius * tangent_radius)[0]
    
                # 주변 점이 너무 적으면 fallback: 가장 가까운 몇 점 사용
                if len(idx) < 2:
                    order = np.argsort(d2)
                    idx = order[:min(8, len(order))]
    
                pts = skel_pts[idx]
                if len(pts) < 2:
                    return None
    
                center = pts.mean(axis=0, keepdims=True)
                pm = pts - center
                C = (pm.T @ pm) / (len(pts) + 1e-6)
                evals, evecs = np.linalg.eigh(C)
                v = evecs[:, np.argmax(evals)]
                theta = np.arctan2(v[1], v[0]) % np.pi
                return theta
    
            if len(ex) >= 2:
                eps = np.stack([ex, ey], axis=1).astype(np.int32)
                for ep in eps:
                    th = _local_tangent(ep.astype(np.float32))
                    if th is not None:
                        endpoints.append(ep)
                        tangents.append(th)
            else:
                # fallback: skeleton 전체 PCA extremal 2점 사용
                center = skel_pts.mean(axis=0, keepdims=True)
                pm = skel_pts - center
                C = (pm.T @ pm) / (len(skel_pts) + 1e-6)
                evals, evecs = np.linalg.eigh(C)
                v = evecs[:, np.argmax(evals)]
                proj = pm @ v
                p1 = skel_pts[np.argmin(proj)].astype(np.int32)
                p2 = skel_pts[np.argmax(proj)].astype(np.int32)
    
                for ep in [p1, p2]:
                    th = _local_tangent(ep.astype(np.float32))
                    if th is None:
                        th = np.arctan2(v[1], v[0]) % np.pi
                    endpoints.append(ep)
                    tangents.append(th)
    
            if len(endpoints) == 0:
                return None
    
            endpoints = np.stack(endpoints, axis=0).astype(np.int32)
            tangents = np.array(tangents, dtype=np.float32)
    
            return {
                "mask": comp.astype(np.uint8),
                "skel": skel.astype(np.uint8),
                "endpoints": endpoints,   # [N,2]
                "tangents": tangents,     # [N]
            }
    
        for b in range(x.shape[0]):
            m = x[b, 0].copy()
    
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    
            comps = []
            for cid in range(1, num_labels):
                area = stats[cid, cv2.CC_STAT_AREA]
                if area < min_area:
                    continue
    
                comp = (labels == cid).astype(np.uint8)
                info = _extract_endpoints_and_tangents(comp)
                if info is None:
                    continue
                comps.append(info)
    
            if len(comps) == 0:
                out[b, 0] = m
                continue
    
            linked = np.zeros_like(m, dtype=np.uint8)
            for c in comps:
                linked = np.maximum(linked, c["mask"])
    
            # pairwise bridge using endpoint local tangents
            for i in range(len(comps)):
                for j in range(i + 1, len(comps)):
                    ep1 = comps[i]["endpoints"]
                    tg1 = comps[i]["tangents"]
                    ep2 = comps[j]["endpoints"]
                    tg2 = comps[j]["tangents"]
    
                    best = None
                    best_score = 1e9
    
                    for ii, p in enumerate(ep1):
                        for jj, q in enumerate(ep2):
                            d = np.linalg.norm(p - q)
                            if d > max_gap:
                                continue
    
                            line_theta_pq = _line_angle(p, q)
                            line_theta_qp = _line_angle(q, p)
    
                            a1 = _angle_diff_deg(tg1[ii], line_theta_pq)
                            a2 = _angle_diff_deg(tg2[jj], line_theta_qp)
    
                            # local tangent 기준 완화 규칙
                            # 둘 다 많이 안 맞을 때만 reject
                            if a1 > max_angle_diff and a2 > max_angle_diff:
                                continue
    
                            # distance 중심 + angle soft penalty
                            score = d + 0.20 * (a1 + a2)
    
                            if score < best_score:
                                best_score = score
                                best = (tuple(p.tolist()), tuple(q.tolist()))
    
                    if best is None:
                        continue
    
                    p, q = best
                    cv2.line(linked, p, q, color=1, thickness=thickness)
    
            # optional light closing after bridge
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            linked = cv2.morphologyEx(linked, cv2.MORPH_CLOSE, kernel_close)
    
            # final cleanup
            num2, lab2, st2, _ = cv2.connectedComponentsWithStats(linked, connectivity=8)
            cleaned = np.zeros_like(linked, dtype=np.uint8)
            for cid in range(1, num2):
                if st2[cid, cv2.CC_STAT_AREA] >= min_area:
                    cleaned[lab2 == cid] = 1
    
            out[b, 0] = cleaned
    
        return torch.from_numpy(out).float().to(device)
    
    def refine_pseudo_crack_mask_endpoint_bridge(
        self,
        crack_mask: torch.Tensor,   # [B,1,H,W]
        min_area: int = 3,
        max_gap: int = 60,
        max_angle_diff: float = 75.0,
        thickness: int = 2,
        trace_len: int = 12,          # endpoint에서 skeleton 따라 추적할 최대 step
        line_gap: int = 20,           # 이 이하 gap은 직선 연결
        bezier_alpha: float = 0.35,   # control point 거리 비율
        bezier_alpha_max: float = 20.0,
        bezier_num_pts: int = 30,
    ):
        import cv2
        import numpy as np
        import torch
    
        device = crack_mask.device
        x = (crack_mask > 0.5).detach().cpu().numpy().astype(np.uint8)
        out = np.zeros_like(x, dtype=np.uint8)
    
        try:
            from skimage.morphology import skeletonize
            skel_fn = lambda z: skeletonize(z > 0).astype(np.uint8)
        except Exception:
            def skel_fn(z):
                if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
                    return (cv2.ximgproc.thinning((z > 0).astype(np.uint8) * 255) > 0).astype(np.uint8)
                return (z > 0).astype(np.uint8)
    
        def _angle_diff_deg(a, b):
            d = abs(a - b) % np.pi
            d = min(d, np.pi - d)
            return np.degrees(d)
    
        def _line_angle(p, q):
            vx = float(q[0] - p[0])
            vy = float(q[1] - p[1])
            return np.arctan2(vy, vx) % np.pi
    
        def _unit_vec(theta):
            return np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
    
        def _get_neighbors_8(skel, x0, y0):
            H, W = skel.shape
            nbrs = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    xx = x0 + dx
                    yy = y0 + dy
                    if 0 <= xx < W and 0 <= yy < H and skel[yy, xx] > 0:
                        nbrs.append((xx, yy))
            return nbrs
    
        def _trace_from_endpoint(skel, ep_xy, max_steps=12):
            """
            endpoint에서 skeleton을 따라 안쪽으로 tracing
            return: traced points list [(x,y), ...]
            """
            x0, y0 = int(ep_xy[0]), int(ep_xy[1])
    
            if skel[y0, x0] == 0:
                return [(x0, y0)]
    
            path = [(x0, y0)]
            visited = set(path)
    
            nbrs = _get_neighbors_8(skel, x0, y0)
            if len(nbrs) == 0:
                return path
    
            prev = (x0, y0)
            cur = nbrs[0]
            path.append(cur)
            visited.add(cur)
    
            for _ in range(max_steps - 1):
                cx, cy = cur
                cand = _get_neighbors_8(skel, cx, cy)
    
                # 직전 점 제외
                cand = [p for p in cand if p != prev]
    
                if len(cand) == 0:
                    break
    
                if len(cand) == 1:
                    nxt = cand[0]
                else:
                    # 현재 진행 방향과 가장 일치하는 점 선택
                    vx = cur[0] - prev[0]
                    vy = cur[1] - prev[1]
                    vnorm = (vx * vx + vy * vy) ** 0.5 + 1e-6
    
                    best = None
                    best_score = -1e9
                    for cc in cand:
                        wx = cc[0] - cur[0]
                        wy = cc[1] - cur[1]
                        wnorm = (wx * wx + wy * wy) ** 0.5 + 1e-6
                        cos_sim = (vx * wx + vy * wy) / (vnorm * wnorm)
                        if cos_sim > best_score:
                            best_score = cos_sim
                            best = cc
                    nxt = best
    
                if nxt in visited:
                    break
    
                path.append(nxt)
                visited.add(nxt)
                prev, cur = cur, nxt
    
            return path
    
        def _path_tangent_from_endpoint(skel, ep_xy, max_steps=12):
            """
            endpoint에서 skeleton path-following으로 tangent 추정
            return angle in [0, pi)
            """
            path = _trace_from_endpoint(skel, ep_xy, max_steps=max_steps)
    
            if len(path) < 2:
                return None
    
            p0 = np.array(path[0], dtype=np.float32)
            p1 = np.array(path[-1], dtype=np.float32)
    
            v = p1 - p0
            if np.linalg.norm(v) < 1e-6:
                return None
    
            theta = np.arctan2(v[1], v[0]) % np.pi
            return theta
    
        def _draw_quadratic_bezier(mask, p, q, th1, th2, gap, thickness=2):
            """
            endpoint tangent를 반영한 quadratic bezier curve bridge
            p, q: tuple/list (x, y)
            th1, th2: endpoint tangent angle in [0, pi)
            """
            p = np.array(p, dtype=np.float32)
            q = np.array(q, dtype=np.float32)
    
            v1 = _unit_vec(th1)
            v2 = _unit_vec(th2)
    
            alpha = min(gap * bezier_alpha, bezier_alpha_max)
            c1 = p + alpha * v1
            c2 = q - alpha * v2
            c = 0.5 * (c1 + c2)
    
            pts = []
            for t in np.linspace(0.0, 1.0, bezier_num_pts):
                pt = ((1.0 - t) ** 2) * p + 2.0 * (1.0 - t) * t * c + (t ** 2) * q
                pts.append(pt.astype(np.int32))
    
            pts = np.array(pts, dtype=np.int32)
            for k in range(len(pts) - 1):
                pk = tuple(pts[k].tolist())
                qk = tuple(pts[k + 1].tolist())
                cv2.line(mask, pk, qk, color=1, thickness=thickness)
    
        def _extract_endpoints_and_tangents(comp):
            """
            comp: [H,W] uint8 binary
            return dict or None
            """
            skel = skel_fn(comp)
    
            kernel = np.array([[1, 1, 1],
                               [1, 0, 1],
                               [1, 1, 1]], dtype=np.uint8)
            deg = cv2.filter2D(skel.astype(np.uint8), -1, kernel)
            ey, ex = np.where((skel > 0) & (deg == 1))
    
            ys_all, xs_all = np.where(skel > 0)
            if len(xs_all) < 2:
                return None
    
            endpoints = []
            tangents = []
    
            if len(ex) >= 1:
                eps = np.stack([ex, ey], axis=1).astype(np.int32)
    
                for ep in eps:
                    th = _path_tangent_from_endpoint(skel, ep, max_steps=trace_len)
                    if th is not None:
                        endpoints.append(ep)
                        tangents.append(th)
    
            # endpoint를 못 찾았거나 tangent가 안 나오면 fallback
            if len(endpoints) == 0:
                skel_pts = np.stack([xs_all, ys_all], axis=1).astype(np.float32)
    
                center = skel_pts.mean(axis=0, keepdims=True)
                pm = skel_pts - center
                C = (pm.T @ pm) / (len(skel_pts) + 1e-6)
                evals, evecs = np.linalg.eigh(C)
                v = evecs[:, np.argmax(evals)]
                proj = pm @ v
    
                p1 = skel_pts[np.argmin(proj)].astype(np.int32)
                p2 = skel_pts[np.argmax(proj)].astype(np.int32)
                fallback_theta = np.arctan2(v[1], v[0]) % np.pi
    
                for ep in [p1, p2]:
                    endpoints.append(ep)
                    tangents.append(fallback_theta)
    
            endpoints = np.stack(endpoints, axis=0).astype(np.int32)
            tangents = np.array(tangents, dtype=np.float32)
    
            return {
                "mask": comp.astype(np.uint8),
                "skel": skel.astype(np.uint8),
                "endpoints": endpoints,   # [N,2]
                "tangents": tangents,     # [N]
            }
    
        for b in range(x.shape[0]):
            m = x[b, 0].copy()
    
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    
            comps = []
            for cid in range(1, num_labels):
                area = stats[cid, cv2.CC_STAT_AREA]
                if area < min_area:
                    continue
    
                comp = (labels == cid).astype(np.uint8)
                info = _extract_endpoints_and_tangents(comp)
                if info is None:
                    continue
                comps.append(info)
    
            if len(comps) == 0:
                out[b, 0] = m
                continue
    
            linked = np.zeros_like(m, dtype=np.uint8)
            for c in comps:
                linked = np.maximum(linked, c["mask"])
    
            # pairwise bridge using path-following tangents + bezier
            for i in range(len(comps)):
                for j in range(i + 1, len(comps)):
                    ep1 = comps[i]["endpoints"]
                    tg1 = comps[i]["tangents"]
                    ep2 = comps[j]["endpoints"]
                    tg2 = comps[j]["tangents"]
    
                    best = None
                    best_score = 1e9
    
                    for ii, p in enumerate(ep1):
                        for jj, q in enumerate(ep2):
                            d = np.linalg.norm(p - q)
                            if d > max_gap:
                                continue
    
                            theta_pq = _line_angle(p, q)
                            theta_qp = _line_angle(q, p)
    
                            a1 = _angle_diff_deg(tg1[ii], theta_pq)
                            a2 = _angle_diff_deg(tg2[jj], theta_qp)
    
                            # 둘 다 많이 틀릴 때만 reject
                            if a1 > max_angle_diff and a2 > max_angle_diff:
                                continue
    
                            score = d + 0.20 * (a1 + a2)
    
                            if score < best_score:
                                best_score = score
                                best = {
                                    "p": tuple(p.tolist()),
                                    "q": tuple(q.tolist()),
                                    "ii": ii,
                                    "jj": jj,
                                    "d": float(d),
                                    "a1": float(a1),
                                    "a2": float(a2),
                                }
    
                    if best is None:
                        continue
    
                    p = best["p"]
                    q = best["q"]
                    ii = best["ii"]
                    jj = best["jj"]
                    gap = best["d"]
                    th1 = tg1[ii]
                    th2 = tg2[jj]
    
                    # small gap: straight line
                    if gap <= line_gap:
                        cv2.line(linked, p, q, color=1, thickness=thickness)
                    # medium gap: bezier curve
                    else:
                        _draw_quadratic_bezier(
                            linked, p, q, th1, th2, gap, thickness=thickness
                        )
    
            # optional light closing after bridge
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            linked = cv2.morphologyEx(linked, cv2.MORPH_CLOSE, kernel_close)
    
            # final cleanup
            num2, lab2, st2, _ = cv2.connectedComponentsWithStats(linked, connectivity=8)
            cleaned = np.zeros_like(linked, dtype=np.uint8)
            for cid in range(1, num2):
                if st2[cid, cv2.CC_STAT_AREA] >= min_area:
                    cleaned[lab2 == cid] = 1
    
            out[b, 0] = cleaned
    
        return torch.from_numpy(out).float().to(device)
    '''
    def propagate_orientation_field(
        self,
        crack_mask: torch.Tensor,      # [B,1,H,W]
        ori_bin_map: torch.Tensor,     # [B,H,W]
        num_bins: int,
        max_radius: int = 25,
        ignore_val: int = 255,
    ):
        import numpy as np
        import cv2
        import torch
    
        device = crack_mask.device
        B, _, H, W = crack_mask.shape
    
        m = (crack_mask > 0.5).detach().cpu().numpy().astype(np.uint8)
        ori = ori_bin_map.detach().cpu().numpy().astype(np.int64)
    
        out = np.full((B, H, W), ignore_val, dtype=np.int64)
    
        for b in range(B):
            src = (m[b, 0] > 0) & (ori[b] != ignore_val)
            if src.sum() == 0:
                continue
    
            inv = (~src).astype(np.uint8)
            dist, labels = cv2.distanceTransformWithLabels(
                inv,
                distanceType=cv2.DIST_L2,
                maskSize=5,
                labelType=cv2.DIST_LABEL_PIXEL
            )
    
            ys, xs = np.where(src)
            src_pts = np.stack([ys, xs], axis=1)
            src_bins = ori[b, ys, xs]
    
            field = np.full((H, W), ignore_val, dtype=np.int64)
    
            valid = dist <= float(max_radius)
            vy, vx = np.where(valid)
    
            for y, x in zip(vy, vx):
                lab = labels[y, x]
                if lab <= 0 or lab > len(src_pts):
                    continue
                sy, sx = src_pts[lab - 1]
                field[y, x] = ori[b, sy, sx]
    
            field[src] = ori[b][src]
            out[b] = field
    
        return torch.from_numpy(out).long().to(device)
    '''
    def extend_crack_along_orientation(
        self,
        crack_mask: torch.Tensor,      # [B,1,H,W]
        ori_bin_map: torch.Tensor,     # [B,H,W]
        num_bins: int,
        min_area: int = 5,
        trace_len: int = 12,
        extend_max_len: int = 40,
        thickness: int = 2,
        smooth_window: int = 7,
        ignore_val: int = 255,
        ema_keep: float = 0.8,
        endpoint_deg: int = 1,
        min_stable_steps: int = 6,
        max_angle_jitter_deg: float = 20.0,
    ):
        import cv2
        import numpy as np
        import torch
    
        device = crack_mask.device
        x = (crack_mask > 0.5).detach().cpu().numpy().astype(np.uint8)
        ori = ori_bin_map.detach().cpu().numpy().astype(np.int64)
        out = np.zeros_like(x, dtype=np.uint8)
    
        try:
            from skimage.morphology import skeletonize
            skel_fn = lambda z: skeletonize(z > 0).astype(np.uint8)
        except Exception:
            def skel_fn(z):
                if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
                    return (cv2.ximgproc.thinning((z > 0).astype(np.uint8) * 255) > 0).astype(np.uint8)
                return (z > 0).astype(np.uint8)
    
        def _unit_vec(theta):
            return np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
    
        def _get_neighbors_8(skel, x0, y0):
            H, W = skel.shape
            nbrs = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    xx = x0 + dx
                    yy = y0 + dy
                    if 0 <= xx < W and 0 <= yy < H and skel[yy, xx] > 0:
                        nbrs.append((xx, yy))
            return nbrs
    
        def _trace_from_endpoint(skel, ep_xy, max_steps=12):
            x0, y0 = int(ep_xy[0]), int(ep_xy[1])
            if skel[y0, x0] == 0:
                return [(x0, y0)]
    
            path = [(x0, y0)]
            visited = set(path)
            nbrs = _get_neighbors_8(skel, x0, y0)
            if len(nbrs) == 0:
                return path
    
            prev = (x0, y0)
            cur = nbrs[0]
            path.append(cur)
            visited.add(cur)
    
            for _ in range(max_steps - 1):
                cx, cy = cur
                cand = [p for p in _get_neighbors_8(skel, cx, cy) if p != prev]
                if len(cand) == 0:
                    break
                if len(cand) == 1:
                    nxt = cand[0]
                else:
                    vx = cur[0] - prev[0]
                    vy = cur[1] - prev[1]
                    vnorm = (vx * vx + vy * vy) ** 0.5 + 1e-6
                    best, best_score = None, -1e9
                    for cc in cand:
                        wx = cc[0] - cur[0]
                        wy = cc[1] - cur[1]
                        wnorm = (wx * wx + wy * wy) ** 0.5 + 1e-6
                        cos_sim = (vx * wx + vy * wy) / (vnorm * wnorm)
                        if cos_sim > best_score:
                            best_score = cos_sim
                            best = cc
                    nxt = best
                if nxt in visited:
                    break
                path.append(nxt)
                visited.add(nxt)
                prev, cur = cur, nxt
            return path
    
        def _path_tangent_from_endpoint(skel, ep_xy, max_steps=12):
            path = _trace_from_endpoint(skel, ep_xy, max_steps=max_steps)
            if len(path) < 2:
                return None
            p0 = np.array(path[0], dtype=np.float32)
            p1 = np.array(path[-1], dtype=np.float32)
            v = p1 - p0
            if np.linalg.norm(v) < 1e-6:
                return None
            return np.arctan2(v[1], v[0]) % np.pi
    
        def _bin_to_angle_center(bin_id):
            return (bin_id + 0.5) * (np.pi / num_bins)
    
        def _read_local_angle(ori_map_b, x0, y0, win=7, fallback_theta=None):
            H, W = ori_map_b.shape
            r = max(1, win // 2)
            vals = []
            for yy in range(max(0, y0 - r), min(H, y0 + r + 1)):
                for xx in range(max(0, x0 - r), min(W, x0 + r + 1)):
                    bid = int(ori_map_b[yy, xx])
                    if bid == ignore_val or bid < 0 or bid >= num_bins:
                        continue
                    vals.append(_bin_to_angle_center(bid))
            if len(vals) == 0:
                return fallback_theta
            vals = np.array(vals, dtype=np.float32)
            c = np.cos(2.0 * vals).mean()
            s = np.sin(2.0 * vals).mean()
            return (0.5 * np.arctan2(s, c)) % np.pi
    
        def _angle_diff_deg(a, b):
            d = abs(a - b) % np.pi
            d = min(d, np.pi - d)
            return np.degrees(d)
    
        for b in range(x.shape[0]):
            m = x[b, 0].copy()
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            linked = m.copy()
    
            for cid in range(1, num_labels):
                if stats[cid, cv2.CC_STAT_AREA] < min_area:
                    continue
    
                comp = (labels == cid).astype(np.uint8)
                skel = skel_fn(comp)
    
                kernel = np.array([[1, 1, 1],
                                   [1, 0, 1],
                                   [1, 1, 1]], dtype=np.uint8)
                deg = cv2.filter2D(skel.astype(np.uint8), -1, kernel)
                ey, ex = np.where((skel > 0) & (deg == endpoint_deg))
    
                if len(ex) == 0:
                    continue
    
                for x0, y0 in zip(ex, ey):
                    init_theta = _path_tangent_from_endpoint(skel, (x0, y0), max_steps=trace_len)
                    if init_theta is None:
                        continue
    
                    prev_dir = _unit_vec(init_theta)
                    px, py = float(x0), float(y0)
                    pts = [(x0, y0)]
                    stable_count = 0
                    prev_theta = init_theta
    
                    for _ in range(extend_max_len):
                        xi, yi = int(round(px)), int(round(py))
                        th = _read_local_angle(ori[b], xi, yi, win=smooth_window, fallback_theta=prev_theta)
                        if th is None:
                            th = prev_theta
    
                        if _angle_diff_deg(th, prev_theta) <= max_angle_jitter_deg:
                            stable_count += 1
                        else:
                            stable_count = 0
    
                        cand1 = _unit_vec(th)
                        cand2 = -cand1
                        dsel = cand1 if np.dot(cand1, prev_dir) >= np.dot(cand2, prev_dir) else cand2
    
                        new_dir = ema_keep * prev_dir + (1.0 - ema_keep) * dsel
                        new_dir = new_dir / (np.linalg.norm(new_dir) + 1e-6)
    
                        px += float(new_dir[0])
                        py += float(new_dir[1])
    
                        xn, yn = int(round(px)), int(round(py))
                        if not (0 <= xn < m.shape[1] and 0 <= yn < m.shape[0]):
                            break
    
                        pts.append((xn, yn))
                        prev_dir = new_dir
                        prev_theta = th
    
                    if stable_count >= min_stable_steps and len(pts) >= min_stable_steps:
                        for k in range(len(pts) - 1):
                            cv2.line(linked, pts[k], pts[k + 1], color=1, thickness=thickness)
    
            out[b, 0] = linked
    
        return torch.from_numpy(out).float().to(device)
    def refine_pseudo_crack_mask_endpoint_bridge(
        self,
        crack_mask: torch.Tensor,   # [B,1,H,W]
        ori_bin_map: torch.Tensor = None,   # [B,H,W]
        num_bins: int = 8,
        min_area: int = 3,
        max_gap: int = 60,
        max_angle_diff: float = 75.0,
        thickness: int = 2,
        trace_len: int = 12,
        line_gap: int = 20,
        bezier_alpha: float = 0.35,
        bezier_alpha_max: float = 20.0,
        bezier_num_pts: int = 30,
        grow_step: float = 1.0,
        grow_max_steps: int = 80,
        hit_radius: int = 2,
        smooth_window: int = 5,
        meet_thr: float = 3.0,
        ignore_val: int = 255,
        ema_keep: float = 0.7,
    ):
        import cv2
        import numpy as np
        import torch
    
        device = crack_mask.device
        x = (crack_mask > 0.5).detach().cpu().numpy().astype(np.uint8)
        ori = None if ori_bin_map is None else ori_bin_map.detach().cpu().numpy().astype(np.int64)
        out = np.zeros_like(x, dtype=np.uint8)
    
        try:
            from skimage.morphology import skeletonize
            skel_fn = lambda z: skeletonize(z > 0).astype(np.uint8)
        except Exception:
            def skel_fn(z):
                if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
                    return (cv2.ximgproc.thinning((z > 0).astype(np.uint8) * 255) > 0).astype(np.uint8)
                return (z > 0).astype(np.uint8)
    
        def _angle_diff_deg(a, b):
            d = abs(a - b) % np.pi
            d = min(d, np.pi - d)
            return np.degrees(d)
    
        def _line_angle(p, q):
            vx = float(q[0] - p[0])
            vy = float(q[1] - p[1])
            return np.arctan2(vy, vx) % np.pi
    
        def _unit_vec(theta):
            return np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
    
        def _get_neighbors_8(skel, x0, y0):
            H, W = skel.shape
            nbrs = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    xx = x0 + dx
                    yy = y0 + dy
                    if 0 <= xx < W and 0 <= yy < H and skel[yy, xx] > 0:
                        nbrs.append((xx, yy))
            return nbrs
    
        def _trace_from_endpoint(skel, ep_xy, max_steps=12):
            x0, y0 = int(ep_xy[0]), int(ep_xy[1])
    
            if skel[y0, x0] == 0:
                return [(x0, y0)]
    
            path = [(x0, y0)]
            visited = set(path)
    
            nbrs = _get_neighbors_8(skel, x0, y0)
            if len(nbrs) == 0:
                return path
    
            prev = (x0, y0)
            cur = nbrs[0]
            path.append(cur)
            visited.add(cur)
    
            for _ in range(max_steps - 1):
                cx, cy = cur
                cand = _get_neighbors_8(skel, cx, cy)
                cand = [p for p in cand if p != prev]
    
                if len(cand) == 0:
                    break
    
                if len(cand) == 1:
                    nxt = cand[0]
                else:
                    vx = cur[0] - prev[0]
                    vy = cur[1] - prev[1]
                    vnorm = (vx * vx + vy * vy) ** 0.5 + 1e-6
    
                    best = None
                    best_score = -1e9
                    for cc in cand:
                        wx = cc[0] - cur[0]
                        wy = cc[1] - cur[1]
                        wnorm = (wx * wx + wy * wy) ** 0.5 + 1e-6
                        cos_sim = (vx * wx + vy * wy) / (vnorm * wnorm)
                        if cos_sim > best_score:
                            best_score = cos_sim
                            best = cc
                    nxt = best
    
                if nxt in visited:
                    break
    
                path.append(nxt)
                visited.add(nxt)
                prev, cur = cur, nxt
    
            return path
    
        def _path_tangent_from_endpoint(skel, ep_xy, max_steps=12):
            path = _trace_from_endpoint(skel, ep_xy, max_steps=max_steps)
    
            if len(path) < 2:
                return None
    
            p0 = np.array(path[0], dtype=np.float32)
            p1 = np.array(path[-1], dtype=np.float32)
    
            v = p1 - p0
            if np.linalg.norm(v) < 1e-6:
                return None
    
            return np.arctan2(v[1], v[0]) % np.pi
    
        def _draw_quadratic_bezier(mask, p, q, th1, th2, gap, thickness=2):
            p = np.array(p, dtype=np.float32)
            q = np.array(q, dtype=np.float32)
    
            v1 = _unit_vec(th1)
            v2 = _unit_vec(th2)
    
            alpha = min(gap * bezier_alpha, bezier_alpha_max)
            c1 = p + alpha * v1
            c2 = q - alpha * v2
            c = 0.5 * (c1 + c2)
    
            pts = []
            for t in np.linspace(0.0, 1.0, bezier_num_pts):
                pt = ((1.0 - t) ** 2) * p + 2.0 * (1.0 - t) * t * c + (t ** 2) * q
                pts.append(pt.astype(np.int32))
    
            pts = np.array(pts, dtype=np.int32)
            for k in range(len(pts) - 1):
                pk = tuple(pts[k].tolist())
                qk = tuple(pts[k + 1].tolist())
                cv2.line(mask, pk, qk, color=1, thickness=thickness)
    
        def _extract_endpoints_and_tangents(comp):
            skel = skel_fn(comp)
    
            kernel = np.array([[1, 1, 1],
                               [1, 0, 1],
                               [1, 1, 1]], dtype=np.uint8)
            deg = cv2.filter2D(skel.astype(np.uint8), -1, kernel)
            ey, ex = np.where((skel > 0) & (deg == 1))
    
            ys_all, xs_all = np.where(skel > 0)
            if len(xs_all) < 2:
                return None
    
            endpoints = []
            tangents = []
    
            if len(ex) >= 1:
                eps = np.stack([ex, ey], axis=1).astype(np.int32)
    
                for ep in eps:
                    th = _path_tangent_from_endpoint(skel, ep, max_steps=trace_len)
                    if th is not None:
                        endpoints.append(ep)
                        tangents.append(th)
    
            if len(endpoints) == 0:
                skel_pts = np.stack([xs_all, ys_all], axis=1).astype(np.float32)
    
                center = skel_pts.mean(axis=0, keepdims=True)
                pm = skel_pts - center
                C = (pm.T @ pm) / (len(skel_pts) + 1e-6)
                evals, evecs = np.linalg.eigh(C)
                v = evecs[:, np.argmax(evals)]
                proj = pm @ v
    
                p1 = skel_pts[np.argmin(proj)].astype(np.int32)
                p2 = skel_pts[np.argmax(proj)].astype(np.int32)
                fallback_theta = np.arctan2(v[1], v[0]) % np.pi
    
                for ep in [p1, p2]:
                    endpoints.append(ep)
                    tangents.append(fallback_theta)
    
            endpoints = np.stack(endpoints, axis=0).astype(np.int32)
            tangents = np.array(tangents, dtype=np.float32)
    
            return {
                "mask": comp.astype(np.uint8),
                "skel": skel.astype(np.uint8),
                "endpoints": endpoints,
                "tangents": tangents,
            }
    
        def _bin_to_angle_center(bin_id, num_bins):
            return (bin_id + 0.5) * (np.pi / num_bins)
    
        def _read_local_angle_from_orimap(ori_map_b, x0, y0, num_bins, ignore_val=255, win=5, fallback_theta=None):
            H, W = ori_map_b.shape
            r = max(1, win // 2)
    
            vals = []
            for yy in range(max(0, y0 - r), min(H, y0 + r + 1)):
                for xx in range(max(0, x0 - r), min(W, x0 + r + 1)):
                    bid = int(ori_map_b[yy, xx])
                    if bid == ignore_val or bid < 0 or bid >= num_bins:
                        continue
                    vals.append(_bin_to_angle_center(bid, num_bins))
    
            if len(vals) == 0:
                return fallback_theta
    
            vals = np.array(vals, dtype=np.float32)
            c = np.cos(2.0 * vals).mean()
            s = np.sin(2.0 * vals).mean()
            th = 0.5 * np.arctan2(s, c)
            return th % np.pi
    
        def _choose_signed_dir(theta, ref_vec):
            v1 = _unit_vec(theta)
            v2 = -v1
            if np.dot(v1, ref_vec) >= np.dot(v2, ref_vec):
                return v1
            return v2
    
        def _grow_path_from_endpoint_bidir(
            ori_map_b,
            comp_map,
            start_xy,
            init_theta,
            src_cid,
            ref_vec,
            num_bins,
            ignore_val=255,
            grow_step=1.0,
            grow_max_steps=80,
            smooth_window=5,
            ema_keep=0.7,
            max_gap=60,
            hit_radius=2,
        ):
            H, W = ori_map_b.shape
            x, y = float(start_xy[0]), float(start_xy[1])
    
            init_dir = _choose_signed_dir(init_theta, ref_vec)
            prev_dir = init_dir.astype(np.float32)
    
            pts = [(int(round(x)), int(round(y)))]
    
            for _ in range(grow_max_steps):
                xi, yi = int(round(x)), int(round(y))
                if not (0 <= xi < W and 0 <= yi < H):
                    break
    
                th = _read_local_angle_from_orimap(
                    ori_map_b,
                    xi, yi,
                    num_bins=num_bins,
                    ignore_val=ignore_val,
                    win=smooth_window,
                    fallback_theta=init_theta,
                )
    
                if th is None:
                    break
    
                cand1 = _unit_vec(th)
                cand2 = -cand1
                if np.dot(cand1, prev_dir) >= np.dot(cand2, prev_dir):
                    dsel = cand1
                else:
                    dsel = cand2
    
                new_dir = ema_keep * prev_dir + (1.0 - ema_keep) * dsel
                nrm = np.linalg.norm(new_dir) + 1e-6
                new_dir = new_dir / nrm
    
                xn = x + grow_step * float(new_dir[0])
                yn = y + grow_step * float(new_dir[1])
    
                xni, yni = int(round(xn)), int(round(yn))
                if not (0 <= xni < W and 0 <= yni < H):
                    break
    
                if (xni, yni) != pts[-1]:
                    pts.append((xni, yni))
    
                yy0, yy1 = max(0, yni - hit_radius), min(H, yni + hit_radius + 1)
                xx0, xx1 = max(0, xni - hit_radius), min(W, xni + hit_radius + 1)
    
                touched = comp_map[yy0:yy1, xx0:xx1]
                touched_ids = np.unique(touched[touched > 0])
                touched_ids = touched_ids[touched_ids != src_cid]
    
                if len(touched_ids) > 0:
                    return pts, True, int(touched_ids[0])
    
                x, y = xn, yn
                prev_dir = new_dir
    
                if np.hypot(x - start_xy[0], y - start_xy[1]) > max_gap:
                    break
    
            return pts, False, None
    
        def _paths_meet(pts1, pts2, meet_thr=3.0):
            if len(pts1) == 0 or len(pts2) == 0:
                return False, None, None
    
            A = np.array(pts1, dtype=np.float32)
            B = np.array(pts2, dtype=np.float32)
    
            d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)
            idx = np.argmin(d2)
            i_best, j_best = np.unravel_index(idx, d2.shape)
    
            if np.sqrt(d2[i_best, j_best]) <= meet_thr:
                return True, int(i_best), int(j_best)
    
            return False, None, None
    
        def _draw_path(mask, pts, thickness=2):
            if len(pts) == 0:
                return
            if len(pts) == 1:
                cv2.circle(mask, pts[0], radius=max(1, thickness // 2), color=1, thickness=-1)
                return
            for k in range(len(pts) - 1):
                cv2.line(mask, pts[k], pts[k + 1], color=1, thickness=thickness)
    
        for b in range(x.shape[0]):
            m = x[b, 0].copy()
    
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    
            comps = []
            for cid in range(1, num_labels):
                area = stats[cid, cv2.CC_STAT_AREA]
                if area < min_area:
                    continue
    
                comp = (labels == cid).astype(np.uint8)
                info = _extract_endpoints_and_tangents(comp)
                if info is None:
                    continue
                comps.append(info)
    
            if len(comps) == 0:
                out[b, 0] = m
                continue
    
            linked = np.zeros_like(m, dtype=np.uint8)
            for c in comps:
                linked = np.maximum(linked, c["mask"])
    
            comp_map = np.zeros_like(m, dtype=np.int32)
            cid_acc = 1
            for c in comps:
                comp_map[c["mask"] > 0] = cid_acc
                cid_acc += 1
    
            for i in range(len(comps)):
                for j in range(i + 1, len(comps)):
                    ep1 = comps[i]["endpoints"]
                    tg1 = comps[i]["tangents"]
                    ep2 = comps[j]["endpoints"]
                    tg2 = comps[j]["tangents"]
    
                    best = None
                    best_score = 1e9
    
                    for ii, p in enumerate(ep1):
                        for jj, q in enumerate(ep2):
                            d = np.linalg.norm(p - q)
                            if d > max_gap:
                                continue
    
                            theta_pq = _line_angle(p, q)
                            theta_qp = _line_angle(q, p)
    
                            a1 = _angle_diff_deg(tg1[ii], theta_pq)
                            a2 = _angle_diff_deg(tg2[jj], theta_qp)
    
                            if a1 > max_angle_diff and a2 > max_angle_diff:
                                continue
    
                            score = d + 0.20 * (a1 + a2)
    
                            if score < best_score:
                                best_score = score
                                best = {
                                    "p": tuple(p.tolist()),
                                    "q": tuple(q.tolist()),
                                    "ii": ii,
                                    "jj": jj,
                                    "d": float(d),
                                    "src_comp_idx": i,
                                    "dst_comp_idx": j,
                                }
    
                    if best is None:
                        continue
    
                    p = best["p"]
                    q = best["q"]
                    ii = best["ii"]
                    jj = best["jj"]
                    gap = best["d"]
                    th1 = tg1[ii]
                    th2 = tg2[jj]
    
                    src_idx = best["src_comp_idx"]
                    dst_idx = best["dst_comp_idx"]
                    src_cid = src_idx + 1
                    dst_cid = dst_idx + 1
    
                    if ori is None:
                        if gap <= line_gap:
                            cv2.line(linked, p, q, color=1, thickness=thickness)
                        else:
                            _draw_quadratic_bezier(linked, p, q, th1, th2, gap, thickness=thickness)
                        continue
    
                    pq_vec = np.array([q[0] - p[0], q[1] - p[1]], dtype=np.float32)
                    pq_norm = np.linalg.norm(pq_vec) + 1e-6
                    pq_vec = pq_vec / pq_norm
                    qp_vec = -pq_vec
    
                    pts_p, hit_p, hit_cid_p = _grow_path_from_endpoint_bidir(
                        ori_map_b=ori[b],
                        comp_map=comp_map,
                        start_xy=p,
                        init_theta=th1,
                        src_cid=src_cid,
                        ref_vec=pq_vec,
                        num_bins=num_bins,
                        ignore_val=ignore_val,
                        grow_step=grow_step,
                        grow_max_steps=grow_max_steps,
                        smooth_window=smooth_window,
                        ema_keep=ema_keep,
                        max_gap=max_gap,
                        hit_radius=hit_radius,
                    )
    
                    pts_q, hit_q, hit_cid_q = _grow_path_from_endpoint_bidir(
                        ori_map_b=ori[b],
                        comp_map=comp_map,
                        start_xy=q,
                        init_theta=th2,
                        src_cid=dst_cid,
                        ref_vec=qp_vec,
                        num_bins=num_bins,
                        ignore_val=ignore_val,
                        grow_step=grow_step,
                        grow_max_steps=grow_max_steps,
                        smooth_window=smooth_window,
                        ema_keep=ema_keep,
                        max_gap=max_gap,
                        hit_radius=hit_radius,
                    )
    
                    accepted = False
    
                    if hit_p and hit_cid_p == dst_cid:
                        _draw_path(linked, pts_p, thickness=thickness)
                        accepted = True
                    elif hit_q and hit_cid_q == src_cid:
                        _draw_path(linked, pts_q, thickness=thickness)
                        accepted = True
                    else:
                        meet, ip, iq = _paths_meet(pts_p, pts_q, meet_thr=meet_thr)
                        if meet:
                            pts_merge = pts_p[:ip + 1] + pts_q[:iq + 1][::-1]
                            _draw_path(linked, pts_merge, thickness=thickness)
                            accepted = True
    
                    if (not accepted) and (gap <= line_gap):
                        _draw_quadratic_bezier(linked, p, q, th1, th2, gap, thickness=thickness)
    
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            linked = cv2.morphologyEx(linked, cv2.MORPH_CLOSE, kernel_close)
    
            num2, lab2, st2, _ = cv2.connectedComponentsWithStats(linked, connectivity=8)
            cleaned = np.zeros_like(linked, dtype=np.uint8)
            for cid in range(1, num2):
                if st2[cid, cv2.CC_STAT_AREA] >= min_area:
                    cleaned[lab2 == cid] = 1
    
            out[b, 0] = cleaned
    
        return torch.from_numpy(out).float().to(device)
    '''
    def refine_pseudo_crack_mask_endpoint_bridge(
        self,
        crack_mask: torch.Tensor,   # [B,1,H,W]
        ori_bin_map: torch.Tensor = None,   # [B,H,W]
        num_bins: int = 8,
        min_area: int = 3,
        max_gap: int = 60,
        max_angle_diff: float = 75.0,
        thickness: int = 2,
        trace_len: int = 12,          # endpoint에서 skeleton 따라 추적할 최대 step
        line_gap: int = 20,           # ori_bin_map 없거나 fallback일 때 이 이하 gap은 직선 연결
        bezier_alpha: float = 0.35,   # fallback quadratic bezier
        bezier_alpha_max: float = 20.0,
        bezier_num_pts: int = 30,
        grow_step: float = 1.0,       # orientation-guided growth step
        grow_max_steps: int = 80,     # 각 endpoint에서 최대 성장 step
        hit_radius: int = 2,          # 다른 component 접촉 판정 반경
        smooth_window: int = 5,       # local orientation smoothing window
        meet_thr: float = 3.0,        # 양쪽 path가 만났다고 보는 거리 threshold
        ignore_val: int = 255,
        ema_keep: float = 0.7,        # growth 방향 EMA
    ):
        import cv2
        import numpy as np
        import torch
    
        device = crack_mask.device
        x = (crack_mask > 0.5).detach().cpu().numpy().astype(np.uint8)
        ori = None if ori_bin_map is None else ori_bin_map.detach().cpu().numpy().astype(np.int64)
        out = np.zeros_like(x, dtype=np.uint8)
    
        try:
            from skimage.morphology import skeletonize
            skel_fn = lambda z: skeletonize(z > 0).astype(np.uint8)
        except Exception:
            def skel_fn(z):
                if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
                    return (cv2.ximgproc.thinning((z > 0).astype(np.uint8) * 255) > 0).astype(np.uint8)
                return (z > 0).astype(np.uint8)
    
        def _angle_diff_deg(a, b):
            d = abs(a - b) % np.pi
            d = min(d, np.pi - d)
            return np.degrees(d)
    
        def _line_angle(p, q):
            vx = float(q[0] - p[0])
            vy = float(q[1] - p[1])
            return np.arctan2(vy, vx) % np.pi
    
        def _unit_vec(theta):
            return np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
    
        def _get_neighbors_8(skel, x0, y0):
            H, W = skel.shape
            nbrs = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    xx = x0 + dx
                    yy = y0 + dy
                    if 0 <= xx < W and 0 <= yy < H and skel[yy, xx] > 0:
                        nbrs.append((xx, yy))
            return nbrs
    
        def _trace_from_endpoint(skel, ep_xy, max_steps=12):
            """
            endpoint에서 skeleton을 따라 안쪽으로 tracing
            return: traced points list [(x,y), ...]
            """
            x0, y0 = int(ep_xy[0]), int(ep_xy[1])
    
            if skel[y0, x0] == 0:
                return [(x0, y0)]
    
            path = [(x0, y0)]
            visited = set(path)
    
            nbrs = _get_neighbors_8(skel, x0, y0)
            if len(nbrs) == 0:
                return path
    
            prev = (x0, y0)
            cur = nbrs[0]
            path.append(cur)
            visited.add(cur)
    
            for _ in range(max_steps - 1):
                cx, cy = cur
                cand = _get_neighbors_8(skel, cx, cy)
    
                # 직전 점 제외
                cand = [p for p in cand if p != prev]
    
                if len(cand) == 0:
                    break
    
                if len(cand) == 1:
                    nxt = cand[0]
                else:
                    # 현재 진행 방향과 가장 일치하는 점 선택
                    vx = cur[0] - prev[0]
                    vy = cur[1] - prev[1]
                    vnorm = (vx * vx + vy * vy) ** 0.5 + 1e-6
    
                    best = None
                    best_score = -1e9
                    for cc in cand:
                        wx = cc[0] - cur[0]
                        wy = cc[1] - cur[1]
                        wnorm = (wx * wx + wy * wy) ** 0.5 + 1e-6
                        cos_sim = (vx * wx + vy * wy) / (vnorm * wnorm)
                        if cos_sim > best_score:
                            best_score = cos_sim
                            best = cc
                    nxt = best
    
                if nxt in visited:
                    break
    
                path.append(nxt)
                visited.add(nxt)
                prev, cur = cur, nxt
    
            return path
    
        def _path_tangent_from_endpoint(skel, ep_xy, max_steps=12):
            """
            endpoint에서 skeleton path-following으로 tangent 추정
            return angle in [0, pi)
            """
            path = _trace_from_endpoint(skel, ep_xy, max_steps=max_steps)
    
            if len(path) < 2:
                return None
    
            p0 = np.array(path[0], dtype=np.float32)
            p1 = np.array(path[-1], dtype=np.float32)
    
            v = p1 - p0
            if np.linalg.norm(v) < 1e-6:
                return None
    
            theta = np.arctan2(v[1], v[0]) % np.pi
            return theta
    
        def _draw_quadratic_bezier(mask, p, q, th1, th2, gap, thickness=2):
            """
            endpoint tangent를 반영한 quadratic bezier curve bridge
            p, q: tuple/list (x, y)
            th1, th2: endpoint tangent angle in [0, pi)
            """
            p = np.array(p, dtype=np.float32)
            q = np.array(q, dtype=np.float32)
    
            v1 = _unit_vec(th1)
            v2 = _unit_vec(th2)
    
            alpha = min(gap * bezier_alpha, bezier_alpha_max)
            c1 = p + alpha * v1
            c2 = q - alpha * v2
            c = 0.5 * (c1 + c2)
    
            pts = []
            for t in np.linspace(0.0, 1.0, bezier_num_pts):
                pt = ((1.0 - t) ** 2) * p + 2.0 * (1.0 - t) * t * c + (t ** 2) * q
                pts.append(pt.astype(np.int32))
    
            pts = np.array(pts, dtype=np.int32)
            for k in range(len(pts) - 1):
                pk = tuple(pts[k].tolist())
                qk = tuple(pts[k + 1].tolist())
                cv2.line(mask, pk, qk, color=1, thickness=thickness)
    
        def _extract_endpoints_and_tangents(comp):
            """
            comp: [H,W] uint8 binary
            return dict or None
            """
            skel = skel_fn(comp)
    
            kernel = np.array([[1, 1, 1],
                               [1, 0, 1],
                               [1, 1, 1]], dtype=np.uint8)
            deg = cv2.filter2D(skel.astype(np.uint8), -1, kernel)
            ey, ex = np.where((skel > 0) & (deg == 1))
    
            ys_all, xs_all = np.where(skel > 0)
            if len(xs_all) < 2:
                return None
    
            endpoints = []
            tangents = []
    
            if len(ex) >= 1:
                eps = np.stack([ex, ey], axis=1).astype(np.int32)
    
                for ep in eps:
                    th = _path_tangent_from_endpoint(skel, ep, max_steps=trace_len)
                    if th is not None:
                        endpoints.append(ep)
                        tangents.append(th)
    
            # endpoint를 못 찾았거나 tangent가 안 나오면 fallback
            if len(endpoints) == 0:
                skel_pts = np.stack([xs_all, ys_all], axis=1).astype(np.float32)
    
                center = skel_pts.mean(axis=0, keepdims=True)
                pm = skel_pts - center
                C = (pm.T @ pm) / (len(skel_pts) + 1e-6)
                evals, evecs = np.linalg.eigh(C)
                v = evecs[:, np.argmax(evals)]
                proj = pm @ v
    
                p1 = skel_pts[np.argmin(proj)].astype(np.int32)
                p2 = skel_pts[np.argmax(proj)].astype(np.int32)
                fallback_theta = np.arctan2(v[1], v[0]) % np.pi
    
                for ep in [p1, p2]:
                    endpoints.append(ep)
                    tangents.append(fallback_theta)
    
            endpoints = np.stack(endpoints, axis=0).astype(np.int32)
            tangents = np.array(tangents, dtype=np.float32)
    
            return {
                "mask": comp.astype(np.uint8),
                "skel": skel.astype(np.uint8),
                "endpoints": endpoints,   # [N,2]
                "tangents": tangents,     # [N]
            }
    
        def _bin_to_angle_center(bin_id, num_bins):
            return (bin_id + 0.5) * (np.pi / num_bins)
    
        def _read_local_angle_from_orimap(ori_map_b, x0, y0, num_bins, ignore_val=255, win=5, fallback_theta=None):
            """
            local orientation 평균 (undirected angle)
            """
            H, W = ori_map_b.shape
            r = max(1, win // 2)
    
            vals = []
            for yy in range(max(0, y0 - r), min(H, y0 + r + 1)):
                for xx in range(max(0, x0 - r), min(W, x0 + r + 1)):
                    bid = int(ori_map_b[yy, xx])
                    if bid == ignore_val or bid < 0 or bid >= num_bins:
                        continue
                    vals.append(_bin_to_angle_center(bid, num_bins))
    
            if len(vals) == 0:
                return fallback_theta
    
            vals = np.array(vals, dtype=np.float32)
    
            # undirected angle average with 2θ trick
            c = np.cos(2.0 * vals).mean()
            s = np.sin(2.0 * vals).mean()
            th = 0.5 * np.arctan2(s, c)
            return th % np.pi
    
        def _choose_signed_dir(theta, ref_vec):
            """
            undirected theta -> ±dir 중 ref_vec와 더 잘 맞는 방향 선택
            """
            v1 = _unit_vec(theta)
            v2 = -v1
            if np.dot(v1, ref_vec) >= np.dot(v2, ref_vec):
                return v1
            return v2
    
        def _grow_path_from_endpoint_bidir(
            ori_map_b,
            comp_map,
            start_xy,
            init_theta,
            src_cid,
            ref_vec,
            num_bins,
            ignore_val=255,
            grow_step=1.0,
            grow_max_steps=80,
            smooth_window=5,
            ema_keep=0.7,
            max_gap=60,
            hit_radius=2,
        ):
            """
            endpoint에서 orientation field 따라 path 성장
            return:
                pts: [(x,y), ...]
                hit_other_comp: bool
                hit_comp_id: int or None
            """
            H, W = ori_map_b.shape
            x, y = float(start_xy[0]), float(start_xy[1])
    
            init_dir = _choose_signed_dir(init_theta, ref_vec)
            prev_dir = init_dir.astype(np.float32)
    
            pts = [(int(round(x)), int(round(y)))]
    
            for _ in range(grow_max_steps):
                xi, yi = int(round(x)), int(round(y))
                if not (0 <= xi < W and 0 <= yi < H):
                    break
    
                th = _read_local_angle_from_orimap(
                    ori_map_b,
                    xi, yi,
                    num_bins=num_bins,
                    ignore_val=ignore_val,
                    win=smooth_window,
                    fallback_theta=init_theta,
                )
    
                if th is None:
                    break
    
                cand1 = _unit_vec(th)
                cand2 = -cand1
                if np.dot(cand1, prev_dir) >= np.dot(cand2, prev_dir):
                    dsel = cand1
                else:
                    dsel = cand2
    
                # sudden turn 완화
                new_dir = ema_keep * prev_dir + (1.0 - ema_keep) * dsel
                nrm = np.linalg.norm(new_dir) + 1e-6
                new_dir = new_dir / nrm
    
                xn = x + grow_step * float(new_dir[0])
                yn = y + grow_step * float(new_dir[1])
    
                xni, yni = int(round(xn)), int(round(yn))
                if not (0 <= xni < W and 0 <= yni < H):
                    break
    
                if (xni, yni) != pts[-1]:
                    pts.append((xni, yni))
    
                # 다른 component 접촉 확인
                yy0, yy1 = max(0, yni - hit_radius), min(H, yni + hit_radius + 1)
                xx0, xx1 = max(0, xni - hit_radius), min(W, xni + hit_radius + 1)
    
                touched = comp_map[yy0:yy1, xx0:xx1]
                touched_ids = np.unique(touched[touched > 0])
                touched_ids = touched_ids[touched_ids != src_cid]
    
                if len(touched_ids) > 0:
                    return pts, True, int(touched_ids[0])
    
                x, y = xn, yn
                prev_dir = new_dir
    
                if np.hypot(x - start_xy[0], y - start_xy[1]) > max_gap:
                    break
    
            return pts, False, None
    
        def _paths_meet(pts1, pts2, meet_thr=3.0):
            """
            두 path가 가까워졌는지 검사
            return:
                meet: bool
                i_best, j_best
            """
            if len(pts1) == 0 or len(pts2) == 0:
                return False, None, None
    
            A = np.array(pts1, dtype=np.float32)
            B = np.array(pts2, dtype=np.float32)
    
            d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)
            idx = np.argmin(d2)
            i_best, j_best = np.unravel_index(idx, d2.shape)
    
            if np.sqrt(d2[i_best, j_best]) <= meet_thr:
                return True, int(i_best), int(j_best)
    
            return False, None, None
    
        def _draw_path(mask, pts, thickness=2):
            if len(pts) == 0:
                return
            if len(pts) == 1:
                cv2.circle(mask, pts[0], radius=max(1, thickness // 2), color=1, thickness=-1)
                return
            for k in range(len(pts) - 1):
                cv2.line(mask, pts[k], pts[k + 1], color=1, thickness=thickness)
    
        for b in range(x.shape[0]):
            m = x[b, 0].copy()
    
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    
            comps = []
            for cid in range(1, num_labels):
                area = stats[cid, cv2.CC_STAT_AREA]
                if area < min_area:
                    continue
    
                comp = (labels == cid).astype(np.uint8)
                info = _extract_endpoints_and_tangents(comp)
                if info is None:
                    continue
                comps.append(info)
    
            if len(comps) == 0:
                out[b, 0] = m
                continue
    
            linked = np.zeros_like(m, dtype=np.uint8)
            for c in comps:
                linked = np.maximum(linked, c["mask"])
    
            # component map 재구성
            comp_map = np.zeros_like(m, dtype=np.int32)
            cid_acc = 1
            for c in comps:
                comp_map[c["mask"] > 0] = cid_acc
                cid_acc += 1
    
            # pairwise bridge
            for i in range(len(comps)):
                for j in range(i + 1, len(comps)):
                    ep1 = comps[i]["endpoints"]
                    tg1 = comps[i]["tangents"]
                    ep2 = comps[j]["endpoints"]
                    tg2 = comps[j]["tangents"]
    
                    best = None
                    best_score = 1e9
    
                    for ii, p in enumerate(ep1):
                        for jj, q in enumerate(ep2):
                            d = np.linalg.norm(p - q)
                            if d > max_gap:
                                continue
    
                            theta_pq = _line_angle(p, q)
                            theta_qp = _line_angle(q, p)
    
                            a1 = _angle_diff_deg(tg1[ii], theta_pq)
                            a2 = _angle_diff_deg(tg2[jj], theta_qp)
    
                            # 둘 다 많이 틀릴 때만 reject
                            if a1 > max_angle_diff and a2 > max_angle_diff:
                                continue
    
                            score = d + 0.20 * (a1 + a2)
    
                            if score < best_score:
                                best_score = score
                                best = {
                                    "p": tuple(p.tolist()),
                                    "q": tuple(q.tolist()),
                                    "ii": ii,
                                    "jj": jj,
                                    "d": float(d),
                                    "a1": float(a1),
                                    "a2": float(a2),
                                    "src_comp_idx": i,
                                    "dst_comp_idx": j,
                                }
    
                    if best is None:
                        continue
    
                    p = best["p"]
                    q = best["q"]
                    ii = best["ii"]
                    jj = best["jj"]
                    gap = best["d"]
                    th1 = tg1[ii]
                    th2 = tg2[jj]
    
                    src_idx = best["src_comp_idx"]
                    dst_idx = best["dst_comp_idx"]
                    src_cid = src_idx + 1
                    dst_cid = dst_idx + 1
    
                    # ori map이 없으면 기존 fallback
                    if ori is None:
                        if gap <= line_gap:
                            cv2.line(linked, p, q, color=1, thickness=thickness)
                        else:
                            _draw_quadratic_bezier(
                                linked, p, q, th1, th2, gap, thickness=thickness
                            )
                        continue
    
                    # 양쪽 endpoint에서 동시에 성장
                    pq_vec = np.array([q[0] - p[0], q[1] - p[1]], dtype=np.float32)
                    pq_norm = np.linalg.norm(pq_vec) + 1e-6
                    pq_vec = pq_vec / pq_norm
                    qp_vec = -pq_vec
    
                    pts_p, hit_p, hit_cid_p = _grow_path_from_endpoint_bidir(
                        ori_map_b=ori[b],
                        comp_map=comp_map,
                        start_xy=p,
                        init_theta=th1,
                        src_cid=src_cid,
                        ref_vec=pq_vec,
                        num_bins=num_bins,
                        ignore_val=ignore_val,
                        grow_step=grow_step,
                        grow_max_steps=grow_max_steps,
                        smooth_window=smooth_window,
                        ema_keep=ema_keep,
                        max_gap=max_gap,
                        hit_radius=hit_radius,
                    )
    
                    pts_q, hit_q, hit_cid_q = _grow_path_from_endpoint_bidir(
                        ori_map_b=ori[b],
                        comp_map=comp_map,
                        start_xy=q,
                        init_theta=th2,
                        src_cid=dst_cid,
                        ref_vec=qp_vec,
                        num_bins=num_bins,
                        ignore_val=ignore_val,
                        grow_step=grow_step,
                        grow_max_steps=grow_max_steps,
                        smooth_window=smooth_window,
                        ema_keep=ema_keep,
                        max_gap=max_gap,
                        hit_radius=hit_radius,
                    )
    
                    accepted = False
    
                    # case 1: p path가 dst component에 직접 닿음
                    if hit_p and hit_cid_p == dst_cid:
                        _draw_path(linked, pts_p, thickness=thickness)
                        accepted = True
    
                    # case 2: q path가 src component에 직접 닿음
                    elif hit_q and hit_cid_q == src_cid:
                        _draw_path(linked, pts_q, thickness=thickness)
                        accepted = True
    
                    # case 3: 두 path가 중간에서 만남
                    else:
                        meet, ip, iq = _paths_meet(pts_p, pts_q, meet_thr=meet_thr)
                        if meet:
                            pts_merge = pts_p[:ip + 1] + pts_q[:iq + 1][::-1]
                            _draw_path(linked, pts_merge, thickness=thickness)
                            accepted = True
    
                    # optional fallback: 매우 짧은 gap은 bezier 허용
                    if (not accepted) and (gap <= line_gap):
                        _draw_quadratic_bezier(
                            linked, p, q, th1, th2, gap, thickness=thickness
                        )
    
            # optional light closing after bridge
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            linked = cv2.morphologyEx(linked, cv2.MORPH_CLOSE, kernel_close)
    
            # final cleanup
            num2, lab2, st2, _ = cv2.connectedComponentsWithStats(linked, connectivity=8)
            cleaned = np.zeros_like(linked, dtype=np.uint8)
            for cid in range(1, num2):
                if st2[cid, cv2.CC_STAT_AREA] >= min_area:
                    cleaned[lab2 == cid] = 1
    
            out[b, 0] = cleaned
    
        return torch.from_numpy(out).float().to(device)
    def refine_pseudo_crack_mask_endpoint_bridge(
        self,
        crack_mask: torch.Tensor,   # [B,1,H,W]
        ori_bin_map: torch.Tensor = None,   # [B,H,W]  <- ori_field_tgt 넣는 것을 권장
        num_bins: int = 8,
        min_area: int = 3,
        max_gap: int = 60,
        max_angle_diff: float = 75.0,
        thickness: int = 2,
        trace_len: int = 12,
        line_gap: int = 20,
        bezier_alpha: float = 0.35,
        bezier_alpha_max: float = 20.0,
        bezier_num_pts: int = 40,
        grow_step: float = 1.0,
        grow_max_steps: int = 80,
        hit_radius: int = 2,
        smooth_window: int = 5,
        meet_thr: float = 3.0,
        ignore_val: int = 255,
        ema_keep: float = 0.7,
    
        # new
        corridor_radius: float = 8.0,      # prior curve corridor
        prior_pull: float = 0.35,          # corridor 밖으로 벗어나려 할 때 prior tangent로 끌어주는 정도
        soft_accept_ratio: float = 0.72,   # hit/meet 안 되어도 endpoint distance가 충분히 줄면 accept
        max_tortuosity: float = 1.75,      # 너무 꼬불꼬불한 path 방지
        fallback_gap_ratio: float = 1.0,   # 최종 cubic fallback 허용 범위
    ):
        import cv2
        import numpy as np
        import torch
    
        device = crack_mask.device
        x = (crack_mask > 0.5).detach().cpu().numpy().astype(np.uint8)
        ori = None if ori_bin_map is None else ori_bin_map.detach().cpu().numpy().astype(np.int64)
        out = np.zeros_like(x, dtype=np.uint8)
    
        try:
            from skimage.morphology import skeletonize
            skel_fn = lambda z: skeletonize(z > 0).astype(np.uint8)
        except Exception:
            def skel_fn(z):
                if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
                    return (cv2.ximgproc.thinning((z > 0).astype(np.uint8) * 255) > 0).astype(np.uint8)
                return (z > 0).astype(np.uint8)
    
        def _angle_diff_deg(a, b):
            d = abs(a - b) % np.pi
            d = min(d, np.pi - d)
            return np.degrees(d)
    
        def _line_angle(p, q):
            vx = float(q[0] - p[0])
            vy = float(q[1] - p[1])
            return np.arctan2(vy, vx) % np.pi
    
        def _unit_vec(theta):
            return np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
    
        def _safe_norm(v):
            return float(np.linalg.norm(v) + 1e-6)
    
        def _get_neighbors_8(skel, x0, y0):
            H, W = skel.shape
            nbrs = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    xx = x0 + dx
                    yy = y0 + dy
                    if 0 <= xx < W and 0 <= yy < H and skel[yy, xx] > 0:
                        nbrs.append((xx, yy))
            return nbrs
    
        def _trace_from_endpoint(skel, ep_xy, max_steps=12):
            x0, y0 = int(ep_xy[0]), int(ep_xy[1])
    
            if skel[y0, x0] == 0:
                return [(x0, y0)]
    
            path = [(x0, y0)]
            visited = set(path)
    
            nbrs = _get_neighbors_8(skel, x0, y0)
            if len(nbrs) == 0:
                return path
    
            prev = (x0, y0)
            cur = nbrs[0]
            path.append(cur)
            visited.add(cur)
    
            for _ in range(max_steps - 1):
                cx, cy = cur
                cand = _get_neighbors_8(skel, cx, cy)
                cand = [p for p in cand if p != prev]
    
                if len(cand) == 0:
                    break
    
                if len(cand) == 1:
                    nxt = cand[0]
                else:
                    vx = cur[0] - prev[0]
                    vy = cur[1] - prev[1]
                    vnorm = (vx * vx + vy * vy) ** 0.5 + 1e-6
    
                    best = None
                    best_score = -1e9
                    for cc in cand:
                        wx = cc[0] - cur[0]
                        wy = cc[1] - cur[1]
                        wnorm = (wx * wx + wy * wy) ** 0.5 + 1e-6
                        cos_sim = (vx * wx + vy * wy) / (vnorm * wnorm)
                        if cos_sim > best_score:
                            best_score = cos_sim
                            best = cc
                    nxt = best
    
                if nxt in visited:
                    break
    
                path.append(nxt)
                visited.add(nxt)
                prev, cur = cur, nxt
    
            return path
    
        def _path_tangent_from_endpoint(skel, ep_xy, max_steps=12):
            path = _trace_from_endpoint(skel, ep_xy, max_steps=max_steps)
            if len(path) < 2:
                return None
    
            p0 = np.array(path[0], dtype=np.float32)
            p1 = np.array(path[-1], dtype=np.float32)
            v = p1 - p0
            if np.linalg.norm(v) < 1e-6:
                return None
    
            theta = np.arctan2(v[1], v[0]) % np.pi
            return theta
    
        def _extract_endpoints_and_tangents(comp):
            skel = skel_fn(comp)
    
            kernel = np.array([[1, 1, 1],
                               [1, 0, 1],
                               [1, 1, 1]], dtype=np.uint8)
            deg = cv2.filter2D(skel.astype(np.uint8), -1, kernel)
            ey, ex = np.where((skel > 0) & (deg == 1))
    
            ys_all, xs_all = np.where(skel > 0)
            if len(xs_all) < 2:
                return None
    
            endpoints = []
            tangents = []
    
            if len(ex) >= 1:
                eps = np.stack([ex, ey], axis=1).astype(np.int32)
                for ep in eps:
                    th = _path_tangent_from_endpoint(skel, ep, max_steps=trace_len)
                    if th is not None:
                        endpoints.append(ep)
                        tangents.append(th)
    
            if len(endpoints) == 0:
                skel_pts = np.stack([xs_all, ys_all], axis=1).astype(np.float32)
                center = skel_pts.mean(axis=0, keepdims=True)
                pm = skel_pts - center
                C = (pm.T @ pm) / (len(skel_pts) + 1e-6)
                evals, evecs = np.linalg.eigh(C)
                v = evecs[:, np.argmax(evals)]
                proj = pm @ v
    
                p1 = skel_pts[np.argmin(proj)].astype(np.int32)
                p2 = skel_pts[np.argmax(proj)].astype(np.int32)
                fallback_theta = np.arctan2(v[1], v[0]) % np.pi
    
                for ep in [p1, p2]:
                    endpoints.append(ep)
                    tangents.append(fallback_theta)
    
            endpoints = np.stack(endpoints, axis=0).astype(np.int32)
            tangents = np.array(tangents, dtype=np.float32)
    
            return {
                "mask": comp.astype(np.uint8),
                "skel": skel.astype(np.uint8),
                "endpoints": endpoints,
                "tangents": tangents,
            }
    
        def _bin_to_angle_center(bin_id, num_bins):
            return (bin_id + 0.5) * (np.pi / num_bins)
    
        def _read_local_angle_from_orimap(ori_map_b, x0, y0, num_bins, ignore_val=255, win=5, fallback_theta=None):
            H, W = ori_map_b.shape
            r = max(1, win // 2)
    
            vals = []
            for yy in range(max(0, y0 - r), min(H, y0 + r + 1)):
                for xx in range(max(0, x0 - r), min(W, x0 + r + 1)):
                    bid = int(ori_map_b[yy, xx])
                    if bid == ignore_val or bid < 0 or bid >= num_bins:
                        continue
                    vals.append(_bin_to_angle_center(bid, num_bins))
    
            if len(vals) == 0:
                return fallback_theta
    
            vals = np.array(vals, dtype=np.float32)
            c = np.cos(2.0 * vals).mean()
            s = np.sin(2.0 * vals).mean()
            th = 0.5 * np.arctan2(s, c)
            return th % np.pi
    
        def _choose_signed_dir(theta, ref_vec):
            v1 = _unit_vec(theta)
            v2 = -v1
            if np.dot(v1, ref_vec) >= np.dot(v2, ref_vec):
                return v1
            return v2
    
        def _build_cubic_bezier_prior(p, q, th1, th2, gap, num_pts=40):
            p = np.array(p, dtype=np.float32)
            q = np.array(q, dtype=np.float32)
    
            v1 = _unit_vec(th1)
            v2 = _unit_vec(th2)
    
            alpha = min(gap * bezier_alpha, bezier_alpha_max)
            c1 = p + alpha * v1
            c2 = q - alpha * v2
    
            pts = []
            ts = np.linspace(0.0, 1.0, num_pts)
            for t in ts:
                pt = ((1 - t) ** 3) * p \
                     + 3 * ((1 - t) ** 2) * t * c1 \
                     + 3 * (1 - t) * (t ** 2) * c2 \
                     + (t ** 3) * q
                pts.append(pt)
            return np.array(pts, dtype=np.float32)
    
        def _draw_polyline(mask, pts, thickness=2):
            if len(pts) == 0:
                return
            if len(pts) == 1:
                pp = tuple(np.round(pts[0]).astype(np.int32).tolist())
                cv2.circle(mask, pp, radius=max(1, thickness // 2), color=1, thickness=-1)
                return
            pts_i = np.round(pts).astype(np.int32)
            for k in range(len(pts_i) - 1):
                cv2.line(mask, tuple(pts_i[k].tolist()), tuple(pts_i[k + 1].tolist()), color=1, thickness=thickness)
    
        def _nearest_prior_info(pt_xy, prior_pts):
            p = np.array(pt_xy, dtype=np.float32)
            D = ((prior_pts - p[None, :]) ** 2).sum(axis=1)
            idx = int(np.argmin(D))
            dist = float(np.sqrt(D[idx]))
            if idx == 0:
                tangent = prior_pts[1] - prior_pts[0]
            elif idx == len(prior_pts) - 1:
                tangent = prior_pts[-1] - prior_pts[-2]
            else:
                tangent = prior_pts[idx + 1] - prior_pts[idx - 1]
            nrm = _safe_norm(tangent)
            tangent = tangent / nrm
            return dist, idx, tangent
    
        def _path_length(pts):
            if len(pts) < 2:
                return 0.0
            arr = np.array(pts, dtype=np.float32)
            return float(np.linalg.norm(arr[1:] - arr[:-1], axis=1).sum())
    
        def _tortuosity_ok(pts, start_xy, end_xy, max_tortuosity=1.75):
            if len(pts) < 2:
                return True
            plen = _path_length(pts)
            chord = float(np.linalg.norm(np.array(end_xy, dtype=np.float32) - np.array(start_xy, dtype=np.float32)) + 1e-6)
            return (plen / chord) <= max_tortuosity
    
        def _paths_meet(pts1, pts2, meet_thr=3.0):
            if len(pts1) == 0 or len(pts2) == 0:
                return False, None, None
    
            A = np.array(pts1, dtype=np.float32)
            B = np.array(pts2, dtype=np.float32)
            d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)
            idx = np.argmin(d2)
            i_best, j_best = np.unravel_index(idx, d2.shape)
    
            if np.sqrt(d2[i_best, j_best]) <= meet_thr:
                return True, int(i_best), int(j_best)
            return False, None, None
    
        def _grow_path_from_endpoint_bidir(
            ori_map_b,
            comp_map,
            start_xy,
            init_theta,
            src_cid,
            ref_vec,
            num_bins,
            prior_pts=None,
            corridor_radius=8.0,
            prior_pull=0.35,
            ignore_val=255,
            grow_step=1.0,
            grow_max_steps=80,
            smooth_window=5,
            ema_keep=0.7,
            max_gap=60,
            hit_radius=2,
        ):
            H, W = ori_map_b.shape
            x, y = float(start_xy[0]), float(start_xy[1])
    
            init_dir = _choose_signed_dir(init_theta, ref_vec)
            prev_dir = init_dir.astype(np.float32)
    
            pts = [(int(round(x)), int(round(y)))]
    
            for _ in range(grow_max_steps):
                xi, yi = int(round(x)), int(round(y))
                if not (0 <= xi < W and 0 <= yi < H):
                    break
    
                th = _read_local_angle_from_orimap(
                    ori_map_b,
                    xi, yi,
                    num_bins=num_bins,
                    ignore_val=ignore_val,
                    win=smooth_window,
                    fallback_theta=init_theta,
                )
    
                if th is None:
                    break
    
                cand1 = _unit_vec(th)
                cand2 = -cand1
                dsel = cand1 if np.dot(cand1, prev_dir) >= np.dot(cand2, prev_dir) else cand2
    
                # prior curve corridor guidance
                if prior_pts is not None and len(prior_pts) >= 2:
                    dist_to_prior, _, prior_tangent = _nearest_prior_info((x, y), prior_pts)
    
                    # corridor 안에서는 orientation 위주, 밖에서는 prior tangent를 더 섞음
                    if dist_to_prior > corridor_radius:
                        w_prior = min(0.85, prior_pull + 0.10 * (dist_to_prior - corridor_radius))
                    else:
                        w_prior = 0.15
    
                    dmix = (1.0 - w_prior) * dsel + w_prior * prior_tangent
                    dmix = dmix / (_safe_norm(dmix))
                else:
                    dmix = dsel
    
                new_dir = ema_keep * prev_dir + (1.0 - ema_keep) * dmix
                new_dir = new_dir / (_safe_norm(new_dir))
    
                xn = x + grow_step * float(new_dir[0])
                yn = y + grow_step * float(new_dir[1])
    
                xni, yni = int(round(xn)), int(round(yn))
                if not (0 <= xni < W and 0 <= yni < H):
                    break
    
                if (xni, yni) != pts[-1]:
                    pts.append((xni, yni))
    
                yy0, yy1 = max(0, yni - hit_radius), min(H, yni + hit_radius + 1)
                xx0, xx1 = max(0, xni - hit_radius), min(W, xni + hit_radius + 1)
    
                touched = comp_map[yy0:yy1, xx0:xx1]
                touched_ids = np.unique(touched[touched > 0])
                touched_ids = touched_ids[touched_ids != src_cid]
    
                if len(touched_ids) > 0:
                    return pts, True, int(touched_ids[0])
    
                x, y = xn, yn
                prev_dir = new_dir
    
                if np.hypot(x - start_xy[0], y - start_xy[1]) > max_gap:
                    break
    
            return pts, False, None
    
        for b in range(x.shape[0]):
            m = x[b, 0].copy()
    
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    
            comps = []
            for cid in range(1, num_labels):
                area = stats[cid, cv2.CC_STAT_AREA]
                if area < min_area:
                    continue
    
                comp = (labels == cid).astype(np.uint8)
                info = _extract_endpoints_and_tangents(comp)
                if info is None:
                    continue
                comps.append(info)
    
            if len(comps) == 0:
                out[b, 0] = m
                continue
    
            linked = np.zeros_like(m, dtype=np.uint8)
            for c in comps:
                linked = np.maximum(linked, c["mask"])
    
            comp_map = np.zeros_like(m, dtype=np.int32)
            cid_acc = 1
            for c in comps:
                comp_map[c["mask"] > 0] = cid_acc
                cid_acc += 1
    
            for i in range(len(comps)):
                for j in range(i + 1, len(comps)):
                    ep1 = comps[i]["endpoints"]
                    tg1 = comps[i]["tangents"]
                    ep2 = comps[j]["endpoints"]
                    tg2 = comps[j]["tangents"]
    
                    best = None
                    best_score = 1e9
    
                    for ii, p in enumerate(ep1):
                        for jj, q in enumerate(ep2):
                            d = np.linalg.norm(p - q)
                            if d > max_gap:
                                continue
    
                            theta_pq = _line_angle(p, q)
                            theta_qp = _line_angle(q, p)
    
                            a1 = _angle_diff_deg(tg1[ii], theta_pq)
                            a2 = _angle_diff_deg(tg2[jj], theta_qp)
    
                            # 너무 엄격하지 않게: 둘 다 매우 틀릴 때만 reject
                            if a1 > max_angle_diff and a2 > max_angle_diff:
                                continue
    
                            score = d + 0.18 * (a1 + a2)
                            if score < best_score:
                                best_score = score
                                best = {
                                    "p": tuple(p.tolist()),
                                    "q": tuple(q.tolist()),
                                    "ii": ii,
                                    "jj": jj,
                                    "d": float(d),
                                    "a1": float(a1),
                                    "a2": float(a2),
                                    "src_comp_idx": i,
                                    "dst_comp_idx": j,
                                }
    
                    if best is None:
                        continue
    
                    p = best["p"]
                    q = best["q"]
                    ii = best["ii"]
                    jj = best["jj"]
                    gap = best["d"]
                    th1 = tg1[ii]
                    th2 = tg2[jj]
    
                    src_idx = best["src_comp_idx"]
                    dst_idx = best["dst_comp_idx"]
                    src_cid = src_idx + 1
                    dst_cid = dst_idx + 1
    
                    # natural prior curve (always available)
                    prior_pts = _build_cubic_bezier_prior(p, q, th1, th2, gap, num_pts=bezier_num_pts)
    
                    # orientation이 없으면 cubic prior로 연결
                    if ori is None:
                        _draw_polyline(linked, prior_pts, thickness=thickness)
                        continue
    
                    pq_vec = np.array([q[0] - p[0], q[1] - p[1]], dtype=np.float32)
                    pq_vec = pq_vec / (_safe_norm(pq_vec))
                    qp_vec = -pq_vec
    
                    pts_p, hit_p, hit_cid_p = _grow_path_from_endpoint_bidir(
                        ori_map_b=ori[b],
                        comp_map=comp_map,
                        start_xy=p,
                        init_theta=th1,
                        src_cid=src_cid,
                        ref_vec=pq_vec,
                        num_bins=num_bins,
                        prior_pts=prior_pts,
                        corridor_radius=corridor_radius,
                        prior_pull=prior_pull,
                        ignore_val=ignore_val,
                        grow_step=grow_step,
                        grow_max_steps=grow_max_steps,
                        smooth_window=smooth_window,
                        ema_keep=ema_keep,
                        max_gap=max_gap,
                        hit_radius=hit_radius,
                    )
    
                    pts_q, hit_q, hit_cid_q = _grow_path_from_endpoint_bidir(
                        ori_map_b=ori[b],
                        comp_map=comp_map,
                        start_xy=q,
                        init_theta=th2,
                        src_cid=dst_cid,
                        ref_vec=qp_vec,
                        num_bins=num_bins,
                        prior_pts=prior_pts[::-1].copy(),
                        corridor_radius=corridor_radius,
                        prior_pull=prior_pull,
                        ignore_val=ignore_val,
                        grow_step=grow_step,
                        grow_max_steps=grow_max_steps,
                        smooth_window=smooth_window,
                        ema_keep=ema_keep,
                        max_gap=max_gap,
                        hit_radius=hit_radius,
                    )
    
                    accepted = False
    
                    # 1) direct hit
                    if hit_p and hit_cid_p == dst_cid and _tortuosity_ok(pts_p, p, q, max_tortuosity=max_tortuosity):
                        _draw_polyline(linked, np.array(pts_p, dtype=np.float32), thickness=thickness)
                        accepted = True
    
                    elif hit_q and hit_cid_q == src_cid and _tortuosity_ok(pts_q, q, p, max_tortuosity=max_tortuosity):
                        _draw_polyline(linked, np.array(pts_q, dtype=np.float32), thickness=thickness)
                        accepted = True
    
                    # 2) meet in the middle
                    else:
                        meet, ip, iq = _paths_meet(pts_p, pts_q, meet_thr=meet_thr)
                        if meet:
                            pts_merge = pts_p[:ip + 1] + pts_q[:iq + 1][::-1]
                            if _tortuosity_ok(pts_merge, p, q, max_tortuosity=max_tortuosity):
                                _draw_polyline(linked, np.array(pts_merge, dtype=np.float32), thickness=thickness)
                                accepted = True
    
                    # 3) soft accept: hit/meet는 아니어도 충분히 가까워졌으면 accept
                    if not accepted and len(pts_p) > 0 and len(pts_q) > 0:
                        end_p = np.array(pts_p[-1], dtype=np.float32)
                        end_q = np.array(pts_q[-1], dtype=np.float32)
                        rem_dist = float(np.linalg.norm(end_p - end_q))
                        if rem_dist <= soft_accept_ratio * gap:
                            pts_merge = pts_p + pts_q[::-1]
                            if _tortuosity_ok(pts_merge, p, q, max_tortuosity=max_tortuosity):
                                _draw_polyline(linked, np.array(pts_merge, dtype=np.float32), thickness=thickness)
                                accepted = True
    
                    # 4) final fallback: 너무 엄격해서 안 이어지는 것 방지
                    #    직선이 아니라 cubic prior를 사용
                    if (not accepted) and (gap <= fallback_gap_ratio * max_gap):
                        _draw_polyline(linked, prior_pts, thickness=thickness)
    
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            linked = cv2.morphologyEx(linked, cv2.MORPH_CLOSE, kernel_close)
    
            num2, lab2, st2, _ = cv2.connectedComponentsWithStats(linked, connectivity=8)
            cleaned = np.zeros_like(linked, dtype=np.uint8)
            for cid in range(1, num2):
                if st2[cid, cv2.CC_STAT_AREA] >= min_area:
                    cleaned[lab2 == cid] = 1
    
            out[b, 0] = cleaned
    
        return torch.from_numpy(out).float().to(device)
    def save_pseudo_refine_compare(
        self,
        img,                    # [B,C,H,W]
        pseudo_lbl_raw,         # [B,1,H,W] or [B,H,W]
        crack_mask_refined,     # [B,1,H,W]
        pseudo_lbl_refined,     # [B,1,H,W] or [B,H,W]
        save_path,
        max_items=4,
        crack_class=1,
    ):
        import os
        import numpy as np
        import matplotlib.pyplot as plt
    
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
        B = min(img.size(0), max_items)
    
        fig, axes = plt.subplots(B, 4, figsize=(16, 4 * B))
        if B == 1:
            axes = np.expand_dims(axes, axis=0)
    
        for i in range(B):
            # image
            x = img[i].detach().cpu()
            x = x.permute(1, 2, 0).numpy()
    
            # normalize for visualization if needed
            if x.dtype != np.uint8:
                x_min, x_max = x.min(), x.max()
                if x_max > x_min:
                    x = (x - x_min) / (x_max - x_min + 1e-8)
                x = np.clip(x, 0, 1)
    
            # raw pseudo
            pr = pseudo_lbl_raw[i]
            if pr.dim() == 3:
                pr = pr[0]
            pr = pr.detach().cpu().numpy()
    
            # refined crack mask
            cmr = crack_mask_refined[i, 0].detach().cpu().numpy()
    
            # refined pseudo
            pf = pseudo_lbl_refined[i]
            if pf.dim() == 3:
                pf = pf[0]
            pf = pf.detach().cpu().numpy()
            cmr_check = (cmr > 0.5).astype(np.float32)
            # crack-only binary maps for easier comparison
            pf_bin = (pf == crack_class).astype(np.float32)
            pr_bin = (pr == crack_class).astype(np.uint8)
            print(f"[sample {i}]")
            print("  raw unique:", np.unique(pr))
            print("  refined unique:", np.unique(pf))
            print("  raw crack pixels:", pr_bin.sum())
            print("  refined mask pixels:", cmr_check.sum())
            print("  refined pseudo crack pixels:", pf_bin.sum())
            axes[i, 0].imshow(x)
            axes[i, 0].set_title("Image")
    
            axes[i, 1].imshow(pr_bin, cmap="gray")
            axes[i, 1].set_title("Raw Pseudo Crack")
    
            axes[i, 2].imshow(cmr, cmap="gray")
            axes[i, 2].set_title("Refined Crack Mask")
    
            axes[i, 3].imshow(pf_bin, cmap="gray")
            axes[i, 3].set_title("Refined Pseudo Crack")
    
            for j in range(4):
                axes[i, j].axis("off")
    
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    def build_refined_pseudo_label(self, pseudo_lbl, crack_mask_refined, crack_class=1):
        """
        pseudo_lbl: [B,1,H,W] or [B,H,W], long
        crack_mask_refined: [B,1,H,W], float(0/1)
        return: same shape as pseudo_lbl
        """
        import torch
    
        if pseudo_lbl.dim() == 3:
            pseudo = pseudo_lbl.unsqueeze(1).clone()
            squeeze_back = True
        else:
            pseudo = pseudo_lbl.clone()
            squeeze_back = False
    
        # 기존 crack class 지우기
        pseudo[pseudo == crack_class] = 0
    
        # refined crack 넣기
        pseudo[crack_mask_refined > 0.5] = crack_class
    
        if squeeze_back:
            pseudo = pseudo.squeeze(1)
    
        return pseudo.long()
    def _componentwise_ori_bin_local_tangent(
        self,
        crack_mask: torch.Tensor,   # [B,1,H,W] float/bool
        num_bins: int,
        min_pixels: int = 20,
        ignore_val: int = 255,
        do_close: bool = False,
        close_ksize: int = 3,
        tangent_radius: int = 7,      # local tangent 추정 반경
        min_tangent_pts: int = 5,     # local window 내 최소 점 개수
        fill_to_mask: bool = True,    # skeleton bin을 crack mask 전체로 전파
    ):
        """
        return: ori_bin_map [B,H,W] (torch.long)
                crack 픽셀: 0 ~ num_bins-1
                non-crack: ignore_val
    
        아이디어:
        1) connected component 분리
        2) 각 component skeletonize
        3) skeleton 각 픽셀 주변 local neighborhood에 대해 PCA -> tangent 방향 계산
        4) skeleton bin map 생성
        5) 필요하면 crack mask 전체 픽셀로 nearest skeleton bin 전파
        """
        import numpy as np
        import cv2
        import torch
    
        B, _, H, W = crack_mask.shape
        device = crack_mask.device
        out = torch.full((B, H, W), ignore_val, dtype=torch.long, device=device)
    
        m = (crack_mask > 0.5).detach().cpu().numpy().astype(np.uint8)
    
        # connected components
        try:
            from scipy.ndimage import label as ndi_label
            label_fn = lambda x: ndi_label(x, structure=np.ones((3, 3), dtype=np.uint8))[0]
        except Exception:
            from skimage.measure import label as sk_label
            label_fn = lambda x: sk_label(x, connectivity=2)
    
        # skeletonize
        try:
            from skimage.morphology import skeletonize
            skel_fn = lambda x: skeletonize(x > 0).astype(np.uint8)
        except Exception:
            def skel_fn(x):
                if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
                    return (cv2.ximgproc.thinning((x > 0).astype(np.uint8) * 255) > 0).astype(np.uint8)
                return (x > 0).astype(np.uint8)
    
        if do_close:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
            for b in range(B):
                m[b, 0] = cv2.morphologyEx(m[b, 0], cv2.MORPH_CLOSE, k)
    
        def angle_to_bin(angle, num_bins):
            # undirected orientation: [0, pi)
            angle = angle % np.pi
            bin_id = int(np.floor(angle / (np.pi / num_bins)))
            return min(max(bin_id, 0), num_bins - 1)
    
        for b in range(B):
            comp = label_fn(m[b, 0])
            if comp.max() == 0:
                continue
    
            bin_map = np.full((H, W), ignore_val, dtype=np.int64)
    
            for cid in range(1, int(comp.max()) + 1):
                comp_mask = (comp == cid).astype(np.uint8)
                ys, xs = np.where(comp_mask > 0)
                if len(xs) < min_pixels:
                    continue
    
                skel = skel_fn(comp_mask)
                yk, xk = np.where(skel > 0)
                if len(xk) < 2:
                    continue
    
                # skeleton 좌표 목록
                skel_pts = np.stack([xk, yk], axis=1).astype(np.int32)   # [N,2]
                skel_bin_map = np.full((H, W), ignore_val, dtype=np.int64)
    
                # 각 skeleton 픽셀마다 local tangent 계산
                for x0, y0 in skel_pts:
                    dx = xk - x0
                    dy = yk - y0
                    dist2 = dx * dx + dy * dy
                    sel = dist2 <= (tangent_radius * tangent_radius)
    
                    if sel.sum() < min_tangent_pts:
                        continue
    
                    P = np.stack([xk[sel], yk[sel]], axis=1).astype(np.float32)  # [M,2]
                    Pm = P - P.mean(axis=0, keepdims=True)
                    C = (Pm.T @ Pm) / (len(P) + 1e-6)
    
                    evals, evecs = np.linalg.eigh(C)
                    v = evecs[:, np.argmax(evals)]   # principal axis (vx, vy)
    
                    angle = np.arctan2(v[1], v[0])   # [-pi, pi]
                    bin_id = angle_to_bin(angle, num_bins)
                    skel_bin_map[y0, x0] = bin_id
    
                if not fill_to_mask:
                    # skeleton pixel에만 orientation 부여
                    valid = (skel_bin_map != ignore_val)
                    bin_map[valid] = skel_bin_map[valid]
                    continue
    
                # -----------------------------
                # skeleton orientation을 component 전체 픽셀로 전파
                # nearest skeleton pixel의 bin 사용
                # -----------------------------
                valid_y, valid_x = np.where(skel_bin_map != ignore_val)
                if len(valid_x) == 0:
                    continue
    
                valid_bins = skel_bin_map[valid_y, valid_x]
                valid_pts = np.stack([valid_x, valid_y], axis=1).astype(np.float32)  # [K,2]
                comp_pts = np.stack([xs, ys], axis=1).astype(np.float32)              # [M,2]
    
                # scipy KDTree 있으면 사용
                assigned_bins = None
                try:
                    from scipy.spatial import cKDTree
                    tree = cKDTree(valid_pts)
                    _, nn_idx = tree.query(comp_pts, k=1)
                    assigned_bins = valid_bins[nn_idx]
                except Exception:
                    # fallback: brute force
                    assigned_bins = np.empty((len(comp_pts),), dtype=np.int64)
                    chunk = 2048
                    for s in range(0, len(comp_pts), chunk):
                        e = min(s + chunk, len(comp_pts))
                        q = comp_pts[s:e]  # [q,2]
                        d2 = ((q[:, None, :] - valid_pts[None, :, :]) ** 2).sum(axis=2)
                        nn_idx = d2.argmin(axis=1)
                        assigned_bins[s:e] = valid_bins[nn_idx]
    
                bin_map[ys, xs] = assigned_bins
    
            out[b] = torch.from_numpy(bin_map).to(device)
    
        return out
    def compute_orientations(self, crack_mask, num_bins=8):
        return self._componentwise_ori_bin_local_tangent(
            crack_mask=crack_mask,
            num_bins=num_bins,
            min_pixels=20,
            ignore_val=255,
            do_close=False,
            close_ksize=3,
            tangent_radius=7,
            min_tangent_pts=5,
            fill_to_mask=True,
        )
    def refine_pseudo_crack_mask_by_orientation_extension(
        self,
        crack_mask: torch.Tensor,     # [B,1,H,W], float
        ori_bin_map: torch.Tensor,    # [B,H,W], long
        num_bins: int,
        ignore_val: int = 255,
        max_extend_len: int = 15,
        thickness: int = 1,
        min_component_area: int = 5,
        only_endpoints: bool = True,
        endpoint_neighbor_thr: int = 1,
    ):
        """
        ori_bin_map을 이용해 crack mask를 방향 기반으로 연장.
        주 용도:
        - endpoint 주변의 짧은 gap 메우기
        - pseudo label refinement 전 crack continuity 강화
    
        return:
            refined_mask: [B,1,H,W] float
        """
        import numpy as np
        import cv2
        import torch
    
        device = crack_mask.device
        B, _, H, W = crack_mask.shape
    
        x = (crack_mask > 0.5).detach().cpu().numpy().astype(np.uint8)
        ori = ori_bin_map.detach().cpu().numpy().astype(np.int64)
    
        out = np.zeros_like(x, dtype=np.uint8)
    
        # connected components
        try:
            from scipy.ndimage import label as ndi_label
            label_fn = lambda z: ndi_label(z, structure=np.ones((3, 3), dtype=np.uint8))[0]
        except Exception:
            from skimage.measure import label as sk_label
            label_fn = lambda z: sk_label(z, connectivity=2)
    
        # skeletonize
        try:
            from skimage.morphology import skeletonize
            skel_fn = lambda z: skeletonize(z > 0).astype(np.uint8)
        except Exception:
            def skel_fn(z):
                if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
                    return (cv2.ximgproc.thinning((z > 0).astype(np.uint8) * 255) > 0).astype(np.uint8)
                return (z > 0).astype(np.uint8)
    
        def bin_to_angle(bin_id, num_bins):
            # bin center angle, undirected orientation in [0, pi)
            return (bin_id + 0.5) * (np.pi / num_bins)
    
        def endpoint_map_from_skeleton(skel):
            """
            endpoint: 8-neighbor 개수가 1인 skeleton pixel
            """
            Hs, Ws = skel.shape
            ep = np.zeros_like(skel, dtype=np.uint8)
            ys, xs = np.where(skel > 0)
            for y, x in zip(ys, xs):
                y0, y1 = max(0, y - 1), min(Hs, y + 2)
                x0, x1 = max(0, x - 1), min(Ws, x + 2)
                n = int(skel[y0:y1, x0:x1].sum()) - 1
                if n <= endpoint_neighbor_thr:
                    ep[y, x] = 1
            return ep
    
        def clip_pt(xx, yy, W, H):
            xx = int(np.clip(xx, 0, W - 1))
            yy = int(np.clip(yy, 0, H - 1))
            return xx, yy
    
        for b in range(B):
            m = x[b, 0].copy()
    
            # 작은 component 제거
            comp = label_fn(m)
            if comp.max() > 0:
                cleaned = np.zeros_like(m, dtype=np.uint8)
                for cid in range(1, int(comp.max()) + 1):
                    area = int((comp == cid).sum())
                    if area >= min_component_area:
                        cleaned[comp == cid] = 1
                m = cleaned
    
            if m.sum() == 0:
                out[b, 0] = m
                continue
    
            skel = skel_fn(m)
            if only_endpoints:
                seeds = endpoint_map_from_skeleton(skel)
            else:
                seeds = skel.copy()
    
            ys, xs = np.where(seeds > 0)
            ext = np.zeros_like(m, dtype=np.uint8)
    
            for y, x0 in zip(ys, xs):
                bin_id = int(ori[b, y, x0])
                if bin_id == ignore_val or bin_id < 0 or bin_id >= num_bins:
                    continue
    
                ang = bin_to_angle(bin_id, num_bins)
    
                # undirected orientation이므로 양방향 모두 시도
                dirs = [
                    (np.cos(ang), np.sin(ang)),
                    (-np.cos(ang), -np.sin(ang)),
                ]
    
                for dx, dy in dirs:
                    x1 = int(round(x0 + max_extend_len * dx))
                    y1 = int(round(y + max_extend_len * dy))
                    x1, y1 = clip_pt(x1, y1, W, H)
    
                    cv2.line(
                        ext,
                        (x0, y),
                        (x1, y1),
                        color=1,
                        thickness=thickness,
                    )
    
            # 기존 crack + extension
            merged = np.maximum(m, ext)
    
            out[b, 0] = merged
    
        return torch.from_numpy(out).float().to(device)
    import numpy as np
    import matplotlib.pyplot as plt
    
    def visualize_orientation_vector(self, crack_mask, ori_bin_map, num_bins=8, idx=0):
    
        mask = crack_mask[idx,0].detach().cpu().numpy()
        ori = ori_bin_map[idx].detach().cpu().numpy()
    
        ys, xs = np.where(mask > 0)
    
        # 너무 많은 화살표 방지
        step = max(1, len(xs)//500)
    
        xs = xs[::step]
        ys = ys[::step]
    
        angles = ori[ys, xs] * np.pi / num_bins
    
        u = np.cos(angles)
        v = np.sin(angles)
    
        plt.figure(figsize=(6,6))
        plt.imshow(mask, cmap="gray")
    
        plt.quiver(xs, ys, u, v, color='red', scale=30)
    
        plt.gca().invert_yaxis()
        plt.title("Orientation vectors")
        plt.show()
    def grow_crack_along_orientation_unconditional(
        self,
        crack_mask: torch.Tensor,      # [B,1,H,W], float
        ori_bin_map: torch.Tensor,     # [B,H,W], int64, 0~num_bins-1, ignore 가능
        num_bins: int = 8,
        grow_step: float = 1.0,
        grow_max_steps: int = 80,
        thickness: int = 3,
        min_area: int = 5,
        trace_len: int = 12,
        smooth_window: int = 7,
        ignore_val: int = 255,
    ):
        """
        acceptance 조건 없이 endpoint에서 ori_bin_map 방향으로 계속 성장.
        - skeleton endpoint를 찾고
        - local tangent + ori_bin prior를 섞어서 초기 방향을 정한 뒤
        - 매 step마다 ori_bin_map 방향을 따라 계속 전진
        - 새 픽셀을 crack으로 칠함
    
        return:
            out_t: [B,1,H,W] float tensor
        """
        import cv2
        import numpy as np
        import torch
    
        device = crack_mask.device
        B, _, H, W = crack_mask.shape
    
        x = (crack_mask > 0.5).detach().cpu().numpy().astype(np.uint8)   # [B,1,H,W]
        ori = ori_bin_map.detach().cpu().numpy().astype(np.int64)        # [B,H,W]
    
        out = np.zeros((B, H, W), dtype=np.uint8)
    
        try:
            from skimage.morphology import skeletonize
            skel_fn = lambda z: skeletonize(z > 0).astype(np.uint8)
        except Exception:
            # fallback: morphology thinning approximation
            def skel_fn(z):
                z = (z > 0).astype(np.uint8)
                skel = np.zeros_like(z, dtype=np.uint8)
                kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
                while True:
                    eroded = cv2.erode(z, kernel)
                    temp = cv2.dilate(eroded, kernel)
                    temp = cv2.subtract(z, temp)
                    skel = cv2.bitwise_or(skel, temp)
                    z = eroded.copy()
                    if z.max() == 0:
                        break
                return (skel > 0).astype(np.uint8)
    
        def angle_to_vec(theta_deg):
            th = np.deg2rad(theta_deg)
            return np.array([np.cos(th), np.sin(th)], dtype=np.float32)
    
        def ori_bin_to_angle(bin_id):
            # 0~num_bins-1 -> [0,180)
            return (180.0 / num_bins) * float(bin_id)
    
        def draw_disk(mask, cx, cy, radius):
            h, w = mask.shape
            x0, y0 = int(round(cx)), int(round(cy))
            rr = max(1, int(radius))
            cv2.circle(mask, (x0, y0), rr, 1, -1)
    
        def find_endpoints(skel):
            """
            8-neighbor degree==1 인 skeleton endpoint
            """
            h, w = skel.shape
            pts = []
            for y in range(1, h - 1):
                for x_ in range(1, w - 1):
                    if skel[y, x_] == 0:
                        continue
                    nb = skel[y - 1:y + 2, x_ - 1:x_ + 2].sum() - 1
                    if nb == 1:
                        pts.append((x_, y))
            return pts
    
        def trace_local_tangent(skel, ep, max_steps=12):
            """
            endpoint에서 skeleton을 따라가며 local tangent 추정
            반환: unit vector [vx, vy] or None
            """
            x0, y0 = ep
            visited = set()
            path = [(x0, y0)]
            cur = (x0, y0)
            prev = None
    
            for _ in range(max_steps):
                x_, y_ = cur
                visited.add(cur)
                nbrs = []
                for yy in range(max(0, y_ - 1), min(H, y_ + 2)):
                    for xx in range(max(0, x_ - 1), min(W, x_ + 2)):
                        if xx == x_ and yy == y_:
                            continue
                        if skel[yy, xx] > 0 and (xx, yy) != prev:
                            nbrs.append((xx, yy))
                if len(nbrs) == 0:
                    break
                if len(nbrs) == 1:
                    nxt = nbrs[0]
                else:
                    # 이전 진행 방향이 있으면 가장 일직선인 쪽 선택
                    if len(path) >= 2:
                        vx = path[-1][0] - path[-2][0]
                        vy = path[-1][1] - path[-2][1]
                        norm = (vx * vx + vy * vy) ** 0.5 + 1e-6
                        vx, vy = vx / norm, vy / norm
                        best, best_score = None, -1e9
                        for cand in nbrs:
                            dx, dy = cand[0] - x_, cand[1] - y_
                            dn = (dx * dx + dy * dy) ** 0.5 + 1e-6
                            dx, dy = dx / dn, dy / dn
                            score = vx * dx + vy * dy
                            if score > best_score:
                                best_score = score
                                best = cand
                        nxt = best
                    else:
                        nxt = nbrs[0]
    
                prev = cur
                cur = nxt
                path.append(cur)
    
            if len(path) < 2:
                return None
    
            pts = np.array(path, dtype=np.float32)
            if len(pts) >= 2:
                # endpoint에서 안쪽으로 향하는 방향
                v = pts[-1] - pts[0]
                n = np.linalg.norm(v) + 1e-6
                v = v / n
                return v
            return None
    
        def smooth_direction_history(hist, win=7):
            if len(hist) == 0:
                return None
            arr = np.array(hist[-win:], dtype=np.float32)
            v = arr.mean(axis=0)
            n = np.linalg.norm(v) + 1e-6
            return v / n
    
        for b in range(B):
            m = x[b, 0].copy()
    
            # 작은 컴포넌트 제거
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
            cleaned = np.zeros_like(m, dtype=np.uint8)
            for cid in range(1, num_labels):
                if stats[cid, cv2.CC_STAT_AREA] >= min_area:
                    cleaned[labels == cid] = 1
            m = cleaned
    
            grown = m.copy()
            skel = skel_fn(m)
            endpoints = find_endpoints(skel)
    
            for ep in endpoints:
                ex, ey = ep
    
                # 1) local tangent
                tangent = trace_local_tangent(skel, ep, max_steps=trace_len)
    
                # 2) ori prior
                bin_id = ori[b, ey, ex]
                ori_vec = None
                if bin_id != ignore_val and 0 <= bin_id < num_bins:
                    theta = ori_bin_to_angle(bin_id)
                    ori_vec = angle_to_vec(theta)
    
                    # tangent와 반대 방향일 수도 있으므로 부호 정렬
                    if tangent is not None and np.dot(tangent, ori_vec) < 0:
                        ori_vec = -ori_vec
    
                # 3) 초기 진행 방향
                if tangent is not None and ori_vec is not None:
                    v = 0.6 * tangent + 0.4 * ori_vec
                elif tangent is not None:
                    v = tangent
                elif ori_vec is not None:
                    v = ori_vec
                else:
                    continue
    
                vn = np.linalg.norm(v) + 1e-6
                v = v / vn
    
                px, py = float(ex), float(ey)
                dir_hist = [v.copy()]
    
                for _ in range(grow_max_steps):
                    ix = int(round(px))
                    iy = int(round(py))
                    if not (0 <= ix < W and 0 <= iy < H):
                        break
    
                    # 현재 위치의 orientation을 계속 반영
                    cur_bin = ori[b, iy, ix]
                    if cur_bin != ignore_val and 0 <= cur_bin < num_bins:
                        theta = ori_bin_to_angle(cur_bin)
                        cur_ori_vec = angle_to_vec(theta)
                        # 방향 반전 정렬
                        if np.dot(cur_ori_vec, v) < 0:
                            cur_ori_vec = -cur_ori_vec
                        v = 0.7 * v + 0.3 * cur_ori_vec
                        v = v / (np.linalg.norm(v) + 1e-6)
    
                    dir_hist.append(v.copy())
                    v_sm = smooth_direction_history(dir_hist, win=smooth_window)
                    if v_sm is not None:
                        v = v_sm
    
                    # 한 step 전진
                    nx = px + grow_step * float(v[0])
                    ny = py + grow_step * float(v[1])
    
                    if not (0 <= int(round(nx)) < W and 0 <= int(round(ny)) < H):
                        break
    
                    # unconditional draw
                    rr = max(1, thickness // 2)
                    cv2.line(
                        grown,
                        (int(round(px)), int(round(py))),
                        (int(round(nx)), int(round(ny))),
                        1,
                        thickness=thickness
                    )
                    draw_disk(grown, nx, ny, rr)
    
                    px, py = nx, ny
    
            out[b] = grown
    
        out_t = torch.from_numpy(out).to(device=device, dtype=torch.float32).unsqueeze(1)
        return out_t
    def refine_pseudo_crack_mask_endpoint_bridge(
        self,
        crack_mask: torch.Tensor,   # [B,1,H,W]
        ori_bin_map: torch.Tensor = None,   # [B,H,W]
        num_bins: int = 8,
        min_area: int = 3,
        max_gap: int = 60,
        max_angle_diff: float = 75.0,
        thickness: int = 2,
        trace_len: int = 12,
        line_gap: int = 20,
        bezier_alpha: float = 0.35,
        bezier_alpha_max: float = 20.0,
        bezier_num_pts: int = 30,
        grow_step: float = 1.0,
        grow_max_steps: int = 80,
        hit_radius: int = 3,
        smooth_window: int = 7,
        meet_thr: float = 4.0,
        ignore_val: int = 255,
        ema_keep: float = 0.8,
        corridor_radius: float = 8.0,
        prior_pull: float = 0.35,
        soft_accept_ratio: float = 0.72,
        max_tortuosity: float = 1.75,
        fallback_gap_ratio: float = 1.0,
        enable_unconditional_fallback: bool = False,
        unconditional_only_for_unmatched: bool = True,
    ):
        """
        endpoint bridging + unmatched endpoint에 대해 unconditional orientation-guided growth fallback
    
        반환:
            out_t: [B,1,H,W] float tensor
        """
        import cv2
        import numpy as np
        import torch
    
        device = crack_mask.device
        B, _, H, W = crack_mask.shape
    
        x = (crack_mask > 0.5).detach().cpu().numpy().astype(np.uint8)   # [B,1,H,W]
        ori = None
        if ori_bin_map is not None:
            ori = ori_bin_map.detach().cpu().numpy().astype(np.int64)    # [B,H,W]
    
        out = np.zeros((B, H, W), dtype=np.uint8)
    
        try:
            from skimage.morphology import skeletonize
            skel_fn = lambda z: skeletonize(z > 0).astype(np.uint8)
        except Exception:
            def skel_fn(z):
                z = (z > 0).astype(np.uint8)
                skel = np.zeros_like(z, dtype=np.uint8)
                kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
                while True:
                    eroded = cv2.erode(z, kernel)
                    temp = cv2.dilate(eroded, kernel)
                    temp = cv2.subtract(z, temp)
                    skel = cv2.bitwise_or(skel, temp)
                    z = eroded.copy()
                    if z.max() == 0:
                        break
                return (skel > 0).astype(np.uint8)
    
        def angle_to_vec(theta_deg):
            th = np.deg2rad(theta_deg)
            return np.array([np.cos(th), np.sin(th)], dtype=np.float32)
    
        def vec_to_angle(v):
            return np.rad2deg(np.arctan2(v[1], v[0]))
    
        def wrap_180(a):
            while a < 0:
                a += 180.0
            while a >= 180.0:
                a -= 180.0
            return a
    
        def angle_diff_undirected(a, b):
            d = abs(wrap_180(a) - wrap_180(b))
            return min(d, 180.0 - d)
    
        def ori_bin_to_angle(bin_id):
            return (180.0 / num_bins) * float(bin_id)
    
        def point_dist(p1, p2):
            return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))
    
        def draw_disk(mask, cx, cy, radius):
            x0, y0 = int(round(cx)), int(round(cy))
            rr = max(1, int(radius))
            cv2.circle(mask, (x0, y0), rr, 1, -1)
    
        def find_endpoints(skel):
            pts = []
            for y in range(1, skel.shape[0] - 1):
                for x_ in range(1, skel.shape[1] - 1):
                    if skel[y, x_] == 0:
                        continue
                    nb = skel[y - 1:y + 2, x_ - 1:x_ + 2].sum() - 1
                    if nb == 1:
                        pts.append((x_, y))
            return pts
    
        def trace_local_tangent(skel, ep, max_steps=12):
            """
            endpoint에서 skeleton 안쪽으로 추적하여 local tangent 추정
            반환: unit vector [vx, vy] 또는 None
            """
            x0, y0 = ep
            path = [(x0, y0)]
            cur = (x0, y0)
            prev = None
    
            for _ in range(max_steps):
                x_, y_ = cur
                nbrs = []
                for yy in range(max(0, y_ - 1), min(H, y_ + 2)):
                    for xx in range(max(0, x_ - 1), min(W, x_ + 2)):
                        if xx == x_ and yy == y_:
                            continue
                        if skel[yy, xx] > 0 and (xx, yy) != prev:
                            nbrs.append((xx, yy))
    
                if len(nbrs) == 0:
                    break
                elif len(nbrs) == 1:
                    nxt = nbrs[0]
                else:
                    if len(path) >= 2:
                        vx = path[-1][0] - path[-2][0]
                        vy = path[-1][1] - path[-2][1]
                        vn = (vx * vx + vy * vy) ** 0.5 + 1e-6
                        vx, vy = vx / vn, vy / vn
                        best, best_score = None, -1e9
                        for cand in nbrs:
                            dx, dy = cand[0] - x_, cand[1] - y_
                            dn = (dx * dx + dy * dy) ** 0.5 + 1e-6
                            dx, dy = dx / dn, dy / dn
                            score = vx * dx + vy * dy
                            if score > best_score:
                                best_score = score
                                best = cand
                        nxt = best
                    else:
                        nxt = nbrs[0]
    
                prev = cur
                cur = nxt
                path.append(cur)
    
            if len(path) < 2:
                return None
    
            pts = np.array(path, dtype=np.float32)
            v = pts[-1] - pts[0]   # endpoint -> inward
            vn = np.linalg.norm(v) + 1e-6
            v = v / vn
            return v
    
        def get_initial_dir(ep, skel, ori_map_b):
            tangent = trace_local_tangent(skel, ep, max_steps=trace_len)
            x_, y_ = ep
    
            ori_vec = None
            if ori_map_b is not None:
                bin_id = ori_map_b[y_, x_]
                if bin_id != ignore_val and 0 <= bin_id < num_bins:
                    theta = ori_bin_to_angle(bin_id)
                    ori_vec = angle_to_vec(theta)
                    if tangent is not None and np.dot(tangent, ori_vec) < 0:
                        ori_vec = -ori_vec
    
            if tangent is not None and ori_vec is not None:
                v = (1.0 - prior_pull) * tangent + prior_pull * ori_vec
            elif tangent is not None:
                v = tangent
            elif ori_vec is not None:
                v = ori_vec
            else:
                return None
    
            vn = np.linalg.norm(v) + 1e-6
            return v / vn
    
        def smooth_direction_history(hist, win=7):
            if len(hist) == 0:
                return None
            arr = np.array(hist[-win:], dtype=np.float32)
            v = arr.mean(axis=0)
            vn = np.linalg.norm(v) + 1e-6
            return v / vn
    
        def segment_hits_mask(p0, p1, mask, radius=3):
            """
            p0->p1 선분 주변 radius 안에 기존 crack가 있는지 검사
            """
            h, w = mask.shape
            n = max(2, int(np.hypot(p1[0] - p0[0], p1[1] - p0[1])) * 2)
            for i in range(n + 1):
                t = i / max(1, n)
                x_ = (1 - t) * p0[0] + t * p1[0]
                y_ = (1 - t) * p0[1] + t * p1[1]
                xi, yi = int(round(x_)), int(round(y_))
                x1, x2 = max(0, xi - radius), min(w, xi + radius + 1)
                y1, y2 = max(0, yi - radius), min(h, yi + radius + 1)
                if mask[y1:y2, x1:x2].sum() > 0:
                    return True
            return False
    
        def draw_polyline(mask, pts, thickness=2):
            if len(pts) < 2:
                return
            for i in range(len(pts) - 1):
                p0 = pts[i]
                p1 = pts[i + 1]
                cv2.line(
                    mask,
                    (int(round(p0[0])), int(round(p0[1]))),
                    (int(round(p1[0])), int(round(p1[1]))),
                    1,
                    thickness=thickness
                )
    
        def draw_bezier(mask, p0, p1, v0, v1, thickness=2, alpha=0.35, alpha_max=20.0, num_pts=30):
            gap = point_dist(p0, p1)
            alpha_pix = min(alpha * gap, alpha_max)
    
            c0 = np.array([p0[0], p0[1]], dtype=np.float32)
            c1 = c0 + alpha_pix * v0
            c2 = np.array([p1[0], p1[1]], dtype=np.float32) - alpha_pix * v1
            c3 = np.array([p1[0], p1[1]], dtype=np.float32)
    
            pts = []
            for i in range(num_pts + 1):
                t = i / num_pts
                pt = ((1 - t) ** 3) * c0 + 3 * ((1 - t) ** 2) * t * c1 + 3 * (1 - t) * (t ** 2) * c2 + (t ** 3) * c3
                pts.append((float(pt[0]), float(pt[1])))
            draw_polyline(mask, pts, thickness=thickness)
    
        def unconditional_growth_from_endpoint(
            grown,
            ep,
            init_dir,
            ori_map_b,
            steps=60,
            step_size=1.0,
            thickness=3,
            smooth_window=7,
        ):
            """
            acceptance 조건 없이 ori 방향으로 계속 성장
            """
            if init_dir is None:
                return grown
    
            px, py = float(ep[0]), float(ep[1])
            v = init_dir.copy()
            dir_hist = [v.copy()]
    
            for _ in range(steps):
                ix = int(round(px))
                iy = int(round(py))
                if not (0 <= ix < W and 0 <= iy < H):
                    break
    
                if ori_map_b is not None:
                    cur_bin = ori_map_b[iy, ix]
                    if cur_bin != ignore_val and 0 <= cur_bin < num_bins:
                        cur_vec = angle_to_vec(ori_bin_to_angle(cur_bin))
                        if np.dot(cur_vec, v) < 0:
                            cur_vec = -cur_vec
                        v = ema_keep * v + (1.0 - ema_keep) * cur_vec
                        vn = np.linalg.norm(v) + 1e-6
                        v = v / vn
    
                dir_hist.append(v.copy())
                v_sm = smooth_direction_history(dir_hist, win=smooth_window)
                if v_sm is not None:
                    v = v_sm
    
                nx = px + step_size * float(v[0])
                ny = py + step_size * float(v[1])
    
                if not (0 <= int(round(nx)) < W and 0 <= int(round(ny)) < H):
                    break
    
                cv2.line(
                    grown,
                    (int(round(px)), int(round(py))),
                    (int(round(nx)), int(round(ny))),
                    1,
                    thickness=thickness
                )
                draw_disk(grown, nx, ny, max(1, thickness // 2))
    
                px, py = nx, ny
    
            return grown
    
        for b in range(B):
            m = x[b, 0].copy()
            ori_b = ori[b] if ori is not None else None
    
            # 작은 컴포넌트 제거
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
            cleaned = np.zeros_like(m, dtype=np.uint8)
            for cid in range(1, num_labels):
                if stats[cid, cv2.CC_STAT_AREA] >= min_area:
                    cleaned[labels == cid] = 1
            m = cleaned
    
            grown = m.copy()
            skel = skel_fn(m)
            endpoints = find_endpoints(skel)
    
            matched_eps = set()
    
            # -------------------------
            # 1) endpoint-to-endpoint bridging
            # -------------------------
            for i in range(len(endpoints)):
                ep_i = endpoints[i]
                v_i = get_initial_dir(ep_i, skel, ori_b)
                if v_i is None:
                    continue
    
                best_j = None
                best_score = -1e9
                best_gap = None
                best_vj = None
    
                for j in range(len(endpoints)):
                    if i == j:
                        continue
                    ep_j = endpoints[j]
    
                    gap = point_dist(ep_i, ep_j)
                    if gap <= 1.0 or gap > max_gap:
                        continue
    
                    v_j = get_initial_dir(ep_j, skel, ori_b)
                    if v_j is None:
                        continue
    
                    d_ij = np.array([ep_j[0] - ep_i[0], ep_j[1] - ep_i[1]], dtype=np.float32)
                    dn = np.linalg.norm(d_ij) + 1e-6
                    d_ij = d_ij / dn
    
                    a1 = angle_diff_undirected(vec_to_angle(v_i), vec_to_angle(d_ij))
                    a2 = angle_diff_undirected(vec_to_angle(v_j), vec_to_angle(-d_ij))
    
                    if a1 > max_angle_diff or a2 > max_angle_diff:
                        continue
    
                    score = (180.0 - a1) + (180.0 - a2) - 0.25 * gap
                    if score > best_score:
                        best_score = score
                        best_j = j
                        best_gap = gap
                        best_vj = v_j
    
                if best_j is None:
                    continue
    
                ep_j = endpoints[best_j]
                v_j = best_vj
                gap = best_gap
    
                # 이미 거의 이어져 있거나 주변에 선분이 닿으면 skip
                if segment_hits_mask(ep_i, ep_j, grown, radius=hit_radius):
                    continue
    
                # 작은 gap은 직선 연결
                if gap <= line_gap:
                    cv2.line(
                        grown,
                        (int(ep_i[0]), int(ep_i[1])),
                        (int(ep_j[0]), int(ep_j[1])),
                        1,
                        thickness=thickness
                    )
                    matched_eps.add(ep_i)
                    matched_eps.add(ep_j)
                else:
                    # 큰 gap은 bezier
                    draw_bezier(
                        grown,
                        ep_i, ep_j,
                        v_i, v_j,
                        thickness=thickness,
                        alpha=bezier_alpha,
                        alpha_max=bezier_alpha_max,
                        num_pts=bezier_num_pts
                    )
                    matched_eps.add(ep_i)
                    matched_eps.add(ep_j)
    
            # -------------------------
            # 2) unmatched endpoint에 대해 unconditional fallback growth
            # -------------------------
            if enable_unconditional_fallback:
                for ep in endpoints:
                    if unconditional_only_for_unmatched and ep in matched_eps:
                        continue
    
                    init_dir = get_initial_dir(ep, skel, ori_b)
                    if init_dir is None:
                        continue
    
                    grown = unconditional_growth_from_endpoint(
                        grown=grown,
                        ep=ep,
                        init_dir=init_dir,
                        ori_map_b=ori_b,
                        steps=grow_max_steps,
                        step_size=grow_step,
                        thickness=thickness,
                        smooth_window=smooth_window,
                    )
    
            out[b] = grown
    
        out_t = torch.from_numpy(out).to(device=device, dtype=torch.float32).unsqueeze(1)
        return out_t
    def refine_pseudo_crack_mask(
        self,
        crack_mask: torch.Tensor,   # [B,1,H,W], float or bool
        close_ksize: int = 5,
        min_area: int = 8,
        line_len: int = 7,
    ):
        import cv2
        import numpy as np
        import torch
    
        device = crack_mask.device
        B, _, H, W = crack_mask.shape
    
        # Tensor -> numpy uint8
        x = (crack_mask > 0.5).detach().cpu().numpy().astype(np.uint8)  # [B,1,H,W]
        out = np.zeros((B, H, W), dtype=np.uint8)
    
        def make_line_kernel(length=7, angle=0):
            k = np.zeros((length, length), dtype=np.uint8)
            c = length // 2
            if angle == 0:
                k[c, :] = 1
            elif angle == 90:
                k[:, c] = 1
            elif angle == 45:
                for i in range(length):
                    k[length - 1 - i, i] = 1
            elif angle == 135:
                for i in range(length):
                    k[i, i] = 1
            return k
    
        for b in range(B):
            m = x[b, 0].copy()   # [H,W]
    
            # 1) 작은 틈 메우기
            if close_ksize > 1:
                kernel_close = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (close_ksize, close_ksize)
                )
                m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel_close)
    
            # 2) 방향성 closing
            merged = m.copy()
            for ang in [0, 45, 90, 135]:
                k = make_line_kernel(line_len, ang)
                mc = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
                merged = np.maximum(merged, mc)
    
            m = merged
    
            # 3) 작은 컴포넌트 제거
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
            cleaned = np.zeros_like(m, dtype=np.uint8)
            for cid in range(1, num_labels):
                if stats[cid, cv2.CC_STAT_AREA] >= min_area:
                    cleaned[labels == cid] = 1
    
            out[b] = cleaned
    
        # numpy -> Tensor
        out_t = torch.from_numpy(out).to(device=device, dtype=torch.float32).unsqueeze(1)  # [B,1,H,W]
        return out_t
        def fill_crack_gaps(self, pseudo_lbl, feat, image_size,
                    max_gap=30,         # 최대 연결할 gap 픽셀 거리
                    sim_threshold=0.5,  # 연결 후보 feature 유사도 기준
                    crack_class=1):
            """
            균열 끝점(endpoint)을 찾아서
            가까운 다른 끝점과 feature가 비슷하면 직선으로 연결
            """
            from skimage.morphology import skeletonize as ski_skeletonize
            from skimage.draw import line as draw_line

            B = pseudo_lbl.shape[0]
            filled = pseudo_lbl.clone()

            for b in range(B):
                mask_np = (pseudo_lbl[b, 0].cpu().numpy() == crack_class).astype(np.uint8)
                if mask_np.sum() == 0:
                    continue

                skeleton = ski_skeletonize(mask_np).astype(np.uint8)

                # 끝점 탐지: 이웃이 1개뿐인 스켈레톤 픽셀
                from scipy.ndimage import convolve
                kernel = np.ones((3, 3), dtype=np.uint8)
                neighbor_count = convolve(skeleton, kernel, mode='constant') * skeleton
                endpoints_y, endpoints_x = np.where(neighbor_count == 2)  # 자기자신(1) + 이웃(1) = 2
                
                if len(endpoints_x) < 2:
                    continue

                # 끝점들의 feature 추출
                fH, fW = feat.shape[-2:]
                orig_H, orig_W = image_size
                ep_coords = np.column_stack((endpoints_x, endpoints_y))
                fx = np.clip((endpoints_x * fW / orig_W).astype(int), 0, fW-1)
                fy = np.clip((endpoints_y * fH / orig_H).astype(int), 0, fH-1)
                ep_feat = feat[b, :, fy, fx].T  # (E, A)
                ep_feat = F.normalize(ep_feat, dim=1)

                # 끝점 쌍 중 gap이 작고 feature 유사도 높은 것 연결
                for i in range(len(ep_coords)):
                    for j in range(i+1, len(ep_coords)):
                        dist = np.linalg.norm(ep_coords[i] - ep_coords[j])
                        if dist > max_gap:
                            continue
                        
                        sim = (ep_feat[i] * ep_feat[j]).sum().item()
                        if sim < sim_threshold:
                            continue
                        
                        # 두 끝점 사이를 직선으로 채움
                        rr, cc = draw_line(endpoints_y[i], endpoints_x[i],
                                        endpoints_y[j], endpoints_x[j])
                        filled[b, 0, rr, cc] = crack_class

            return filled
    def update_proto_bank(self, tgt_proj, pseudo_label,
                           complexity, crack_class=1, momentum=0.99):
        """
        complexity 낮은 (신뢰도 높은) batch만 prototype bank 업데이트
        """
        reliability = 1.0 - complexity
    
        # 신뢰도 낮으면 업데이트 안 함
        if reliability < 0.3:
            return
    
        # [B, C, H, W] → [B, H, W, C]
        feat_flat  = tgt_proj.permute(0, 2, 3, 1)
        crack_mask = (pseudo_label == crack_class)   # [B, H, W]
    
        crack_feats = feat_flat[crack_mask]           # [N_crack, C]
        if crack_feats.shape[0] == 0:
            return
    
        cur_proto = crack_feats.mean(dim=0)           # [C]
    
        if self.crack_proto_bank is None:
            self.crack_proto_bank  = cur_proto.detach()
            self.proto_reliability = reliability
        else:
            self.crack_proto_bank = (
                momentum * self.crack_proto_bank +
                (1 - momentum) * cur_proto.detach()
            )
            self.proto_reliability = (
                momentum * self.proto_reliability +
                (1 - momentum) * reliability
            )
    
        print(
            f"[ProtoBank] reliability={reliability:.3f} "
            f"proto_reliability={self.proto_reliability:.3f} "
            f"crack_feats={crack_feats.shape[0]}"
        )
    def refine_pseudo_by_feature_graph(self, pseudo_label, feat, pseudo_prob,
                                   crack_class=1, ignore_index=255,
                                   sim_threshold=0.5, max_pixels=1000,
                                   neighbor_ratio=0.05,
                                   # 핵심 추가 파라미터
                                   low_conf_threshold=0.7):  # 낮은 confidence bg 픽셀 대상
        B, C, fH, fW = feat.shape
        orig_H, orig_W = pseudo_label.shape[-2:]
        refined = pseudo_label.clone()

        for b in range(B):
            lbl = pseudo_label[b]
            prob = pseudo_prob[b]  # (H, W)

            # ── crack 픽셀 샘플링 ──────────────────────────────────────────
            crack_yx = (lbl == crack_class).nonzero(as_tuple=False)
            if len(crack_yx) == 0:
                continue
            if len(crack_yx) > max_pixels:
                idx = torch.linspace(0, len(crack_yx)-1, max_pixels).long()
                crack_yx = crack_yx[idx]

            fy = (crack_yx[:, 0] * fH / orig_H).long().clamp(0, fH-1)
            fx = (crack_yx[:, 1] * fW / orig_W).long().clamp(0, fW-1)
            node_feat = feat[b, :, fy, fx].T  # (N, C)

            # ── fill 대상: bg(0) 중 confidence가 낮은 픽셀만 ────────────────
            # 모델이 확신 없이 bg로 분류한 픽셀 = crack일 가능성 높음
            low_conf_bg_mask = (lbl == 0) & (prob < low_conf_threshold)

            candidate_yx = low_conf_bg_mask.nonzero(as_tuple=False)
            if len(candidate_yx) == 0:
                continue
            if len(candidate_yx) > max_pixels:
                idx = torch.linspace(0, len(candidate_yx)-1, max_pixels).long()
                candidate_yx = candidate_yx[idx]

            iy = (candidate_yx[:, 0] * fH / orig_H).long().clamp(0, fH-1)
            ix = (candidate_yx[:, 1] * fW / orig_W).long().clamp(0, fW-1)
            cand_feat = feat[b, :, iy, ix].T  # (M, C)

            # ── cosine similarity 기반 fill ────────────────────────────────
            cross_sim = cand_feat @ node_feat.T                            # (M, N)
            crack_neighbor_count = (cross_sim > sim_threshold).sum(dim=1) # (M,)

            min_neighbors = max(1, int(len(crack_yx) * neighbor_ratio))
            fill_mask = crack_neighbor_count >= min_neighbors

            fill_yx = candidate_yx[fill_mask]
            if len(fill_yx) > 0:
                refined[b, fill_yx[:, 0], fill_yx[:, 1]] = crack_class

        return refined
    def extract_width_map_batch(self, label, crack_class=1):
        """(B, H, W) → (B, 1, H, W) width map"""
        from scipy.ndimage import distance_transform_edt
        B, H, W = label.shape
        width_maps = torch.zeros(B, 1, H, W, device=label.device)
        for b in range(B):
            crack = (label[b].cpu().numpy() == crack_class).astype(np.uint8)
            if crack.sum() == 0:
                continue
            dist = distance_transform_edt(crack).astype(np.float32)
            width_maps[b, 0] = torch.from_numpy(dist).to(label.device)
        return width_maps
    def refine_pseudo_morphology(self, pseudo_label, pseudo_prob,
                                  crack_class=1, ignore_index=255,
                                  cur_iter=0):
        """
        crack의 선형/연속성 특성을 이용한 정제
        - 파라미터 자동화: min_crack_length, prob_threshold
        - 순서: gap fill → noise 제거
        """
        import cv2
        B = pseudo_label.shape[0]
        refined = pseudo_label.clone()
    
        for b in range(B):
            lbl        = pseudo_label[b].cpu().numpy().astype(np.uint8)
            prob       = pseudo_prob[b].cpu().numpy()
            crack_mask = (lbl == crack_class).astype(np.uint8)
            before_crack = crack_mask.sum()
    
            if crack_mask.sum() == 0:
                print(f"[Morph iter={cur_iter} b={b}] crack=0, skip")
                continue
    
            H, W = lbl.shape
    
            # ── min_crack_length 자동화 ───────────────────────────────────
            size_based_length = max(5, int(min(H, W) * 0.01))
    
            num_labels_init, _, stats_init, _ = cv2.connectedComponentsWithStats(
                crack_mask, connectivity=8
            )
            if num_labels_init > 1:
                lengths = [
                    max(stats_init[i, cv2.CC_STAT_WIDTH],
                        stats_init[i, cv2.CC_STAT_HEIGHT])
                    for i in range(1, num_labels_init)
                ]
                dist_based_length = int(np.percentile(lengths, 20))
                min_crack_length  = min(size_based_length, dist_based_length)
                min_crack_length  = max(3, min_crack_length)
            else:
                min_crack_length = size_based_length
    
            # ── prob_threshold 자동화 (Otsu) ─────────────────────────────
            crack_probs = prob[crack_mask == 1]
            bg_probs    = prob[lbl == 0]
    
            if len(crack_probs) > 0 and len(bg_probs) > 0:
                n_sample      = min(len(crack_probs), len(bg_probs), 10000)
                sampled_crack = np.random.choice(crack_probs, n_sample, replace=False)
                sampled_bg    = np.random.choice(bg_probs,    n_sample, replace=False)
                all_probs     = np.concatenate([sampled_crack, sampled_bg])
    
                hist, bin_edges = np.histogram(all_probs, bins=100, range=(0, 1))
                bin_centers     = (bin_edges[:-1] + bin_edges[1:]) / 2
    
                best_thresh, best_var = 0.5, -1
                for t_idx in range(1, len(hist)):
                    w0 = hist[:t_idx].sum()
                    w1 = hist[t_idx:].sum()
                    if w0 == 0 or w1 == 0:
                        continue
                    mu0 = (hist[:t_idx] * bin_centers[:t_idx]).sum() / w0
                    mu1 = (hist[t_idx:] * bin_centers[t_idx:]).sum() / w1
                    between_var = w0 * w1 * (mu0 - mu1) ** 2
                    if between_var > best_var:
                        best_var    = between_var
                        best_thresh = bin_centers[t_idx]
    
                prob_threshold = float(np.clip(best_thresh, 0.3, 0.8))
            else:
                prob_threshold = 0.5
    
            # ── Step 1: 끊긴 crack 갭 연결 (먼저) ───────────────────────
            kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated = cv2.dilate(crack_mask, kernel, iterations=2)
    
            # bg로 예측됐지만 crack prob이 높은 픽셀 → gap 후보
            high_conf_crack_bg = (lbl == 0) & (prob > prob_threshold)
            gap_fill    = dilated & high_conf_crack_bg.astype(np.uint8)
            filled_mask = np.clip(crack_mask + gap_fill, 0, 1).astype(np.uint8)
            after_step1 = int(filled_mask.sum())
    
            # ── Step 2: 작은 noise crack 제거 (나중에) ───────────────────
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                filled_mask, connectivity=8
            )
            clean_mask = np.zeros_like(crack_mask)
            for i in range(1, num_labels):
                length = max(stats[i, cv2.CC_STAT_WIDTH],
                             stats[i, cv2.CC_STAT_HEIGHT])
                if length >= min_crack_length:
                    clean_mask[labels == i] = 1
    
            after_step2 = int(clean_mask.sum())
    
            # ── pseudo_label 업데이트 ─────────────────────────────────────
            refined_lbl = lbl.copy()
            removed = (filled_mask == 1) & (clean_mask == 0)
            refined_lbl[removed]         = 0
            refined_lbl[clean_mask == 1] = crack_class
    
            refined[b] = torch.from_numpy(refined_lbl).to(pseudo_label.device)
    
            # ── 픽셀 수 출력 ─────────────────────────────────────────────
            gap_count     = int(gap_fill.sum())
            removed_count = after_step1 - after_step2
            total_diff    = after_step2 - int(before_crack)
    
            print(
                f"[Morph iter={cur_iter} b={b}] "
                f"before={int(before_crack)} "
                f"→ step1(gap fill)={after_step1}(+{gap_count}) "
                f"→ step2(noise제거)={after_step2}(-{removed_count}) "
                f"| total({'+' if total_diff>=0 else ''}{total_diff}) "
                f"| min_len={min_crack_length} prob_thresh={prob_threshold:.3f}"
            )
    
        return refined
    '''
    
    def refine_pseudo_morphology(self, pseudo_label, pseudo_prob,
                                crack_class=1, ignore_index=255,
                                prob_threshold=0.5,
                                min_crack_length=10):
        """
        crack의 선형/연속성 특성을 이용한 정제
        1. 고립된 noise crack 제거 (너무 작은 connected component)
        2. 끊긴 crack 연결 (dilate → 연결 → erode)
        """
        B = pseudo_label.shape[0]
        refined = pseudo_label.clone()
        import cv2
        for b in range(B):
            lbl = pseudo_label[b].cpu().numpy().astype(np.uint8)
            prob = pseudo_prob[b].cpu().numpy()

            crack_mask = (lbl == crack_class).astype(np.uint8)
            if crack_mask.sum() == 0:
                continue
        
            # ── Step 1: 작은 noise crack 제거 ─────────────────────────────
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                crack_mask, connectivity=8
            )
            clean_mask = np.zeros_like(crack_mask)
            for i in range(1, num_labels):
                length = max(stats[i, cv2.CC_STAT_WIDTH],
                            stats[i, cv2.CC_STAT_HEIGHT])
                if length >= min_crack_length:
                    clean_mask[labels == i] = 1

            # ── Step 2: 끊긴 crack 갭 연결 ────────────────────────────────
            # dilate로 근처 픽셀 탐색
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated = cv2.dilate(clean_mask, kernel, iterations=2)

            # dilate된 영역 중 low-confidence bg → crack 후보
            low_conf_bg = (lbl == 0) & (prob < prob_threshold)
            gap_fill = dilated & low_conf_bg.astype(np.uint8)

            final_mask = np.clip(clean_mask + gap_fill, 0, 1)

            # ── pseudo_label 업데이트 ──────────────────────────────────────
            refined_lbl = lbl.copy()
            # noise 제거된 부분 → bg로
            removed = (crack_mask == 1) & (clean_mask == 0)
            refined_lbl[removed] = 0
            # gap fill된 부분 → crack으로
            refined_lbl[gap_fill == 1] = crack_class

            refined[b] = torch.from_numpy(refined_lbl).to(pseudo_label.device)

        return refined


    '''
    def _get_line_coords(self, p1, p2, shape):
        """두 점 사이의 픽셀 좌표 리스트 반환"""
        num = int(np.linalg.norm(p1 - p2) * 1.5) + 1
        rs = np.linspace(p1[0], p2[0], num).astype(int)
        cs = np.linspace(p1[1], p2[1], num).astype(int)
        return np.clip(rs, 0, shape[0]-1), np.clip(cs, 0, shape[1]-1)
    def _get_line_coords(self, p1, p2, shape):
        """두 점 사이의 픽셀 좌표들을 샘플링"""
        num = int(np.linalg.norm(p1 - p2) * 1.5) + 1
        rs = np.linspace(p1[0], p2[0], num).astype(int)
        cs = np.linspace(p1[1], p2[1], num).astype(int)
        return np.clip(rs, 0, shape[0]-1), np.clip(cs, 0, shape[1]-1)
    def get_boundary_mask(self, mask, k=3):
        """이미지/라벨에서 경계선 추출"""
        padding = k // 2
        # MaxPool을 이용한 Dilation 효과
        dilated = F.max_pool2d(mask, kernel_size=k, stride=1, padding=padding)
        return (dilated != mask).float()

    def soft_skel(self, x, iter=3):
        """미분 가능한 방식으로 세선화(Thinning) 수행"""
        for _ in range(iter):
            # min-max 연산을 통한 골격 추출
            p1 = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
            p2 = F.max_pool2d(p1, kernel_size=3, stride=1, padding=1)
            x = x - F.relu(x - p2)
        return x

    def dice_loss(self, pred, target):
        """경계선 및 골격 일치도를 위한 Dice Loss"""
        smooth = 1e-5
        inter = (pred * target).sum()
        union = pred.sum() + target.sum()
        return 1 - (2. * inter + smooth) / (union + smooth)

    def update_proto_bank(self, tgt_proj, pseudo_label,
                           complexity, crack_class=1, momentum=0.99):
        """
        complexity 낮은 (신뢰도 높은) batch만 prototype bank 업데이트
        """
        reliability = 1.0 - complexity
    
        # 신뢰도 낮으면 업데이트 안 함
        if reliability < 0.3:
            return
    
        # [B, C, H, W] → [B, H, W, C]
        feat_flat  = tgt_proj.permute(0, 2, 3, 1)
        crack_mask = (pseudo_label == crack_class)   # [B, H, W]
    
        crack_feats = feat_flat[crack_mask]           # [N_crack, C]
        if crack_feats.shape[0] == 0:
            return
    
        cur_proto = crack_feats.mean(dim=0)           # [C]
    
        if self.crack_proto_bank is None:
            self.crack_proto_bank  = cur_proto.detach()
            self.proto_reliability = reliability
        else:
            self.crack_proto_bank = (
                momentum * self.crack_proto_bank +
                (1 - momentum) * cur_proto.detach()
            )
            self.proto_reliability = (
                momentum * self.proto_reliability +
                (1 - momentum) * reliability
            )
    
        print(
            f"[ProtoBank] reliability={reliability:.3f} "
            f"proto_reliability={self.proto_reliability:.3f} "
            f"crack_feats={crack_feats.shape[0]}"
        )
    def refine_pseudo_morphology(self, pseudo_label, pseudo_prob,
                              encoder_feat, crack_class=1,
                              min_crack_length=10):
        import cv2
        import torch.nn.functional as F
    
        B = pseudo_label.shape[0]
        refined = pseudo_label.clone()
        complexity_list = []
    
        for b in range(B):
            lbl        = pseudo_label[b].cpu().numpy().astype(np.uint8)
            prob       = pseudo_prob[b].cpu().numpy()
            feat       = encoder_feat[b]
            crack_mask = (lbl == crack_class).astype(np.uint8)
    
            if crack_mask.sum() == 0:
                complexity_list.append(0.5)
                continue
    
            H, W = lbl.shape
    
            # ── Affinity Map ──────────────────────────────────────────────
            crack_pixel_mask = torch.from_numpy(crack_mask).bool()
            crack_feats      = feat[:, crack_pixel_mask]
    
            if crack_feats.shape[1] == 0:
                complexity_list.append(0.5)
                continue
    
            # 현재 batch prototype
            crack_proto = crack_feats.mean(dim=1, keepdim=True)  # [C, 1]
    
            # ── proto bank 혼합 ───────────────────────────────────────────
            if self.crack_proto_bank is not None:
                bank_proto = self.crack_proto_bank.unsqueeze(1)  # [C, 1]
                w          = self.proto_reliability               # bank 신뢰도
                crack_proto = (
                    w * bank_proto.to(crack_proto.device) +
                    (1 - w) * crack_proto
                )
    
            feat_flat  = feat.view(feat.shape[0], -1)            # [C, H*W]
            proto_norm = F.normalize(crack_proto, dim=0)         # [C, 1]
    
            affinity = (proto_norm.T @ feat_flat).view(H, W).cpu().numpy()
            affinity = (affinity + 1) / 2
    
            # ── complexity 계산 ───────────────────────────────────────────
            crack_aff    = float(affinity[crack_mask == 1].mean())
            bg_aff       = float(affinity[lbl == 0].mean())
            separation   = float(np.clip(crack_aff - bg_aff, 0, 1))
            complexity   = float(1.0 - separation)
            affinity_mid = (crack_aff + bg_aff) / 2
            complexity_list.append(complexity)
    
            # ── Step 1: gap fill 먼저 ─────────────────────────────────────
            kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated = cv2.dilate(crack_mask, kernel, iterations=2)
    
            affinity_thresh  = crack_aff * 0.8
            high_affinity_bg = (lbl == 0) & (affinity > affinity_thresh)
            gap_fill         = dilated & high_affinity_bg.astype(np.uint8)
            filled_mask      = np.clip(crack_mask + gap_fill, 0, 1).astype(np.uint8)
    
            # ── Step 2: noise 제거 ────────────────────────────────────────
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                filled_mask, connectivity=8
            )
            clean_mask = np.zeros_like(crack_mask)
    
            for i in range(1, num_labels):
                component_pixels = (labels == i)
                length = max(stats[i, cv2.CC_STAT_WIDTH],
                            stats[i, cv2.CC_STAT_HEIGHT])
    
                original_in_comp = int(crack_mask[component_pixels].sum())
                total_in_comp    = int(component_pixels.sum())
                gap_ratio        = 1.0 - (original_in_comp / (total_in_comp + 1e-6))
    
                gap_pixels_in_comp = component_pixels & (gap_fill == 1)
                if gap_pixels_in_comp.sum() > 0:
                    gap_affinity = float(affinity[gap_pixels_in_comp].mean())
                else:
                    gap_affinity = float(affinity[component_pixels].mean())
    
                # 조건 1: gap affinity mid 이하 → noise
                if gap_affinity < affinity_mid:
                    if original_in_comp > 0:
                        clean_mask[component_pixels & (crack_mask == 1)] = 1
                    continue
    
                # 조건 2: gap 비율 높을수록 length 기준 엄격
                effective_min = max(
                    min_crack_length * (1.0 + gap_ratio * 0.5),
                    min_crack_length * 0.5
                )
    
                if length >= effective_min:
                    clean_mask[labels == i] = 1
                else:
                    if original_in_comp > 0:
                        clean_mask[component_pixels & (crack_mask == 1)] = 1
    
            # ── pseudo_label 업데이트 ─────────────────────────────────────
            refined_lbl                  = lbl.copy()
            removed                      = (filled_mask == 1) & (clean_mask == 0)
            refined_lbl[removed]         = 0
            refined_lbl[clean_mask == 1] = crack_class
    
            refined[b] = torch.from_numpy(refined_lbl).to(pseudo_label.device)
    
            # ── 로그 ──────────────────────────────────────────────────────
            gap_count     = int(gap_fill.sum())
            removed_count = int(filled_mask.sum()) - int(clean_mask.sum())
            total_diff    = int(clean_mask.sum()) - int(crack_mask.sum())
    
            print(
                f"[Morph b={b}] "
                f"before={int(crack_mask.sum())} "
                f"→ gap fill={int(filled_mask.sum())}(+{gap_count}) "
                f"→ noise제거={int(clean_mask.sum())}(-{removed_count}) "
                f"| total({'+' if total_diff >= 0 else ''}{total_diff}) "
                f"| crack_aff={crack_aff:.3f} bg_aff={bg_aff:.3f} "
                f"| mid={affinity_mid:.3f} complexity={complexity:.3f} "
                f"| bank={'O' if self.crack_proto_bank is not None else 'X'}"
            )
    
        avg_complexity = float(np.mean(complexity_list))
        return refined, avg_complexity
    def refine_pseudo_morphology(self, pseudo_label, pseudo_prob,
                                  encoder_feat, crack_class=1,
                                  prob_threshold=0.5,
                                  min_crack_length=10):
        import cv2
        import torch.nn.functional as F
    
        B = pseudo_label.shape[0]
        refined = pseudo_label.clone()
        complexity_list = []
    
        for b in range(B):
            lbl        = pseudo_label[b].cpu().numpy().astype(np.uint8)
            prob       = pseudo_prob[b].cpu().numpy()
            feat       = encoder_feat[b]
            crack_mask = (lbl == crack_class).astype(np.uint8)
    
            if crack_mask.sum() == 0:
                complexity_list.append(0.5)
                continue
    
            H, W = lbl.shape
    
            # ── Affinity Map (noise 제거에만 사용) ───────────────────────
            crack_pixel_mask = torch.from_numpy(crack_mask).bool()
            crack_feats      = feat[:, crack_pixel_mask]
    
            if crack_feats.shape[1] == 0:
                complexity_list.append(0.5)
                continue
    
            crack_proto = crack_feats.mean(dim=1, keepdim=True)
            if self.crack_proto_bank is not None:
                bank_proto = self.crack_proto_bank.unsqueeze(1)
                w          = self.proto_reliability
                crack_proto = w * bank_proto.to(crack_proto.device) + (1-w) * crack_proto
    
            feat_flat  = feat.view(feat.shape[0], -1)
            proto_norm = F.normalize(crack_proto, dim=0)
            affinity   = (proto_norm.T @ feat_flat).view(H, W).cpu().numpy()
            affinity   = (affinity + 1) / 2
    
            crack_aff    = float(affinity[crack_mask == 1].mean())
            bg_aff       = float(affinity[lbl == 0].mean())
            affinity_mid = (crack_aff + bg_aff) / 2
            complexity   = float(1.0 - np.clip(crack_aff - bg_aff, 0, 1))
            complexity_list.append(complexity)
    
            # ── Step 1: noise 제거 (원래 length + affinity 보조) ─────────
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                crack_mask, connectivity=8
            )
            clean_mask = np.zeros_like(crack_mask)
    
            for i in range(1, num_labels):
                component_pixels = (labels == i)
                length = max(stats[i, cv2.CC_STAT_WIDTH],
                            stats[i, cv2.CC_STAT_HEIGHT])
                comp_affinity = float(affinity[component_pixels].mean())
    
                # affinity 중간값 이하면 noise
                if comp_affinity < affinity_mid:
                    continue
    
                if length >= min_crack_length:
                    clean_mask[labels == i] = 1
    
            # ── Step 2: gap fill (원래 방식 그대로) ──────────────────────
            kernel      = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated     = cv2.dilate(clean_mask, kernel, iterations=2)
            low_conf_bg = (lbl == 0) & (prob < prob_threshold)
            gap_fill    = dilated & low_conf_bg.astype(np.uint8)
    
            # ── pseudo_label 업데이트 ─────────────────────────────────────
            refined_lbl          = lbl.copy()
            removed              = (crack_mask == 1) & (clean_mask == 0)
            refined_lbl[removed] = 0
            refined_lbl[gap_fill == 1] = crack_class
    
            refined[b] = torch.from_numpy(refined_lbl).to(pseudo_label.device)
    
            gap_count     = int(gap_fill.sum())
            removed_count = int(crack_mask.sum()) - int(clean_mask.sum())
            total_diff    = int(clean_mask.sum()) + gap_count - int(crack_mask.sum())
    
            print(
                f"[Morph b={b}] "
                f"before={int(crack_mask.sum())} "
                f"→ noise제거={int(clean_mask.sum())}(-{removed_count}) "
                f"→ gap fill={int(clean_mask.sum()) + gap_count}(+{gap_count}) "
                f"| total({'+' if total_diff >= 0 else ''}{total_diff}) "
                f"| crack_aff={crack_aff:.3f} bg_aff={bg_aff:.3f} "
                f"| complexity={complexity:.3f}"
            )
    
        avg_complexity = float(np.mean(complexity_list))
        return refined, avg_complexity
    def refine_pseudo_morphology(self, pseudo_label, pseudo_prob,
                                  encoder_feat, crack_class=1):
        import cv2
        import torch.nn.functional as F
    
        B = pseudo_label.shape[0]
        refined = pseudo_label.clone()
        complexity_list = []
    
        for b in range(B):
            lbl        = pseudo_label[b].cpu().numpy().astype(np.uint8)
            prob       = pseudo_prob[b].cpu().numpy()
            feat       = encoder_feat[b]
            crack_mask = (lbl == crack_class).astype(np.uint8)
    
            if crack_mask.sum() == 0:
                complexity_list.append(0.5)
                continue
    
            H, W = lbl.shape
    
            # ── min_crack_length 자동화 ───────────────────────────────────
            size_based = max(3, int(min(H, W) * 0.01))
    
            num_labels_init, _, stats_init, _ = cv2.connectedComponentsWithStats(
                crack_mask, connectivity=8
            )
            if num_labels_init > 1:
                lengths = [
                    max(stats_init[i, cv2.CC_STAT_WIDTH],
                        stats_init[i, cv2.CC_STAT_HEIGHT])
                    for i in range(1, num_labels_init)
                ]
                dist_based       = int(np.percentile(lengths, 20))
                min_crack_length = max(3, max(size_based, dist_based))
            else:
                min_crack_length = size_based
    
            # ── Affinity Map ──────────────────────────────────────────────
            crack_pixel_mask = torch.from_numpy(crack_mask).bool()
            crack_feats      = feat[:, crack_pixel_mask]
    
            if crack_feats.shape[1] == 0:
                complexity_list.append(0.5)
                continue
    
            crack_proto = crack_feats.mean(dim=1, keepdim=True)
    
            if self.crack_proto_bank is not None:
                bank_proto  = self.crack_proto_bank.unsqueeze(1)
                w           = self.proto_reliability
                crack_proto = (
                    w * bank_proto.to(crack_proto.device) +
                    (1 - w) * crack_proto
                )
    
            feat_flat  = feat.view(feat.shape[0], -1)
            proto_norm = F.normalize(crack_proto, dim=0)
    
            affinity = (proto_norm.T @ feat_flat).view(H, W).cpu().numpy()
            affinity = (affinity + 1) / 2
    
            # ── complexity / separation 계산 ──────────────────────────────
            crack_aff    = float(affinity[crack_mask == 1].mean())
            bg_aff       = float(affinity[lbl == 0].mean())
            separation   = float(np.clip(crack_aff - bg_aff, 0, 1))
            complexity   = float(1.0 - separation)
            affinity_mid = (crack_aff + bg_aff) / 2
            complexity_list.append(complexity)
    
            # ── Step 1: gap fill 먼저 ─────────────────────────────────────
            kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated = cv2.dilate(crack_mask, kernel, iterations=2)
    
            affinity_thresh  = crack_aff * 0.8
            high_affinity_bg = (lbl == 0) & (affinity > affinity_thresh)
            gap_fill         = dilated & high_affinity_bg.astype(np.uint8)
            filled_mask      = np.clip(crack_mask + gap_fill, 0, 1).astype(np.uint8)
    
            # ── Step 2: noise 제거 ────────────────────────────────────────
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                filled_mask, connectivity=8
            )
            clean_mask = np.zeros_like(crack_mask)
    
            for i in range(1, num_labels):
                component_pixels = (labels == i)
                length = max(stats[i, cv2.CC_STAT_WIDTH],
                            stats[i, cv2.CC_STAT_HEIGHT])
    
                original_in_comp = int(crack_mask[component_pixels].sum())
                total_in_comp    = int(component_pixels.sum())
                gap_ratio        = 1.0 - (original_in_comp / (total_in_comp + 1e-6))
    
                gap_pixels_in_comp = component_pixels & (gap_fill == 1)
                if gap_pixels_in_comp.sum() > 0:
                    gap_affinity = float(affinity[gap_pixels_in_comp].mean())
                else:
                    gap_affinity = float(affinity[component_pixels].mean())
    
                # 조건 1: gap affinity mid 이하 → noise
                if gap_affinity < affinity_mid:
                    if original_in_comp > 0:
                        clean_mask[component_pixels & (crack_mask == 1)] = 1
                    continue
    
                # 조건 2: separation 기반 gap_weight 자동화
                gap_weight    = 0.5 * (1.0 - separation)
                effective_min = max(
                    min_crack_length * (1.0 + gap_ratio * gap_weight),
                    min_crack_length * 0.5
                )
    
                if length >= effective_min:
                    clean_mask[labels == i] = 1
                else:
                    if original_in_comp > 0:
                        clean_mask[component_pixels & (crack_mask == 1)] = 1
    
            # ── pseudo_label 업데이트 ─────────────────────────────────────
            refined_lbl                  = lbl.copy()
            removed                      = (filled_mask == 1) & (clean_mask == 0)
            refined_lbl[removed]         = 0
            refined_lbl[clean_mask == 1] = crack_class
    
            refined[b] = torch.from_numpy(refined_lbl).to(pseudo_label.device)
    
            # ── 로그 ──────────────────────────────────────────────────────
            gap_count     = int(gap_fill.sum())
            removed_count = int(filled_mask.sum()) - int(clean_mask.sum())
            total_diff    = int(clean_mask.sum()) - int(crack_mask.sum())
    
            print(
                f"[Morph b={b}] "
                f"before={int(crack_mask.sum())} "
                f"→ gap fill={int(filled_mask.sum())}(+{gap_count}) "
                f"→ noise제거={int(clean_mask.sum())}(-{removed_count}) "
                f"| total({'+' if total_diff >= 0 else ''}{total_diff}) "
                f"| crack_aff={crack_aff:.3f} bg_aff={bg_aff:.3f} "
                f"| sep={separation:.3f} min_len={min_crack_length} "
                f"| complexity={complexity:.3f} "
                f"| bank={'O' if self.crack_proto_bank is not None else 'X'}"
            )
    
        avg_complexity = float(np.mean(complexity_list))
        return refined, avg_complexity
    def forward_train(self, img, img_metas, gt_semantic_seg, target_img, target_img_metas, target_gt_semantic_seg):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        import cv2
        import numpy as np
        import torch
        
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training

        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }

        weak_img, weak_target_img = img.clone(), target_img.clone()

        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False

        # ── pseudo label 생성 ──────────────────────────────────────────────────
        ema_target_logits = self.get_ema_model().encode_decode(
            weak_target_img, target_img_metas
        )
        # ── pseudo label 생성 + proj 추출 (backbone 1번) ──────────────────
        ema_target_softmax = torch.softmax(ema_target_logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_target_softmax, dim=1)
        
        # ── pseudo_weight 계산 ────────────────────────────────────────────────
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold)
        ps_large_p[pseudo_label == 255] = False
        ps_size = pseudo_label.numel()
        pseudo_weight = ps_large_p.sum().item() / ps_size
        # pseudo RandomCrop
        if self.pseudo_random_crop:
            weak_target_img, pseudo_label = self.random_crop(weak_target_img, pseudo_label, prod=self.prod)
            if self.regen_pseudo:
                # crop 후 재생성 - 이건 어쩔 수 없이 다시 추출
                with torch.no_grad():
                    feat_x = self.get_ema_model().extract_feat(weak_target_img)
                    
                    ema_target_logits = self.get_ema_model().decode_head.forward(feat_x)
                    if isinstance(ema_target_logits, (list, tuple)):
                        ema_target_logits = ema_target_logits[-1]
                    
                    tgt_proj = self.get_ema_model().auxiliary_head.forward(feat_x)
                    if isinstance(tgt_proj, (list, tuple)):
                        tgt_proj = tgt_proj[0]
                    tgt_proj = tgt_proj.detach()
                ema_target_logits = self.get_ema_model().encode_decode(weak_target_img, target_img_metas)
                ema_target_softmax = torch.softmax(ema_target_logits.detach(), dim=1)
                pseudo_prob, pseudo_label = torch.max(ema_target_softmax, dim=1)
                ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
                ps_size = np.size(np.array(pseudo_label.cpu()))
                pseudo_weight = torch.sum(ps_large_p).item() / ps_size
            target_img = weak_target_img.clone()

        '''
        # ── 호출부 ───────────────────────────────────────────────────────────────
        pseudo_label = self.refine_pseudo_crack(
            image        = target_img,
            pseudo_label = pseudo_label,
            pseudo_prob  = pseudo_prob,
            cur_iter     = self.local_iter,   # 현재 iteration 넘겨주기
        )
        
        pseudo_label = self.refine_pseudo_morphology(
            pseudo_label=pseudo_label,
            pseudo_prob=pseudo_prob,
            crack_class=1,
            min_crack_length=10,
            prob_threshold=0.7,
        )
        '''
        # ── proj 해상도 맞추기 ────────────────────────────────────────────
        H, W = pseudo_label.shape[-2:]
        tgt_proj_up = F.interpolate(
            tgt_proj, size=(H, W), mode='bilinear', align_corners=False
        )
        # 변경
        # ── pseudo label 정제 ─────────────────────────────────────────────
        pseudo_label, complexity = self.refine_pseudo_morphology(
            pseudo_label = pseudo_label,
            pseudo_prob  = pseudo_prob,
            encoder_feat = tgt_proj_up,
            crack_class  = 1,
        )
        '''
        # pseudo_weight에 신뢰도 반영
        reliability   = 1.0 - complexity
        pseudo_weight = pseudo_weight * (0.5 + 0.5 * reliability)
        '''
        # proto bank 업데이트 (신뢰도 높을 때만)
        self.update_proto_bank(
            tgt_proj     = tgt_proj_up,
            pseudo_label = pseudo_label,
            complexity   = complexity,
            crack_class  = 1,
        )

        if self.enable_strong_aug:
            img, gt_semantic_seg = strong_transform(
                strong_parameters,
                data=img,
                target=gt_semantic_seg
            )
            target_img, _ = strong_transform(
                strong_parameters,
                data=target_img,
                target=pseudo_label.unsqueeze(1)
            )
        
        pseudo_weight = pseudo_weight * torch.ones(
            pseudo_label.shape, device=dev)
        #pseudo_weight = pseudo_weight.to(dev)  # device만 맞춤
        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        gt_pixel_weight = torch.ones(pseudo_weight.shape, device=dev)

        ema_source_logits = self.get_ema_model().encode_decode(weak_img, img_metas)
        ema_source_softmax = torch.softmax(ema_source_logits.detach(), dim=1)
        _, source_pseudo_label = torch.max(ema_source_softmax, dim=1)

        weak_gt_semantic_seg = gt_semantic_seg.clone().detach()
        # forward_train 안에서 source width_map 추출 후 loss에 넘기기
        src_width_map = self.extract_width_map_batch(
            weak_gt_semantic_seg.squeeze(1)  # (B,1,H,W) → (B,H,W)
        )
        crack_class = 1
        # source용 crack mask (GT 기반)
        crack_mask_src = (weak_gt_semantic_seg == crack_class).float()    # [B,1,H,W]
        # update distribution
        ema_src_feat, ema_attn_s4, H4, W4 = self.get_ema_model().extract_auxiliary_feat(weak_img, return_attn=True)
        mean = {}
        covariance = {}
        bank = {}
        mean_crack_bins = {}
        bank_crack_bins = {}
        bank_crack_bd = {}
        bank_crack_in = {}

        if self.contrast_mode == 'multiple_select':
            for idx in range(len(self.calc_layers)):
                feat, mask = contrast_preparations(ema_src_feat[idx], weak_gt_semantic_seg, self.enable_avg_pool,
                                                   self.scale_min_ratio, self.num_classes, self.ignore_index)
                self.feat_distributions[idx].update_proto(features=feat.detach(), labels=mask)
                mean[idx] = self.feat_distributions[idx].Ave
                covariance[idx] = self.feat_distributions[idx].CoVariance
                bank[idx] = self.feat_distributions[idx].MemoryBank
        else:  # 'resize_concat' or None
            feat_h, feat_w = int(ema_src_feat.shape[-2]), int(ema_src_feat.shape[-1])
            #print(f" 1. src_width_map - Shape: {src_width_map.shape}, Dim: {src_width_map.dim()}, Dtype: {src_width_map.dtype}")
            # 2. width_map을 [B, 1, H, W]로 만들어서 리사이징 후 다시 [B, H, W]로
            # 폭(px) 데이터이므로 보간 시 값이 왜곡되지 않게 'nearest' 모드를 권장합니다.
            src_width_map = F.interpolate(
                src_width_map, 
                size=(feat_h, feat_w), 
                mode='nearest'
            ).squeeze(1).long() # 다시 정수형으로 변환
            # ------------------------------
            
            # 1. contrast_preparations에서 msk(선택된 픽셀 인덱스)를 받아옴
            feat, mask, msk = contrast_preparations(
                ema_src_feat, 
                weak_gt_semantic_seg, 
                self.enable_avg_pool, 
                self.scale_min_ratio, 
                self.num_classes, 
                self.ignore_index,
                return_idx=True  # 반드시 True로 설정
            )

            # 2. width_map에서 선택된 픽셀들에 해당하는 폭 정보 추출
            # src_width_map은 [B, H, W] 형태이므로 flatten하여 msk로 인덱싱
            width_flat = src_width_map.view(-1)
            layer_width = width_flat[msk]

            # 3. 폭별 할당(Assignment) 계산 (0: Thin, 1: Med, 2: Wide)
            with torch.no_grad():
                crack_only_mask = (mask == self.crack_idx)
                assignments = torch.zeros_like(mask, dtype=torch.long)
                if crack_only_mask.sum() > 0:
                    # c_widths를 float 타입으로 가져옵니다.
                    c_widths = layer_width[crack_only_mask].float() 
                    
                    # 이제 quantile 계산이 가능합니다.
                    t33 = c_widths.quantile(0.33)
                    t66 = c_widths.quantile(0.66)
                    
                    # 할당(assignment) 계산
                    c_assign = torch.where(c_widths < t33, torch.tensor(0, device=c_widths.device), 
                               torch.where(c_widths < t66, torch.tensor(1, device=c_widths.device), 
                               torch.tensor(2, device=c_widths.device)))
                    
                    assignments[crack_only_mask] = c_assign.long()

            # 4. ProtoEstimator 업데이트 (기본 & 하위 프로토타입 모두)
            self.feat_distributions.update_proto(features=feat.detach(), labels=mask)
            self.feat_distributions.update_sub_proto(
                features=feat.detach(), 
                labels=mask, 
                width_assignments=assignments
            )

            # 5. 전역 프로토타입 정보 가져오기
            mean = self.feat_distributions.Ave
            sub_mean = self.feat_distributions.SubAve  # 새로 추가한 SubAve
            covariance = self.feat_distributions.CoVariance
            bank = self.feat_distributions.MemoryBank
        use_bank_now = self.local_iter >= self.start_bank_iter
        src_mode = 'dec'  # stands for ce only
        if self.local_iter >= self.start_distribution_iter:
            src_mode = 'all'  # stands for ce + cl
        # --- 수정 부분 ---

        source_losses = self.get_model().forward_train(img, img_metas, gt_semantic_seg, return_feat=False, return_proj=True,
                                                       mean=mean, covariance=covariance, sub_mean=sub_mean, crack_width_map=src_width_map, bank=bank, mode=src_mode, cur_iter=self.local_iter,   is_source=True, is_target=False)
        src_proj = source_losses.pop('proj_feat')
        source_loss, source_log_vars = self._parse_losses(source_losses)
        log_vars.update(add_prefix(source_log_vars, 'src'))
        total_loss = source_loss
        if self.local_iter >= self.start_distribution_iter:
            # target cl
            pseudo_lbl = pseudo_label.clone()  # pseudo label should not be overwritten
            pseudo_lbl[pseudo_weight == 0.] = self.ignore_index
            pseudo_lbl = pseudo_lbl.unsqueeze(1)
            crack_mask_tgt = (pseudo_lbl == crack_class).float()        # [B,1,H,W]
            # target width_map (pseudo label 기반)
            tgt_width_map = self.extract_width_map_batch(
                pseudo_label  # (B,H,W) - ignore 덮어쓰기 전 원본 pseudo_label 사용
            )

            feat_h, feat_w = int(ema_src_feat.shape[-2]), int(ema_src_feat.shape[-1])
            
            tgt_width_map = F.interpolate(
                tgt_width_map,
                size=(feat_h, feat_w),
                mode='nearest'
            ).squeeze(1).long()
            target_losses = self.get_model().forward_train(target_img, target_img_metas, pseudo_lbl, sub_mean=sub_mean, crack_width_map=tgt_width_map, return_feat=False, return_proj=True, mean=mean, covariance=covariance, bank=bank, mode='aux', cur_iter=self.local_iter, is_source=False, is_target=True)
                        # ✅ target width_map 추출
            
            tgt_proj = target_losses.pop('proj_feat')
            target_loss, target_log_vars = self._parse_losses(target_losses)
            log_vars.update(add_prefix(target_log_vars, 'tgt'))
            total_loss = total_loss + target_loss
            #target_loss.backward()

        local_enable_self_training = \
            self.enable_self_training and \
            (not self.push_off_self_training or self.local_iter >= self.start_distribution_iter)

        # mixed ce (ssl)
        if local_enable_self_training:
            # Apply mixing
            mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
            mix_masks = get_class_masks(gt_semantic_seg)
            '''
            mix_masks = get_crack_mix_masks(
                gt_semantic_seg,
                crack_class=1,
                dilate_kernel=7,
                mode='dilate'
            )
            
            #mix_masks, mix_strategy = get_diverse_crack_mask(gt_semantic_seg, target_img, cur_iter=self.local_iter, max_iter=40000)
            #mix_masks, mix_strategy = get_diverse_crack_mask(gt_semantic_seg, cur_iter=self.local_iter, max_iter=40000)
            #mix_masks, mix_strategy = get_diverse_crack_mask(gt_semantic_seg, max_iter=80000)
            #mix_masks, mix_strategy = get_diverse_crack_mask(gt_semantic_seg, max_iter=80000)
            # ✅ Morphology-aware mask
            masks, strategy = get_diverse_crack_mask(
                label_src        = gt_semantic_seg,   # source label (mix용)
                crack_class      = crack_class,
                cur_iter         = self.local_iter,
                max_iter         = self.max_iters,
                mode             = 'curriculum',
                img_src          = target_img,        # ← target image
                pseudo_prob      = pseudo_prob,       # target pseudo prob
                complexity_ratio = complexity,        # target affinity 기반
            )
            '''
            #mix_masks = get_morphology_aware_crack_mask(
            #    gt_semantic_seg,
            #    thin_crack_thresh=3.0,   # 3px 이하 = thin crack
            #    context_radius=15)       # thin crack 보호 반경
            '''
            masks, strategy = get_diverse_crack_mask(
                label_src        = gt_semantic_seg,
                crack_class      = crack_class,
                cur_iter         = self.local_iter,
                max_iter         = self.max_iters,
                complexity_ratio = complexity,
            )
            '''
            for i in range(batch_size):
                #m = mix_masks[i][0].bool()            # [H,W]  (mask=1 영역이 source를 유지)
                #src = gt_semantic_seg[i][0]           # [H,W]  (0=bg, 1=crack)
                #tgt = pseudo_label[i]                 # [H,W]
                #pw_tgt = pseudo_weight[i]             # [H,W] or [1,H,W] depending on your impl
                #pw_src = gt_pixel_weight[i]           # [H,W] (보통 all-ones)
                strong_parameters['mix'] = mix_masks[i]
                mixed_img[i], mixed_lbl[i] = strong_transform(
                    strong_parameters,
                    data=torch.stack((weak_img[i], weak_target_img[i])),
                    target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i])))
                _, pseudo_weight[i] = strong_transform(
                    strong_parameters,
                    target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
                '''
                # ✅ Width-adaptive weight
                pseudo_weight[i] = get_width_adaptive_weight(
                    pseudo_weight[i].unsqueeze(0),
                    mixed_lbl[i].unsqueeze(0),
                    thin_boost=5.0,
                    wide_boost=2.0,
                    thin_thresh=3.0).squeeze(0)
                '''
            mixed_img = torch.cat(mixed_img)
            mixed_lbl = torch.cat(mixed_lbl).long()
             #✅ 4000iter마다 strategy 터미널 출력 (기존 시각화 출력과 같은 조건)
            mix_losses = self.get_model().forward_train(mixed_img, img_metas, mixed_lbl, pseudo_weight,
                                                        return_feat=False, mode='dec', add_crack_loss=False)
            mix_loss, mix_log_vars = self._parse_losses(mix_losses)
            log_vars.update(add_prefix(mix_log_vars, 'mix'))
            total_loss = total_loss + mix_loss
            #mix_loss.backward()
        from mmengine.logging import MMLogger
        logger = MMLogger.get_current_instance()
        if self.local_iter % self.debug_img_interval == 0:
            '''
            ps_mask = (pseudo_prob >= self.pseudo_threshold)

            coverage = ps_mask.float().mean().item()
        
            crack_ratio = (pseudo_label == 1).float().mean().item()
        
            denom = ps_mask.float().sum().clamp_min(1.0)
            crack_in_selected = ((pseudo_label == 1) & ps_mask).float().sum() / denom
        
            mean_conf = pseudo_prob.mean().item()
            std_conf  = pseudo_prob.std().item()
        
            logger.info(
                f"[Iter {self.local_iter:06d}] "
                f"[PSEUDO] coverage={coverage*100:.2f}% | "
                f"crack_ratio={crack_ratio*100:.4f}% | "
                f"crack_in_selected={crack_in_selected*100:.4f}% | "
                f"conf(mean,std)=({mean_conf:.3f},{std_conf:.3f})"
            )
            '''
            out_dir = os.path.join(self.train_cfg['work_dir'], 'visualize_meta')
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
            '''
            if local_enable_self_training:
                vis_mixed_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)
            
            ema_src_logits = self.get_ema_model().encode_decode(weak_img, img_metas)
            ema_softmax = torch.softmax(ema_src_logits.detach(), dim=1)
            _, src_pseudo_label = torch.max(ema_softmax, dim=1)
            
            for j in range(batch_size):
                rows, cols = 2, 6
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        'hspace': 0.1,
                        'wspace': 0,
                        'top': 0.95,
                        'bottom': 0,
                        'right': 1,
                        'left': 0
                    },
                )
            
                subplotimg(axs[0][0], vis_img[j], f'{img_metas[j]["ori_filename"]}')
                subplotimg(
                    axs[1][0],
                    vis_trg_img[j],
                    f'{os.path.basename(target_img_metas[j]["ori_filename"]).replace("_leftImg8bit", "")}'
                )
            
                subplotimg(
                    axs[0][1],
                    src_pseudo_label[j],
                    'Source Pseudo Label',
                    cmap='cityscapes',
                    nc=self.num_classes
                )
                subplotimg(
                    axs[1][1],
                    pseudo_label[j],
                    'Target Pseudo Label',
                    cmap='cityscapes',
                    nc=self.num_classes
                )
            
                # 추가: refined pseudo label
                subplotimg(
                    axs[1][2],
                    pseudo_lbl_refined[j],
                    'Refined Target Pseudo',
                    cmap='cityscapes',
                    nc=self.num_classes
                )
            
                subplotimg(
                    axs[0][2],
                    gt_semantic_seg[j],
                    'Source Seg GT',
                    cmap='cityscapes',
                    nc=self.num_classes
                )
            
                if target_gt_semantic_seg.dim() > 1:
                    subplotimg(
                        axs[0][3],
                        target_gt_semantic_seg[j],
                        'Target Seg GT',
                        cmap='cityscapes',
                        nc=self.num_classes
                    )
            
                # 추가: refined crack mask도 같이 보면 좋음
                subplotimg(
                    axs[1][3],
                    crack_mask_tgt_refined[j][0] if crack_mask_tgt_refined.dim() == 4 else crack_mask_tgt_refined[j],
                    'Refined Crack Mask',
                    cmap='gray'
                )
            
                subplotimg(
                    axs[0][4],
                    pseudo_weight[j],
                    'Pseudo W.',
                    vmin=0,
                    vmax=1
                )
            
                if local_enable_self_training:
                    subplotimg(
                        axs[1][4],
                        mix_masks[j][0],
                        'Mixed Mask',
                        cmap='gray'
                    )
                    subplotimg(
                        axs[0][5],
                        vis_mixed_img[j],
                        'Mixed ST Image'
                    )
                    subplotimg(
                        axs[1][5],
                        mixed_lbl[j],
                        'Mixed ST Label',
                        cmap='cityscapes',
                        nc=self.num_classes
                    )
            
                for ax in axs.flat:
                    ax.axis('off')
            
                plt.savefig(
                    os.path.join(out_dir, f'{(self.local_iter + 1):06d}_{j}.png')
                )
                plt.close()
            '''
            if local_enable_self_training:
                vis_mixed_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)
            ema_src_logits = self.get_ema_model().encode_decode(weak_img, img_metas)
            ema_softmax = torch.softmax(ema_src_logits.detach(), dim=1)
            _, src_pseudo_label = torch.max(ema_softmax, dim=1)
            for j in range(batch_size):
                rows, cols = 2, 5
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        'hspace': 0.1,
                        'wspace': 0,
                        'top': 0.95,
                        'bottom': 0,
                        'right': 1,
                        'left': 0
                    },
                )
                subplotimg(axs[0][0], vis_img[j], f'{img_metas[j]["ori_filename"]}')
                subplotimg(axs[1][0], vis_trg_img[j],
                           f'{os.path.basename(target_img_metas[j]["ori_filename"]).replace("_leftImg8bit", "")}')
                subplotimg(
                    axs[0][1],
                    src_pseudo_label[j],
                    'Source Pseudo Label',
                    cmap='cityscapes',
                    nc=self.num_classes)
                subplotimg(
                    axs[1][1],
                    pseudo_label[j],
                    'Target Pseudo Label',
                    cmap='cityscapes',
                    nc=self.num_classes)
                subplotimg(
                    axs[0][2],
                    gt_semantic_seg[j],
                    'Source Seg GT',
                    cmap='cityscapes',
                    nc=self.num_classes)
                if target_gt_semantic_seg.dim() > 1:
                    subplotimg(
                        axs[1][2],
                        target_gt_semantic_seg[j],
                        'Target Seg GT',
                        cmap='cityscapes',
                        nc=self.num_classes
                    )
                subplotimg(
                    axs[0][3], pseudo_weight[j], 'Pseudo W.', vmin=0, vmax=1)
                if local_enable_self_training:
                    subplotimg(
                        axs[1][3],
                        mix_masks[j][0],
                        'Mixed Mask',
                        cmap='gray'
                    )
                    subplotimg(
                        axs[0][4],
                        vis_mixed_img[j],
                        'Mixed ST Image')
                    subplotimg(
                        axs[1][4],
                        mixed_lbl[j],
                        'Mixed ST Label',
                        cmap='cityscapes',
                        nc=self.num_classes
                    )
                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir,
                                 f'{(self.local_iter + 1):06d}_{j}.png'))
                plt.close()
           
        self.local_iter += 1
        return dict(
            loss=total_loss,
            log_vars=log_vars,
            num_samples=img.size(0),
        )
