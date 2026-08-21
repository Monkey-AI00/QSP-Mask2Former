#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第三阶段：把 ShapePriorAdapter 注入到 PointRend 的 Coarse Mask 路径中。

说明：
- PointRend 的 ROI_MASK_HEAD 默认是 `PointRendMaskHead`（不是 MaskRCNNConvUpsampleHead）。
- `PointRendMaskHead.forward()` 内部会先通过 `_roi_pooler()` 提取 ROI 特征，再交给 `self.coarse_head(...)`
  生成 coarse mask，随后 point head 才做细化。
- 强光导致 coarse 阶段断裂时，后续 PointRend 很难“救回来”，因此我们在 coarse 之前注入先验修复特征。

用法：
1) 在训练脚本最上面加入：`import custom_heads`（触发注册）
2) 设置：`cfg.MODEL.ROI_MASK_HEAD.NAME = "ShapeAwareCoarseMaskHead"`
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np

from detectron2.layers import ShapeSpec
from detectron2.modeling import ROI_MASK_HEAD_REGISTRY
from detectron2.projects.point_rend.mask_head import PointRendMaskHead

# 第二阶段模块
from detectron2.projects.point_rend import ShapePriorAdapter


def _default_prior_path() -> str:
    # 默认指向工作区 outputs/plug_prior/plug_canonical_prior.npy（Phase-1 输出）
    # 优先使用 WORKSPACE_ROOT，其次按当前文件路径回溯到 workspace 根目录。
    ws = os.environ.get("WORKSPACE_ROOT", "").strip()
    if ws:
        return os.path.abspath(os.path.join(ws, "outputs", "plug_prior", "plug_canonical_prior.npy"))

    here = os.path.dirname(os.path.abspath(__file__))  # .../workspace/src/detectron2/projects/CV/train
    return os.path.abspath(
        os.path.join(here, "..", "..", "..", "..", "..", "outputs", "plug_prior", "plug_canonical_prior.npy")
    )


@ROI_MASK_HEAD_REGISTRY.register()
class ShapeAwareCoarseMaskHead(PointRendMaskHead):
    """
    Shape-aware 的 PointRend ROI mask head：
    - 继承 PointRendMaskHead，保留原有 coarse + point refinement 逻辑
    - 在生成 coarse mask 之前，用 ShapePriorAdapter 对 ROI 特征做“洞补全/结构约束”
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__(cfg, input_shape)

        # ROI 特征通道数：按 PointRend 的 roi_pooler_in_features 做 concat 后的通道数
        in_channels = int(np.sum([input_shape[f].channels for f in self.roi_pooler_in_features]))
        prior_size = int(self.roi_pooler_size)  # 通常是 cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION（默认 14）

        prior_path = os.environ.get("SHAPE_PRIOR_PATH", "").strip() or _default_prior_path()
        if not os.path.exists(prior_path):
            raise FileNotFoundError(
                f"Shape prior file not found: {prior_path}\n"
                "请先运行 Phase-1 生成 .npy，或设置环境变量 SHAPE_PRIOR_PATH 指向它。"
            )

        self.shape_adapter = ShapePriorAdapter(
            prior=prior_path,
            in_channels=in_channels,
            prior_size=prior_size,
        )

        # Optional debug cache (enabled by env var in visualization scripts)
        self._shape_debug: bool = os.environ.get("SHAPE_PRIOR_DEBUG", "").strip().lower() in ("1", "true", "yes", "y", "on")
        self._shape_debug_top_idx: Optional[int] = None
        self.shape_debug_cache: Dict[str, object] = {}

        print("✅ [ShapeAwareCoarseMaskHead] Successfully Initialized with Shape Prior!")
        print(f"   - prior_path: {prior_path}")
        print(f"   - in_channels: {in_channels}, prior_size: {prior_size}")

    def forward(self, features, instances):
        """
        与 PointRendMaskHead.forward 基本一致，仅在 coarse_head 前加入 ShapePriorAdapter。
        """
        if self.training:
            proposal_boxes = [x.proposal_boxes for x in instances]
            roi_features = self._roi_pooler(features, proposal_boxes)  # (R,C,H,W)
            roi_features, pred_angle = self.shape_adapter(roi_features)

            coarse_mask = self.coarse_head(roi_features)
            # 第三阶段先不加角度损失，避免改训练逻辑（如需可在后续阶段加入辅助 loss）
            _ = pred_angle

            # 复用 PointRendMaskHead 的后续逻辑：mask_rcnn_loss + point head 细化
            from detectron2.modeling.roi_heads.mask_head import mask_rcnn_loss
            from detectron2.projects.point_rend.point_head import roi_mask_point_loss

            losses = {"loss_mask": mask_rcnn_loss(coarse_mask, instances)}
            if not self.mask_point_on:
                return losses

            point_coords, point_labels = self._sample_train_points(coarse_mask, instances)
            point_fine_grained_features = self._point_pooler(features, proposal_boxes, point_coords)
            point_logits = self._get_point_logits(point_fine_grained_features, point_coords, coarse_mask)
            losses["loss_mask_point"] = roi_mask_point_loss(point_logits, instances, point_labels)
            return losses
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            roi_features = self._roi_pooler(features, pred_boxes)
            if self._shape_debug and len(instances) == 1 and hasattr(instances[0], "scores") and len(instances[0]) > 0:
                # pick top-scoring instance in this (single) image
                try:
                    self._shape_debug_top_idx = int(instances[0].scores.argmax().item())
                except Exception:
                    self._shape_debug_top_idx = 0
            else:
                self._shape_debug_top_idx = None

            if self._shape_debug:
                self.shape_debug_cache = {}
            raw_coarse_mask = None
            if self._shape_debug:
                # Raw branch: apply the original coarse head to the unrefined
                # ROI features before ShapePriorAdapter changes them.
                raw_coarse_mask = self.coarse_head(roi_features)
                if len(instances) == 1:
                    # Run the complete PointRend subdivision on a copy of the
                    # instances so the raw branch does not overwrite outputs.
                    raw_instances = [instances[0][:]]
                    self._subdivision_inference(
                        features,
                        raw_coarse_mask,
                        raw_instances,
                        debug_prefix="raw_",
                    )
            if self._shape_debug:
                debug_out = self.shape_adapter(roi_features, return_debug=True)
                roi_features = debug_out.x_out
                # store only top instance to keep memory small
                idx = self._shape_debug_top_idx
                if idx is None:
                    idx = 0
                self.shape_debug_cache.update({
                    "pred_angle": debug_out.pred_angle[idx : idx + 1].detach().float().cpu(),
                    "rotated_prior": debug_out.rotated_prior[idx : idx + 1].detach().float().cpu(),
                    "gate_map": debug_out.gate_map[idx : idx + 1].detach().float().cpu(),  # (1,1,H,W)
                })
                # refined feature (channel mean), for visualization only
                try:
                    fm = debug_out.x_out[idx : idx + 1].detach().float().mean(dim=1, keepdim=True).cpu()  # (1,1,H,W)
                    self.shape_debug_cache["refined_feature_mean"] = fm
                except Exception:
                    pass
            else:
                roi_features, _pred_angle = self.shape_adapter(roi_features)
            coarse_mask = self.coarse_head(roi_features)
            if self._shape_debug:
                idx = self._shape_debug_top_idx
                if idx is None:
                    idx = 0
                if raw_coarse_mask is not None:
                    self.shape_debug_cache["raw_coarse_mask_logits"] = (
                        raw_coarse_mask[idx : idx + 1].detach().float().cpu()
                    )
                self.shape_debug_cache["coarse_mask_logits"] = coarse_mask[idx : idx + 1].detach().float().cpu()
            return self._subdivision_inference(features, coarse_mask, instances)

    def _subdivision_inference(self, features, mask_representations, instances, debug_prefix: str = ""):
        """
        Same as PointRendMaskHead._subdivision_inference, but optionally cache
        upsample intermediates for visualization.
        """
        if not self._shape_debug:
            return super()._subdivision_inference(features, mask_representations, instances)

        # ---- copied from PointRendMaskHead._subdivision_inference with small debug taps ----
        from detectron2.layers import interpolate
        from detectron2.modeling.roi_heads.mask_head import mask_rcnn_inference
        from detectron2.projects.point_rend.point_features import (
            generate_regular_grid_point_coords,
            get_uncertain_point_coords_on_grid,
        )

        pred_boxes = [x.pred_boxes for x in instances]
        # best-effort: class tensor exists in inference
        pred_classes = None
        if len(instances) == 1 and hasattr(instances[0], "pred_classes"):
            from detectron2.layers import cat

            pred_classes = cat([x.pred_classes for x in instances])

        mask_logits = None
        upsample_k = 0
        debug_idx = self._shape_debug_top_idx if self._shape_debug_top_idx is not None else 0

        # +1 include initial step
        for step_i in range(self.mask_point_subdivision_steps + 1):
            if mask_logits is None:
                point_coords = generate_regular_grid_point_coords(
                    (pred_classes.size(0) if pred_classes is not None else 0) or mask_representations.shape[0],
                    self.mask_point_subdivision_init_resolution,
                    pred_boxes[0].device,
                )
            else:
                mask_logits = interpolate(mask_logits, scale_factor=2, mode="bilinear", align_corners=False)
                upsample_k += 1

                # uncertainty map
                if pred_classes is None:
                    # fallback: class-agnostic uncertainty
                    uncertainty_map = -mask_logits.abs()
                else:
                    from detectron2.projects.point_rend.mask_head import calculate_uncertainty

                    uncertainty_map = calculate_uncertainty(mask_logits, pred_classes)
                point_indices, point_coords = get_uncertain_point_coords_on_grid(
                    uncertainty_map, self.mask_point_subdivision_num_points
                )

            # Run the point head
            fine_grained_features = self._point_pooler(features, pred_boxes, point_coords)
            point_logits = self._get_point_logits(fine_grained_features, point_coords, mask_representations)

            if mask_logits is None:
                R, C, _ = point_logits.shape
                mask_logits = point_logits.reshape(
                    R,
                    C,
                    self.mask_point_subdivision_init_resolution,
                    self.mask_point_subdivision_init_resolution,
                )
                # empty boxes
                if mask_logits.shape[0] == 0:
                    mask_rcnn_inference(mask_logits, instances)
                    return instances
            else:
                R, C, H, W = mask_logits.shape
                point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
                mask_logits = (
                    mask_logits.reshape(R, C, H * W).scatter_(2, point_indices, point_logits).view(R, C, H, W)
                )
                # store refined mask at this upsample resolution
                try:
                    self.shape_debug_cache[f"{debug_prefix}mask_logits_upsample{upsample_k}"] = (
                        mask_logits[debug_idx : debug_idx + 1].detach().float().cpu()
                    )
                except Exception:
                    pass

        # final
        try:
            self.shape_debug_cache[f"{debug_prefix}mask_logits_final"] = (
                mask_logits[debug_idx : debug_idx + 1].detach().float().cpu()
            )
        except Exception:
            pass

        mask_rcnn_inference(mask_logits, instances)
        return instances


