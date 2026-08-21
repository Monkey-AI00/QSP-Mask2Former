#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第二阶段：ShapePriorAdapter

目标：把 Phase-1 的“静态形状先验”(mean mask) 动态对齐到每个 ROI 的姿态，然后用门控方式融合进 ROI 特征图。

模块由三部分组成：
1) Angle Predictor：从 ROI 特征预测旋转角度（弧度）
2) Spatial Transformer：用 affine_grid + grid_sample 把静态 prior 旋转成动态 prior
3) Gated Fusion：预测 gate，融合 (x, gate * prior) 并残差输出

输入/输出约定：
- 输入 x_features: (B, C, H, W) ，通常来自 ROIAlign/PointRend 的 _roi_pooler
- 输出 x_out: (B, C, H, W)
- 输出 pred_angle: (B, 1) 旋转角（弧度），可用于第三阶段设计辅助损失（可选）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _load_prior_as_tensor(prior: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
    """
    加载 Phase-1 生成的 prior（.npy）为 float tensor，shape=[1,1,H,W]。
    """
    if isinstance(prior, torch.Tensor):
        t = prior.detach().float()
    elif isinstance(prior, np.ndarray):
        t = torch.from_numpy(prior).float()
    elif isinstance(prior, str):
        arr = np.load(prior)
        t = torch.from_numpy(arr).float()
    else:
        raise TypeError(f"Unsupported prior type: {type(prior)}")

    if t.ndim == 2:
        t = t.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    elif t.ndim == 3:
        # [1,H,W] or [C,H,W] -> assume single channel
        if t.shape[0] != 1:
            t = t[:1]
        t = t.unsqueeze(0)  # [1,1,H,W]
    elif t.ndim == 4:
        # [N,C,H,W] -> take first if needed
        if t.shape[0] != 1:
            t = t[:1]
        if t.shape[1] != 1:
            t = t[:, :1]
    else:
        raise ValueError(f"prior tensor must be 2D/3D/4D, got shape={tuple(t.shape)}")

    # 如果 prior 是 0~255 的图，归一化到 0~1
    if float(t.max()) > 1.0:
        t = t / 255.0
    return t.contiguous()


class AnglePredictor(nn.Module):
    """
    从 ROI 特征预测旋转角度（弧度）。默认初始化为 0（从“不旋转”开始学）。
    """

    def __init__(self, in_channels: int, spatial_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * spatial_size * spatial_size, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

        # 初始化最后一层为 0：初始 pred_angle=0
        nn.init.constant_(self.net[-1].weight, 0.0)
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B,C,H,W) -> (B,1)
        return self.net(x)


class SpatialPriorTransformer(nn.Module):
    """
    根据 pred_angle 动态旋转先验（grid_sample）。
    """

    def __init__(self, prior_tensor: torch.Tensor, prior_size: int):
        super().__init__()

        if prior_tensor.shape[-1] != prior_size or prior_tensor.shape[-2] != prior_size:
            prior_tensor = F.interpolate(prior_tensor, size=(prior_size, prior_size), mode="bilinear", align_corners=False)
        self.register_buffer("mean_shape", prior_tensor)  # [1,1,H,W]

    def forward(self, pred_angle: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            pred_angle: (B,1) radians
            out_hw: (H,W) 输出空间尺寸（通常等于 ROI 特征尺寸）
        Returns:
            rotated_prior: (B,1,H,W)
        """
        B = pred_angle.shape[0]
        H, W = int(out_hw[0]), int(out_hw[1])
        if B == 0:
            # 某些评测图片可能没有任何有效 ROI，affine_grid 不支持 batch=0。
            # 这里直接返回空 prior batch，保持下游张量维度一致。
            return self.mean_shape.new_zeros((0, 1, H, W))

        cos_theta = torch.cos(pred_angle)
        sin_theta = torch.sin(pred_angle)
        zeros = torch.zeros_like(pred_angle)

        # [[cos, -sin, 0], [sin, cos, 0]]
        r1 = torch.cat([cos_theta, -sin_theta, zeros], dim=1)  # (B,3)
        r2 = torch.cat([sin_theta, cos_theta, zeros], dim=1)  # (B,3)
        affine = torch.cat([r1.unsqueeze(1), r2.unsqueeze(1)], dim=1)  # (B,2,3)

        grid = F.affine_grid(affine, size=(B, 1, H, W), align_corners=False)
        prior_batch = self.mean_shape.expand(B, -1, -1, -1)  # (B,1,priorH,priorW)
        if prior_batch.shape[-2:] != (H, W):
            prior_batch = F.interpolate(prior_batch, size=(H, W), mode="bilinear", align_corners=False)

        rotated = F.grid_sample(prior_batch, grid, align_corners=False, padding_mode="zeros")
        return rotated


class GatedFusion(nn.Module):
    """
    门控融合：预测 gate，并融合 concat([x, gate * prior]) -> 1x1 conv -> 残差。
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.attention_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.fusion_conv = nn.Conv2d(in_channels + 1, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, prior: torch.Tensor, *, return_gate: bool = False):
        gate = torch.sigmoid(self.attention_conv(x))  # (B,1,H,W)
        fused = torch.cat([x, gate * prior], dim=1)  # (B,C+1,H,W)
        out = self.fusion_conv(fused)
        y = out + x
        if return_gate:
            return y, gate
        return y


@dataclass
class ShapePriorAdapterOutput:
    x_out: torch.Tensor
    pred_angle: torch.Tensor
    rotated_prior: torch.Tensor
    gate_map: torch.Tensor


class ShapePriorAdapter(nn.Module):
    """
    总模块：AnglePredictor + SpatialPriorTransformer + GatedFusion
    """

    def __init__(self, prior: Union[str, np.ndarray, torch.Tensor], in_channels: int = 256, prior_size: int = 28):
        """
        Args:
            prior: Phase 1 生成的 .npy 路径，或 numpy/torch 形式的 prior
            in_channels: 输入特征通道数（FPN通常 256）
            prior_size: ROIAlign/roi_pooler 输出尺寸（通常 14 或 28）
        """
        super().__init__()

        prior_tensor = _load_prior_as_tensor(prior)  # [1,1,H,W]
        self.prior_size = int(prior_size)

        self.angle_predictor = AnglePredictor(in_channels=in_channels, spatial_size=self.prior_size)
        self.spatial_transformer = SpatialPriorTransformer(prior_tensor=prior_tensor, prior_size=self.prior_size)
        self.gated_fusion = GatedFusion(in_channels=in_channels)

    def forward(self, x_features: torch.Tensor, return_debug: bool = False):
        """
        Args:
            x_features: (B,C,H,W)
            return_debug: True 时额外返回 rotated_prior 方便调试/可视化
        Returns:
            默认：x_out, pred_angle
            return_debug=True：ShapePriorAdapterOutput
        """
        B, C, H, W = x_features.shape
        if B == 0:
            # 无 ROI 时直接透传，避免进入 STN 的 affine_grid(batch=0) 报错。
            pred_angle = x_features.new_zeros((0, 1))
            rotated_prior = x_features.new_zeros((0, 1, H, W))
            if return_debug:
                gate = x_features.new_zeros((0, 1, H, W))
                return ShapePriorAdapterOutput(
                    x_out=x_features,
                    pred_angle=pred_angle,
                    rotated_prior=rotated_prior,
                    gate_map=gate,
                )
            return x_features, pred_angle
        if H != self.prior_size or W != self.prior_size:
            # 这里严格一些：ROI 特征尺寸应与 prior_size 一致，否则 AnglePredictor 的 flatten 维度会不匹配
            raise ValueError(
                f"x_features spatial size {(H, W)} != prior_size {self.prior_size}. "
                "请把 prior_size 设为 ROIAlign/POOLER_RESOLUTION 一致的值。"
            )

        pred_angle = self.angle_predictor(x_features)  # (B,1)
        rotated_prior = self.spatial_transformer(pred_angle, out_hw=(H, W))  # (B,1,H,W)
        if return_debug:
            x_out, gate = self.gated_fusion(x_features, rotated_prior, return_gate=True)  # (B,C,H,W), (B,1,H,W)
        else:
            x_out = self.gated_fusion(x_features, rotated_prior)  # (B,C,H,W)
            gate = None

        if return_debug:
            assert gate is not None
            return ShapePriorAdapterOutput(x_out=x_out, pred_angle=pred_angle, rotated_prior=rotated_prior, gate_map=gate)
        return x_out, pred_angle


