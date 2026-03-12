#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成“标准形状先验”(Canonical Shape Prior / Mean Mask)：
- 从 COCO 标注中读取实例 mask
- 按 bbox 裁剪并缩放到固定大小
- 使用 PCA/图像矩估计主轴方向并旋转对齐
- 用上下半区质量(像素和)解决 180° 头尾翻转歧义
- 对齐后的 mask 累加取平均，得到 soft prior (float32, [H,W])

输出：
- .npy (默认保存均值 mask)
- 可选：输出可视化 png（适合无 GUI 环境）

示例：
  python generate_mean_shape.py \
    --ann /path/to/train_coco.json \
    --output plug_canonical_prior.npy \
    --target-size 28 28 \
    --temp-size 64 64 \
    --pad 10 \
    --vis plug_prior.png
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable, List, Optional, Tuple

import numpy as np
from pycocotools.coco import COCO

# OpenCV 优先；若环境没装，则自动使用 scipy + PIL 做后备实现
try:
    import cv2  # type: ignore

    _HAS_CV2 = True
except Exception:
    cv2 = None  # type: ignore
    _HAS_CV2 = False

from PIL import Image
from scipy import ndimage


def get_orientation(mask: np.ndarray) -> float:
    """
    使用 PCA/图像矩计算 mask 主轴角度（单位：度）
    返回值范围约为 [-180, 180]。
    """
    y, x = np.nonzero(mask)
    if len(x) == 0:
        return 0.0

    data = np.vstack([x, y]).T.astype(np.float32)
    mean = np.mean(data, axis=0, keepdims=True)
    centered = data - mean

    cov = np.cov(centered.T)
    # cov 可能出现极端情况下的数值问题，做一个小的保护
    if not np.all(np.isfinite(cov)):
        return 0.0

    evals, evecs = np.linalg.eig(cov)
    order = np.argsort(evals)[::-1]
    major_axis = evecs[:, order[0]]
    angle = float(np.degrees(np.arctan2(major_axis[1], major_axis[0])))
    return angle


def rotate_mask(mask: np.ndarray, angle_deg: float) -> np.ndarray:
    """围绕中心旋转 mask（float/uint 均可），输出与输入同尺寸。"""
    if _HAS_CV2:
        h, w = mask.shape[:2]
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)  # type: ignore[attr-defined]
        rotated = cv2.warpAffine(  # type: ignore[attr-defined]
            mask,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,  # type: ignore[attr-defined]
            borderMode=cv2.BORDER_CONSTANT,  # type: ignore[attr-defined]
            borderValue=0,
        )
        return rotated

    # scipy 后备：围绕中心旋转，保持原尺寸，常数 0 填充
    rotated = ndimage.rotate(mask, angle=angle_deg, reshape=False, order=1, mode="constant", cval=0.0)
    return rotated.astype(mask.dtype, copy=False)


def rotate_180(mask: np.ndarray) -> np.ndarray:
    if _HAS_CV2:
        return cv2.rotate(mask, cv2.ROTATE_180)  # type: ignore[attr-defined]
    return np.rot90(mask, 2)


def resize_mask(mask: np.ndarray, size_wh: Tuple[int, int]) -> np.ndarray:
    """将 mask resize 到 (W,H)。优先 OpenCV，否则 PIL。"""
    w, h = int(size_wh[0]), int(size_wh[1])
    if _HAS_CV2:
        return cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA)  # type: ignore[attr-defined]

    # PIL 后备：用 bilinear（对 soft mask 合理；对二值也可）
    arr = mask.astype(np.float32, copy=False)
    img = Image.fromarray(arr, mode="F")
    img2 = img.resize((w, h), resample=Image.BILINEAR)
    return np.array(img2, dtype=np.float32)


def align_mask(mask: np.ndarray, prefer_bottom_heavy: bool = True) -> np.ndarray:
    """
    将 mask 旋转到“竖直”方向，并通过上下半区质量解决 180° 翻转歧义。
    - prefer_bottom_heavy=True 表示把更“重”的一头放到下方。
    """
    angle = get_orientation(mask)
    rotate_angle = -angle + 90.0  # 目标主轴竖直
    rotated = rotate_mask(mask, rotate_angle)

    h = rotated.shape[0]
    top_mass = float(np.sum(rotated[: h // 2, :]))
    bottom_mass = float(np.sum(rotated[h // 2 :, :]))

    # 统一头尾：默认“重的一头朝下”
    if prefer_bottom_heavy:
        if top_mass > bottom_mass:
            rotated = rotate_180(rotated)
    else:
        if bottom_mass > top_mass:
            rotated = rotate_180(rotated)

    return rotated


def safe_crop_with_pad(mask_full: np.ndarray, bbox_xywh: Iterable[float], pad: int) -> Optional[np.ndarray]:
    """按 bbox 裁剪，并加入 pad；裁剪结果为空则返回 None。"""
    x, y, w, h = [int(round(v)) for v in bbox_xywh]
    if w <= 0 or h <= 0:
        return None

    H, W = mask_full.shape[:2]
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(W, x + w + pad), min(H, y + h + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = mask_full[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def get_cat_ids(coco: COCO, category_names: Optional[List[str]]) -> List[int]:
    if not category_names:
        return coco.getCatIds()
    cat_ids: List[int] = []
    for name in category_names:
        ids = coco.getCatIds(catNms=[name])
        cat_ids.extend(ids)
    # 去重保持顺序
    seen = set()
    out: List[int] = []
    for cid in cat_ids:
        if cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


def collect_aligned_masks(
    ann_file: str,
    target_size: Tuple[int, int] = (28, 28),
    temp_size: Tuple[int, int] = (64, 64),
    pad: int = 10,
    min_wh: int = 5,
    category_names: Optional[List[str]] = None,
    prefer_bottom_heavy: bool = True,
    max_instances: Optional[int] = None,
    log_every: int = 100,
) -> np.ndarray:
    coco = COCO(ann_file)
    cat_ids = get_cat_ids(coco, category_names)
    # 重要：pycocotools 的 getImgIds(catIds=[...]) 在传入多个类别时会做“交集”
    # 这会导致像你这种 categories 里有空类/无标注类时，img_ids 直接变成空集合。
    # 这里改为：先取全量图片，再在 getAnnIds 时按 cat_ids 过滤，从而得到“并集/不丢图”的效果。
    img_ids = coco.getImgIds()

    aligned_masks: List[np.ndarray] = []

    for i, img_id in enumerate(img_ids):
        ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=cat_ids if cat_ids else None, iscrowd=None)
        anns = coco.loadAnns(ann_ids)

        for ann in anns:
            if max_instances is not None and len(aligned_masks) >= max_instances:
                break
            if "bbox" not in ann:
                continue

            mask_full = coco.annToMask(ann).astype(np.float32)
            crop = safe_crop_with_pad(mask_full, ann["bbox"], pad=pad)
            if crop is None:
                continue

            h, w = crop.shape[:2]
            if w < min_wh or h < min_wh:
                continue

            # 先缩放到 temp_size（对齐阶段更稳定）
            norm = resize_mask(crop, (temp_size[0], temp_size[1]))
            aligned = align_mask(norm, prefer_bottom_heavy=prefer_bottom_heavy)

            # 再缩放到最终 size（与 ROIAlign 等固定分辨率一致）
            final = resize_mask(aligned, (target_size[0], target_size[1]))

            aligned_masks.append(final.astype(np.float32))

        if max_instances is not None and len(aligned_masks) >= max_instances:
            break
        if log_every > 0 and (i % log_every == 0):
            # i 从 0 开始计数，这里用 i+1 让进度更直观
            print(f"已处理图片 {i+1}/{len(img_ids)}，累计实例 {len(aligned_masks)}")

    if not aligned_masks:
        raise RuntimeError("未统计到任何有效实例 mask，请检查类别过滤、标注文件路径与 bbox/mask 是否正常。")
    return np.stack(aligned_masks, axis=0).astype(np.float32)


def build_mean_shape(
    ann_file: str,
    target_size: Tuple[int, int] = (28, 28),
    temp_size: Tuple[int, int] = (64, 64),
    pad: int = 10,
    min_wh: int = 5,
    category_names: Optional[List[str]] = None,
    prefer_bottom_heavy: bool = True,
    max_instances: Optional[int] = None,
    log_every: int = 100,
) -> Tuple[np.ndarray, int]:
    aligned_masks = collect_aligned_masks(
        ann_file=ann_file,
        target_size=target_size,
        temp_size=temp_size,
        pad=pad,
        min_wh=min_wh,
        category_names=category_names,
        prefer_bottom_heavy=prefer_bottom_heavy,
        max_instances=max_instances,
        log_every=log_every,
    )
    mean_shape = aligned_masks.mean(axis=0)
    return mean_shape.astype(np.float32), int(aligned_masks.shape[0])


def _run_kmeans(features: np.ndarray, k: int, seed: int = 42, max_iter: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    n = int(features.shape[0])
    if k <= 0 or k > n:
        raise ValueError(f"k must be in [1, {n}], got {k}")
    rng = np.random.default_rng(seed)
    centers = features[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros((n,), dtype=np.int64)

    for _ in range(max_iter):
        distances = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for i in range(k):
            mask = labels == i
            if np.any(mask):
                centers[i] = features[mask].mean(axis=0)
            else:
                centers[i] = features[rng.integers(0, n)]
    return centers, labels


def build_prior_bank(
    ann_file: str,
    bank_size: int,
    target_size: Tuple[int, int] = (28, 28),
    temp_size: Tuple[int, int] = (64, 64),
    pad: int = 10,
    min_wh: int = 5,
    category_names: Optional[List[str]] = None,
    prefer_bottom_heavy: bool = True,
    max_instances: Optional[int] = None,
    log_every: int = 100,
    seed: int = 42,
) -> Tuple[np.ndarray, int]:
    aligned_masks = collect_aligned_masks(
        ann_file=ann_file,
        target_size=target_size,
        temp_size=temp_size,
        pad=pad,
        min_wh=min_wh,
        category_names=category_names,
        prefer_bottom_heavy=prefer_bottom_heavy,
        max_instances=max_instances,
        log_every=log_every,
    )
    num_masks, h, w = aligned_masks.shape
    if bank_size <= 1:
        return aligned_masks.mean(axis=0).astype(np.float32)[None, ...], int(num_masks)

    # Keep template 0 as the global mean for backward compatibility.
    num_cluster_templates = min(int(bank_size) - 1, num_masks)
    mean_shape = aligned_masks.mean(axis=0).astype(np.float32)
    flat = aligned_masks.reshape(num_masks, -1)
    _, labels = _run_kmeans(flat, num_cluster_templates, seed=seed)

    cluster_means: List[np.ndarray] = []
    cluster_sizes: List[int] = []
    for i in range(num_cluster_templates):
        members = aligned_masks[labels == i]
        if members.shape[0] == 0:
            continue
        cluster_means.append(members.mean(axis=0).astype(np.float32))
        cluster_sizes.append(int(members.shape[0]))

    order = np.argsort(np.asarray(cluster_sizes))[::-1]
    bank = [mean_shape]
    for idx in order:
        bank.append(cluster_means[int(idx)])

    bank_arr = np.stack(bank, axis=0).astype(np.float32)
    return bank_arr.reshape(-1, h, w), int(num_masks)


def save_vis_png(mean_shape: np.ndarray, vis_path: str, cmap_name: str = "jet") -> None:
    """
    将 [H,W] 的 float mask 可视化成 png（0~1 归一化 -> colormap）。
    使用 OpenCV 写文件，避免依赖 matplotlib + GUI。
    """
    os.makedirs(os.path.dirname(os.path.abspath(vis_path)), exist_ok=True)

    if mean_shape.ndim == 2:
        shapes = mean_shape[None, ...]
    elif mean_shape.ndim == 3:
        shapes = mean_shape
    else:
        raise ValueError(f"mean_shape must be [H,W] or [K,H,W], got {mean_shape.shape}")

    vis_panels: List[np.ndarray] = []
    for shape in shapes:
        m = shape.copy()
        m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
        if m.max() > 0:
            m = m / m.max()
        img_u8 = (m * 255.0).clip(0, 255).astype(np.uint8)

        if _HAS_CV2:
            name = (cmap_name or "jet").lower()
            cmap_map = {
                "jet": cv2.COLORMAP_JET,  # type: ignore[attr-defined]
                "hot": cv2.COLORMAP_HOT,  # type: ignore[attr-defined]
                "viridis": cv2.COLORMAP_VIRIDIS,  # type: ignore[attr-defined]
                "plasma": cv2.COLORMAP_PLASMA,  # type: ignore[attr-defined]
                "magma": cv2.COLORMAP_MAGMA,  # type: ignore[attr-defined]
                "inferno": cv2.COLORMAP_INFERNO,  # type: ignore[attr-defined]
            }
            cmap = cmap_map.get(name, cv2.COLORMAP_JET)  # type: ignore[attr-defined]
            colored = cv2.applyColorMap(img_u8, cmap)  # type: ignore[attr-defined]
        else:
            os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
            import matplotlib.cm as cm

            try:
                import matplotlib

                colormap = matplotlib.colormaps.get_cmap(cmap_name or "jet")  # type: ignore[attr-defined]
            except Exception:
                colormap = cm.get_cmap(cmap_name or "jet")
            rgba = colormap(img_u8.astype(np.float32) / 255.0)
            colored = (rgba[..., :3] * 255.0).astype(np.uint8)
        vis_panels.append(colored)

    if len(vis_panels) == 1:
        panel = vis_panels[0]
    else:
        sep = np.full((vis_panels[0].shape[0], 4, 3), 255, dtype=np.uint8)
        row = []
        for i, panel in enumerate(vis_panels):
            row.append(panel)
            if i < len(vis_panels) - 1:
                row.append(sep)
        panel = np.concatenate(row, axis=1)

    if _HAS_CV2:
        cv2.imwrite(vis_path, panel)  # type: ignore[attr-defined]
    else:
        Image.fromarray(panel, mode="RGB").save(vis_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate canonical mean shape prior from COCO masks.")
    p.add_argument("--ann", required=True, help="COCO annotation json path (e.g., train_coco.json)")
    p.add_argument("--output", default="plug_canonical_prior.npy", help="Output .npy file path")
    p.add_argument("--target-size", nargs=2, type=int, default=[28, 28], metavar=("W", "H"))
    p.add_argument("--temp-size", nargs=2, type=int, default=[64, 64], metavar=("W", "H"))
    p.add_argument("--pad", type=int, default=10, help="Padding around bbox crop")
    p.add_argument("--min-wh", type=int, default=5, help="Skip instances with cropped w/h < min_wh")
    p.add_argument(
        "--cat",
        action="append",
        default=None,
        help="Category name filter (can be passed multiple times). If omitted, use all categories.",
    )
    p.add_argument(
        "--prefer-bottom-heavy",
        action="store_true",
        help="Resolve 180deg ambiguity by putting heavier half to bottom (default behavior).",
    )
    p.add_argument(
        "--prefer-top-heavy",
        action="store_true",
        help="Opposite of --prefer-bottom-heavy (put heavier half to top).",
    )
    p.add_argument("--max-instances", type=int, default=None, help="Optional cap on number of instances used")
    p.add_argument("--log-every", type=int, default=100, help="Print progress every N images (0 disables)")
    p.add_argument(
        "--bank-size",
        type=int,
        default=1,
        help="If >1, output a prior bank [K,H,W]. Template 0 is global mean; remaining templates are cluster means.",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for prior bank clustering")
    p.add_argument("--vis", default=None, help="Optional output png path for visualization")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ann_file = os.path.abspath(args.ann)
    if not os.path.exists(ann_file):
        raise FileNotFoundError(f"annotation file not found: {ann_file}")

    target_w, target_h = args.target_size
    temp_w, temp_h = args.temp_size
    prefer_bottom = True
    if args.prefer_top_heavy:
        prefer_bottom = False
    if args.prefer_bottom_heavy:
        prefer_bottom = True

    if int(args.bank_size) > 1:
        mean_shape, n = build_prior_bank(
            ann_file=ann_file,
            bank_size=int(args.bank_size),
            target_size=(target_w, target_h),
            temp_size=(temp_w, temp_h),
            pad=args.pad,
            min_wh=args.min_wh,
            category_names=args.cat,
            prefer_bottom_heavy=prefer_bottom,
            max_instances=args.max_instances,
            log_every=args.log_every,
            seed=int(args.seed),
        )
    else:
        mean_shape, n = build_mean_shape(
            ann_file=ann_file,
            target_size=(target_w, target_h),
            temp_size=(temp_w, temp_h),
            pad=args.pad,
            min_wh=args.min_wh,
            category_names=args.cat,
            prefer_bottom_heavy=prefer_bottom,
            max_instances=args.max_instances,
            log_every=args.log_every,
        )

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, mean_shape)
    kind = "prior bank" if mean_shape.ndim == 3 else "mean prior"
    print(f"完成：累计实例 N={n}，已保存 {kind} 到：{out_path}，shape={mean_shape.shape}, dtype={mean_shape.dtype}")

    if args.vis:
        vis_path = os.path.abspath(args.vis)
        save_vis_png(mean_shape, vis_path)
        print(f"已保存可视化到：{vis_path}")


if __name__ == "__main__":
    main()


