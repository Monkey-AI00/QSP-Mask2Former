from typing import Any

import numpy as np

from validation import validate_intrinsics


def _depth_meters(depth: np.ndarray) -> np.ndarray:
    values = np.asarray(depth, dtype=float)
    positive = values[values > 0.0]
    if positive.size == 0:
        raise ValueError("depth image contains no valid values")
    scale = 0.001 if float(np.median(positive)) > 10.0 else 1.0
    return values * scale


def depth_to_point_cloud(depth: np.ndarray, intrinsics: dict[str, Any], stride: int = 8) -> np.ndarray:
    if stride <= 0:
        raise ValueError("point-cloud stride must be positive")
    camera = validate_intrinsics(intrinsics)
    depth_m = _depth_meters(depth)
    height, width = depth_m.shape[:2]
    vv, uu = np.mgrid[0:height:stride, 0:width:stride]
    zz = depth_m[0:height:stride, 0:width:stride]
    valid = np.isfinite(zz) & (zz > 0.0)
    xx = (uu - camera["cx"]) * zz / camera["fx"]
    yy = (vv - camera["cy"]) * zz / camera["fy"]
    points = np.stack([xx[valid], yy[valid], zz[valid]], axis=1)
    if points.size == 0:
        raise ValueError("depth image produced an empty point cloud")
    return points.astype(float)


def _pool_mean(array: np.ndarray, factor: int) -> np.ndarray:
    height, width = array.shape[:2]
    out_h, out_w = max(1, height // factor), max(1, width // factor)
    cropped = array[: out_h * factor, : out_w * factor]
    if cropped.size == 0:
        return np.mean(array, axis=(0, 1), keepdims=True)
    tail = cropped.shape[2:]
    reshaped = cropped.reshape(out_h, factor, out_w, factor, *tail)
    return reshaped.mean(axis=(1, 3))


def _pad_vector(values: np.ndarray, length: int = 8) -> np.ndarray:
    vector = np.ravel(np.asarray(values, dtype=float))
    if vector.size >= length:
        return vector[:length]
    return np.pad(vector, (0, length - vector.size))


def _image_conv_pool_stack(
    rgb_float: np.ndarray, depth_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, list[int]]]:
    """Execute a fixed-weight conv/pool/conv/pool prototype without a training dependency."""
    depth_scale = max(float(np.percentile(depth_m, 95)), 1e-6)
    multimodal = np.concatenate([rgb_float, (depth_m / depth_scale)[..., None]], axis=2)
    padded = np.pad(multimodal, ((1, 1), (1, 1), (0, 0)), mode="edge")
    conv1 = (
        padded[0:-2:2, 1:-1:2]
        + padded[2::2, 1:-1:2]
        + padded[1:-1:2, 0:-2:2]
        + padded[1:-1:2, 2::2]
        + 4.0 * padded[1:-1:2, 1:-1:2]
    ) / 8.0
    pool1 = _pool_mean(conv1, 2)
    padded_second = np.pad(pool1, ((1, 1), (1, 1), (0, 0)), mode="edge")
    conv2 = np.maximum(
        0.0,
        padded_second[:-2, 1:-1]
        + padded_second[2:, 1:-1]
        + padded_second[1:-1, :-2]
        + padded_second[1:-1, 2:]
        - 4.0 * padded_second[1:-1, 1:-1],
    )
    pool2 = _pool_mean(conv2, 2)
    shapes = {
        "conv1": list(conv1.shape),
        "pool1": list(pool1.shape),
        "conv2": list(conv2.shape),
        "pool2": list(pool2.shape),
    }
    return pool1, pool2, shapes


def _point_mlp_pool_stack(points: np.ndarray) -> tuple[np.ndarray, dict[str, list[int]]]:
    weights1 = np.array(
        [[0.7, -0.2, 0.4, 0.1, -0.5, 0.3, 0.6, -0.1],
         [0.1, 0.8, -0.3, 0.5, 0.2, -0.4, 0.1, 0.7],
         [-0.2, 0.1, 0.9, 0.3, 0.6, 0.2, -0.5, 0.4]],
        dtype=float,
    )
    mlp1 = np.tanh(points @ weights1)
    pooled = np.max(mlp1, axis=0)
    weights2 = np.eye(8, dtype=float) * 0.75 + np.ones((8, 8), dtype=float) * 0.03125
    mlp2 = np.tanh(pooled @ weights2)
    return mlp2, {"mlp1": list(mlp1.shape), "max_pool": list(pooled.shape), "mlp2": list(mlp2.shape)}


def extract_multimodal_features(rgb: np.ndarray, depth: np.ndarray, point_cloud: np.ndarray) -> dict[str, np.ndarray]:
    rgb_float = np.asarray(rgb, dtype=float) / 255.0
    depth_m = _depth_meters(depth)
    points = np.asarray(point_cloud, dtype=float)
    if rgb_float.ndim != 3 or rgb_float.shape[2] != 3:
        raise ValueError("rgb image must have shape HxWx3")
    if depth_m.shape[:2] != rgb_float.shape[:2]:
        raise ValueError("rgb and depth images must be aligned")
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("point cloud must have shape Nx3 with finite values")

    image_scale_1, image_scale_2, image_shapes = _image_conv_pool_stack(rgb_float, depth_m)
    point_features, point_shapes = _point_mlp_pool_stack(points)
    image_scale_1_token = _pad_vector(np.r_[image_scale_1.mean((0, 1)), image_scale_1.std((0, 1))])
    image_scale_2_token = _pad_vector(np.r_[image_scale_2.mean((0, 1)), image_scale_2.std((0, 1))])
    point_token = _pad_vector(point_features)

    tokens = np.stack([image_scale_1_token, image_scale_2_token, point_token], axis=0)
    scores = tokens @ tokens.T / np.sqrt(tokens.shape[1])
    scores -= scores.max(axis=1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=1, keepdims=True)
    attended = weights @ tokens
    projection = np.eye(tokens.shape[1], dtype=float) * 0.8 + np.ones_like(np.eye(tokens.shape[1])) * 0.025
    fused_feature = attended.mean(axis=0) @ projection
    return {
        "rgb_token": image_scale_1_token,
        "depth_token": image_scale_2_token,
        "image_scale_1_token": image_scale_1_token,
        "image_scale_2_token": image_scale_2_token,
        "point_token": point_token,
        "attention_weights": weights,
        "fused_feature": fused_feature,
        "layer_shapes": {"image_branch": image_shapes, "point_branch": point_shapes},
        "architecture_trace": {
            "image_branch": ["conv1", "pool1", "conv2", "pool2"],
            "point_branch": ["mlp1", "max_pool", "mlp2"],
            "fusion": ["concatenate", "self_attention", "conv1x1"],
            "global_head": ["global_average_pool", "fully_connected", "softmax"],
            "local_head": ["upsample", "convolution", "sigmoid"],
        },
    }


def global_semantic_head(fused_feature: np.ndarray, replay_probabilities: dict[str, float]) -> dict[str, float]:
    classes = list(replay_probabilities)
    if not classes:
        return {}
    feature = _pad_vector(np.asarray(fused_feature, dtype=float))
    norm = max(float(np.linalg.norm(feature)), 1e-9)
    normalized = feature / norm
    rows = []
    for index, _ in enumerate(classes):
        rows.append(np.roll(np.linspace(-0.5, 0.5, normalized.size), index))
    weights = np.asarray(rows, dtype=float)
    prior = np.asarray([max(float(replay_probabilities[name]), 1e-9) for name in classes])
    logits = np.log(prior) + 0.02 * (weights @ normalized)
    logits -= logits.max()
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()
    return {name: float(value) for name, value in zip(classes, probabilities)}


def local_discriminative_head(
    image_shape: tuple[int, int],
    keypoints: dict[str, list[float]],
    peak_scores: dict[str, float],
) -> dict[str, np.ndarray]:
    """Replay a local head as coordinate upsampling, Gaussian convolution and sigmoid response."""
    return build_gaussian_heatmaps(image_shape, keypoints, peak_scores)


def build_gaussian_heatmaps(
    image_shape: tuple[int, int],
    keypoints: dict[str, list[float]],
    peak_scores: dict[str, float],
    heatmap_shape: tuple[int, int] = (64, 64),
    sigma: float = 1.8,
) -> dict[str, np.ndarray]:
    image_h, image_w = image_shape
    heat_h, heat_w = heatmap_shape
    yy, xx = np.mgrid[0:heat_h, 0:heat_w]
    heatmaps: dict[str, np.ndarray] = {}
    for name, xy in keypoints.items():
        center_x = float(xy[0]) * heat_w / max(image_w, 1)
        center_y = float(xy[1]) * heat_h / max(image_h, 1)
        peak = float(np.clip(peak_scores.get(name, 0.0), 0.0, 1.0))
        heatmap = peak * np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * sigma**2))
        heatmaps[name] = heatmap.astype(np.float32)
    return heatmaps
