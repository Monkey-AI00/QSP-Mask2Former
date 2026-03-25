import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from config import DATA_DIR


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def _read_depth(path: Path) -> np.ndarray:
    return np.array(Image.open(path), dtype=np.float32)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _write_depth(path: Path, depth: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(depth.astype(np.uint16)).save(path)


def _find_rgb_path(car_dir: Path, view_id: str) -> Path:
    candidates = [
        car_dir / f"{view_id}_rgb.jpg",
        car_dir / f"{view_id}_rgb.jpeg",
        car_dir / f"{view_id}_rgb.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"missing rgb image for {view_id} in {car_dir}")


def _infer_available_views(car_dir: Path) -> list[str]:
    views: set[str] = set()
    for path in car_dir.iterdir():
        name = path.name
        if "_rgb" not in name:
            continue
        stem = path.stem
        if stem.endswith("_rgb"):
            views.add(stem[:-4])
    ordered = ["view_far", "view_mid", "view_close"]
    remaining = sorted(v for v in views if v not in ordered)
    return [v for v in ordered if v in views] + remaining


def _default_meta(car_id: str, views: list[str]) -> dict[str, Any]:
    return {
        "car_id": car_id,
        "car_name": f"Demo Vehicle {car_id}",
        "class_name": "ev_sedan",
        "charge_port_side": "front_right",
        "available_views": views,
    }


def _default_graph(car_id: str) -> dict[str, Any]:
    return {
        "car_id": car_id,
        "root_node": car_id,
        "observable_nodes": ["logo", "left_headlight", "right_headlight"],
        "inference_nodes": ["charge_port_center"],
        "relations": {
            "logo": {"to_charge_port_2d": [115, 28], "to_charge_port_3d": [0.21, -0.03, 0.02]},
            "left_headlight": {"to_charge_port_2d": [285, 14], "to_charge_port_3d": [0.39, -0.01, 0.03]},
            "right_headlight": {"to_charge_port_2d": [-55, 10], "to_charge_port_3d": [0.05, -0.01, 0.01]},
        },
    }


def _default_kpts(car_id: str, view_id: str, rgb: np.ndarray) -> dict[str, Any]:
    h, w = rgb.shape[:2]
    if "close" in view_id:
        keypoints = {
            "logo": [round(w * 0.47), round(h * 0.54)],
            "left_headlight": [round(w * 0.29), round(h * 0.53)],
            "right_headlight": [round(w * 0.73), round(h * 0.51)],
        }
        scores = {"logo": 0.91, "left_headlight": 0.72, "right_headlight": 0.88}
        cls_probs = {car_id: 0.91, "car_A": 0.06, "car_C": 0.03}
        quality = {"distance": "close", "angle": "front", "blur": 0.08}
    elif "mid" in view_id:
        keypoints = {
            "logo": [round(w * 0.49), round(h * 0.52)],
            "left_headlight": [round(w * 0.33), round(h * 0.54)],
            "right_headlight": [round(w * 0.67), round(h * 0.53)],
        }
        scores = {"logo": 0.79, "left_headlight": 0.68, "right_headlight": 0.7}
        cls_probs = {car_id: 0.8, "car_A": 0.12, "car_C": 0.08}
        quality = {"distance": "mid", "angle": "side", "blur": 0.16}
    else:
        keypoints = {
            "logo": [round(w * 0.5), round(h * 0.53)],
            "left_headlight": [round(w * 0.35), round(h * 0.55)],
            "right_headlight": [round(w * 0.65), round(h * 0.54)],
        }
        scores = {"logo": 0.61, "left_headlight": 0.44, "right_headlight": 0.43}
        cls_probs = {car_id: 0.6, "car_A": 0.24, "car_C": 0.16}
        quality = {"distance": "far", "angle": "oblique", "blur": 0.3}
    return {
        "view_id": view_id,
        "vehicle_id": car_id,
        "keypoints_2d": keypoints,
        "keypoint_scores": scores,
        "cls_probs": cls_probs,
        "view_quality": quality,
    }


def _default_depth(view_id: str, rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    base = np.tile(np.linspace(2400, 1500, w, dtype=np.float32), (h, 1))
    if "mid" in view_id:
        base -= 250
    elif "close" in view_id:
        base -= 500
    yy, xx = np.mgrid[0:h, 0:w]
    cx = int(w * (0.72 if "close" in view_id else 0.66))
    cy = int(h * 0.54)
    rx = max(40, int(w * 0.08))
    ry = max(30, int(h * 0.1))
    mask = ((xx - cx) ** 2) / float(rx**2) + ((yy - cy) ** 2) / float(ry**2) <= 1.0
    base[mask] -= 150
    return np.clip(base, 600, 4000).astype(np.uint16)


def ensure_car_assets(car_id: str) -> None:
    car_dir = DATA_DIR / car_id
    car_dir.mkdir(parents=True, exist_ok=True)
    views = _infer_available_views(car_dir)
    if not views:
        return

    meta_path = car_dir / "meta.json"
    if not meta_path.exists():
        _write_json(meta_path, _default_meta(car_id, views))

    graph_path = DATA_DIR / "graphs" / f"{car_id}_graph.json"
    if not graph_path.exists():
        _write_json(graph_path, _default_graph(car_id))

    for view_id in views:
        rgb_path = _find_rgb_path(car_dir, view_id)
        rgb = _read_rgb(rgb_path)
        kpts_path = car_dir / f"{view_id}_kpts.json"
        depth_path = car_dir / f"{view_id}_depth.png"
        if not kpts_path.exists():
            _write_json(kpts_path, _default_kpts(car_id, view_id, rgb))
        if not depth_path.exists():
            _write_depth(depth_path, _default_depth(view_id, rgb))


def load_meta(car_id: str) -> dict[str, Any]:
    ensure_car_assets(car_id)
    return _read_json(DATA_DIR / car_id / "meta.json")


def list_views(car_id: str) -> list[str]:
    meta = load_meta(car_id)
    return list(meta.get("available_views", []))


def load_graph(car_id: str) -> dict[str, Any]:
    ensure_car_assets(car_id)
    return _read_json(DATA_DIR / "graphs" / f"{car_id}_graph.json")


def load_view(car_id: str, view_id: str) -> dict[str, Any]:
    ensure_car_assets(car_id)
    car_dir = DATA_DIR / car_id
    rgb = _read_rgb(_find_rgb_path(car_dir, view_id))
    depth = _read_depth(car_dir / f"{view_id}_depth.png")
    kpts = _read_json(car_dir / f"{view_id}_kpts.json")
    meta = load_meta(car_id)
    return {
        "car_id": car_id,
        "view_id": view_id,
        "rgb": rgb,
        "depth": depth,
        "kpts": kpts,
        "meta": meta,
    }
