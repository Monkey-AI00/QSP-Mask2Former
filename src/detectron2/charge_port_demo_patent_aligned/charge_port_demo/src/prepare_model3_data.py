import json
from pathlib import Path

import numpy as np
from PIL import Image

from config import DATA_DIR


GRID_NAMES = {
    (0, 0): "grid_11_oblique_far.png",
    (0, 1): "grid_12_oblique_mid.png",
    (0, 2): "grid_13_headlight_close.png",
    (1, 0): "grid_21_front_full.png",
    (1, 1): "grid_22_logo_close.png",
    (1, 2): "grid_23_front_mid.png",
    (2, 0): "grid_31_rear_port_open.png",
    (2, 1): "grid_32_port_closeup.png",
    (2, 2): "grid_33_rear_side.png",
}

VIEW_TO_GRID = {
    "view_far": "grid_21_front_full.png",
    "view_mid": "grid_23_front_mid.png",
    "view_close": "grid_31_rear_port_open.png",
}


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _pseudo_depth(view_id: str, rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    base = np.tile(np.linspace(2600, 1500, w, dtype=np.float32), (h, 1))
    if view_id == "view_mid":
        base -= 300
    elif view_id == "view_close":
        base -= 900
    yy, xx = np.mgrid[0:h, 0:w]
    if view_id == "view_close":
        cx, cy, rx, ry = 624, 291, 90, 70
    elif view_id == "view_mid":
        cx, cy, rx, ry = int(w * 0.5), int(h * 0.58), 120, 80
    else:
        cx, cy, rx, ry = int(w * 0.72), int(h * 0.55), 120, 90
    mask = ((xx - cx) ** 2) / float(rx**2) + ((yy - cy) ** 2) / float(ry**2) <= 1.0
    base[mask] -= 180
    return np.clip(base, 600, 4000).astype(np.uint16)


def _model3_meta() -> dict:
    return {
        "car_id": "model3",
        "car_name": "Tesla Model 3",
        "class_name": "tesla_model3",
        "charge_port_side": "rear_left",
        "available_views": ["view_far", "view_mid", "view_close"],
        "port_reference_image": "grid_32_port_closeup.png",
    }


def _model3_graph() -> dict:
    return {
        "car_id": "model3",
        "root_node": "model3",
        "observable_nodes": ["logo", "left_headlight", "right_headlight", "left_tail_light", "rear_door_handle"],
        "inference_nodes": ["charge_port_center"],
        "relations": {
            "logo": {"to_charge_port_2d": [180, 180], "to_charge_port_3d": [0.4, 0.05, 0.05]},
            "left_headlight": {"to_charge_port_2d": [360, 210], "to_charge_port_3d": [0.6, 0.06, 0.08]},
            "right_headlight": {"to_charge_port_2d": [20, 220], "to_charge_port_3d": [0.2, 0.06, 0.08]},
            "left_tail_light": {"to_charge_port_2d": [84, -12], "to_charge_port_3d": [0.12, 0.0, 0.03]},
            "rear_door_handle": {"to_charge_port_2d": [-138, 4], "to_charge_port_3d": [-0.16, 0.0, 0.02]},
        },
    }


def _model3_kpts(view_id: str) -> dict:
    if view_id == "view_far":
        return {
            "view_id": view_id,
            "vehicle_id": "model3",
            "keypoints_2d": {
                "logo": [469, 228],
                "left_headlight": [168, 251],
                "right_headlight": [770, 249],
            },
            "keypoint_scores": {
                "logo": 0.50,
                "left_headlight": 0.39,
                "right_headlight": 0.39,
            },
            "cls_probs": {
                "model3": 0.56,
                "car_A": 0.24,
                "car_B": 0.20,
            },
            "view_quality": {
                "distance": "far",
                "angle": "front",
                "blur": 0.18,
            },
        }
    if view_id == "view_mid":
        return {
            "view_id": view_id,
            "vehicle_id": "model3",
            "keypoints_2d": {
                "logo": [470, 186],
                "left_headlight": [109, 192],
                "right_headlight": [826, 191],
            },
            "keypoint_scores": {
                "logo": 0.61,
                "left_headlight": 0.55,
                "right_headlight": 0.55,
            },
            "cls_probs": {
                "model3": 0.64,
                "car_A": 0.2,
                "car_B": 0.16,
            },
            "view_quality": {
                "distance": "mid",
                "angle": "front",
                "blur": 0.12,
            },
        }
    return {
        "view_id": view_id,
        "vehicle_id": "model3",
        "keypoints_2d": {
            "left_tail_light": [732, 286],
            "rear_door_handle": [119, 274],
        },
        "keypoint_scores": {
            "left_tail_light": 0.92,
            "rear_door_handle": 0.86,
        },
        "cls_probs": {
            "model3": 0.96,
            "car_A": 0.02,
            "car_B": 0.02,
        },
        "view_quality": {
            "distance": "close",
            "angle": "oblique",
            "blur": 0.04,
        },
        "true_port_2d": [624, 291],
        "true_port_score": 0.99,
        "port_reference_image": "grid_32_port_closeup.png",
    }


def split_grid_image(model3_dir: Path) -> None:
    collage_path = model3_dir / "model3.png"
    img = Image.open(collage_path).convert("RGB")
    width, height = img.size
    tile_w, tile_h = width // 3, height // 3
    for row in range(3):
        for col in range(3):
            out_name = GRID_NAMES[(row, col)]
            out_path = model3_dir / out_name
            if out_path.exists():
                continue
            tile = img.crop((col * tile_w, row * tile_h, (col + 1) * tile_w, (row + 1) * tile_h))
            tile.save(out_path)


def prepare_model3_data() -> None:
    model3_dir = DATA_DIR / "model3"
    graphs_dir = DATA_DIR / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    split_grid_image(model3_dir)

    for view_id, tile_name in VIEW_TO_GRID.items():
        rgb = np.array(Image.open(model3_dir / tile_name).convert("RGB"), dtype=np.uint8)
        Image.fromarray(rgb).save(model3_dir / f"{view_id}_rgb.png")
        Image.fromarray(_pseudo_depth(view_id, rgb)).save(model3_dir / f"{view_id}_depth.png")
        _save_json(model3_dir / f"{view_id}_kpts.json", _model3_kpts(view_id))

    _save_json(model3_dir / "meta.json", _model3_meta())
    _save_json(graphs_dir / "model3_graph.json", _model3_graph())


if __name__ == "__main__":
    prepare_model3_data()
    print("Prepared model3 demo data.")
