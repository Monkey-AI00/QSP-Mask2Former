from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import FRAME_DIR, VIDEO_FPS, VIDEO_HOLD_SECONDS


def _to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(image.astype(np.uint8))


def _to_np(image: Image.Image) -> np.ndarray:
    return np.array(image, dtype=np.uint8)


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def draw_keypoints(image: np.ndarray, keypoints: dict[str, list[float]], scores: dict[str, float]) -> np.ndarray:
    pil = _to_pil(image)
    draw = ImageDraw.Draw(pil)
    font = _load_font(max(15, pil.width // 62))
    radius = max(8, pil.width // 110)
    for name, xy in keypoints.items():
        x, y = float(xy[0]), float(xy[1])
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(255, 255, 0), width=4)
        label = f"{name}:{scores.get(name, 0.0):.2f}"
        bbox = draw.textbbox((0, 0), label, font=font)
        label_w = bbox[2] - bbox[0]
        label_h = bbox[3] - bbox[1]
        prefer_right = x < pil.width * 0.62
        lx = x + radius + 8 if prefer_right else x - radius - 8 - label_w
        ly = y - radius - 8
        if ly < 4:
            ly = y + radius + 8
        lx = max(4, min(lx, pil.width - label_w - 12))
        ly = max(4, min(ly, pil.height - label_h - 10))
        draw.rounded_rectangle(
            (lx - 6, ly - 4, lx + label_w + 6, ly + label_h + 4),
            radius=8,
            fill=(20, 20, 20),
        )
        draw.text((lx, ly), label, fill=(255, 255, 0), font=font)
    return _to_np(pil)


def draw_confidence_panel(image: np.ndarray, conf_result: dict[str, float], perception_result: dict[str, Any]) -> np.ndarray:
    pil = _to_pil(image)
    draw = ImageDraw.Draw(pil)
    margin = max(20, pil.width // 80)
    panel_w = max(320, pil.width // 3)
    panel_h = max(150, pil.height // 5)
    x0, y0, x1, y1 = margin, pil.height - panel_h - margin, margin + panel_w, pil.height - margin
    title_font = _load_font(max(18, pil.width // 48))
    body_font = _load_font(max(15, pil.width // 62))
    draw.rounded_rectangle((x0, y0, x1, y1), radius=16, fill=(15, 20, 35))
    line_y = y0 + 14
    draw.text((x0 + 18, line_y), f"vehicle: {perception_result['vehicle_id']}", fill=(255, 255, 255), font=title_font)
    cls_probs = ", ".join(f"{k}:{v:.2f}" for k, v in perception_result["cls_probs"].items())
    line_y += max(26, pil.width // 55)
    draw.text((x0 + 18, line_y), f"cls: {cls_probs}", fill=(190, 220, 255), font=body_font)
    line_y += max(24, pil.width // 58)
    draw.text((x0 + 18, line_y), f"cls_conf: {conf_result['cls_conf']:.2f}", fill=(255, 255, 255), font=body_font)
    line_y += max(24, pil.width // 58)
    draw.text((x0 + 18, line_y), f"kpt_conf: {conf_result['kpt_conf']:.2f}", fill=(255, 255, 255), font=body_font)
    line_y += max(24, pil.width // 58)
    draw.text((x0 + 18, line_y), f"final_conf: {conf_result['final_conf']:.2f}", fill=(255, 200, 90), font=body_font)
    return _to_np(pil)


def draw_port_result(image: np.ndarray, port_result: dict[str, Any]) -> np.ndarray:
    pil = _to_pil(image)
    draw = ImageDraw.Draw(pil)
    font = _load_font(max(20, pil.width // 50))
    port_2d = port_result.get("port_2d")
    if port_2d is not None:
        x, y = float(port_2d[0]), float(port_2d[1])
        radius = max(12, pil.width // 90)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=(0, 255, 0), width=5)
        draw.rounded_rectangle((x + radius + 6, y - radius - 4, x + radius + 220, y + radius + 8), radius=10, fill=(0, 35, 0))
        draw.text((x + radius + 16, y - radius), "charge_port", fill=(0, 255, 0), font=font)
    return _to_np(pil)


def draw_action_panel(image: np.ndarray, action: dict[str, Any], title: str = "action") -> np.ndarray:
    pil = _to_pil(image)
    draw = ImageDraw.Draw(pil)
    margin = max(20, pil.width // 80)
    panel_w = max(320, pil.width // 3)
    panel_h = max(150, pil.height // 5)
    x0, y0, x1, y1 = pil.width - panel_w - margin, pil.height - panel_h - margin, pil.width - margin, pil.height - margin
    title_font = _load_font(max(18, pil.width // 48))
    body_font = _load_font(max(15, pil.width // 62))
    draw.rounded_rectangle((x0, y0, x1, y1), radius=16, fill=(30, 40, 25))
    draw.text((x0 + 18, y0 + 14), title, fill=(255, 255, 255), font=title_font)
    line_y = y0 + max(38, pil.width // 42)
    for key, value in action.items():
        draw.text((x0 + 18, line_y), f"{key}: {value}", fill=(205, 245, 205), font=body_font)
        line_y += max(22, pil.width // 60)
    return _to_np(pil)


def save_frame(image: np.ndarray, frame_id: str, frame_dir: Path | None = None) -> Path:
    destination = frame_dir or FRAME_DIR
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{frame_id}.png"
    _to_pil(image).save(path)
    return path


def build_video_from_frames(frame_dir: Path, output_path: Path) -> Path | None:
    frame_paths = sorted(frame_dir.glob("*.png"))
    if not frame_paths:
        return None
    try:
        import cv2  # type: ignore
    except Exception:
        images = [_to_pil(np.array(Image.open(path).convert("RGB"))) for path in frame_paths]
        gif_path = output_path.with_suffix(".gif")
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        duration_ms = max(100, int(round(1000.0 * float(VIDEO_HOLD_SECONDS))))
        images[0].save(gif_path, save_all=True, append_images=images[1:], duration=duration_ms, loop=0)
        return gif_path

    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        return None
    height, width = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps = float(VIDEO_FPS)
    repeat_count = max(1, int(round(fps * float(VIDEO_HOLD_SECONDS))))
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for path in frame_paths:
        frame = cv2.imread(str(path))
        if frame is not None:
            for _ in range(repeat_count):
                writer.write(frame)
    writer.release()
    return output_path
