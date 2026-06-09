from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont


SRC = Path(r"F:\港口插线\CV\xor_zoom_grid.png")
OUT_DIR = Path(r"E:\codex-workspace\QSP论文")
OUT_PNG = OUT_DIR / "xor_zoom_grid_5col_sci.png"
OUT_TIFF = OUT_DIR / "xor_zoom_grid_5col_sci.tiff"
OUT_PDF = OUT_DIR / "xor_zoom_grid_5col_sci.pdf"

SOURCE_COLS = [0, 1, 2, 5, 8]
COL_LABELS = ["Raw", "GT", "Mask R-CNN", "Mask2Former", "QSP-Mask2Former"]
XOR_PANEL_INDICES = [2, 3, 4]

N_SOURCE_COLS = 9
N_ROWS = 5
SOURCE_LABEL_STRIP = 20

# row, ROI name, crop inside the original 520 x 416 tile: x0, y0, x1, y1
ROI_SPECS = [
    (1, "Saturated boundary", (115, 125, 315, 300)),
    (3, "Reflective cap", (260, 145, 455, 325)),
    (4, "Occluded edge", (175, 15, 390, 210)),
]


def get_font(size, bold=False):
    candidates = []
    if bold:
        candidates.extend(
            [
                r"C:\Windows\Fonts\arialbd.ttf",
                r"C:\Windows\Fonts\calibrib.ttf",
            ]
        )
    candidates.extend(
        [
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\calibri.ttf",
            r"C:\Windows\Fonts\segoeui.ttf",
        ]
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def text_size(draw, text, font):
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def enhance_panel(panel, is_xor):
    if not is_xor:
        return panel
    panel = ImageEnhance.Brightness(panel).enhance(1.24)
    panel = ImageEnhance.Contrast(panel).enhance(1.06)
    panel = ImageEnhance.Color(panel).enhance(1.18)
    panel = ImageEnhance.Sharpness(panel).enhance(1.25)
    return panel


def crop_source_panel(src, source_col, row, source_tile_w, source_tile_h):
    sx0 = source_col * source_tile_w
    sy0 = row * source_tile_h + SOURCE_LABEL_STRIP
    panel = src.crop((sx0, sy0, sx0 + source_tile_w, (row + 1) * source_tile_h)).convert("RGB")
    return panel.resize((source_tile_w, source_tile_h), Image.Resampling.LANCZOS)


def scale_roi(crop, tile_h):
    scale = tile_h / float(tile_h - SOURCE_LABEL_STRIP)
    x0, y0, x1, y1 = crop
    return (
        x0,
        max(0, int(round((y0 - SOURCE_LABEL_STRIP) * scale))),
        x1,
        max(0, int(round((y1 - SOURCE_LABEL_STRIP) * scale))),
    )


def assemble_main_grid(src):
    source_tile_w = src.width // N_SOURCE_COLS
    source_tile_h = src.height // N_ROWS
    tile_w = source_tile_w
    tile_h = source_tile_h

    header_h = 72
    legend_h = 78
    grid_w = tile_w * len(SOURCE_COLS)
    grid_h = tile_h * N_ROWS
    canvas = Image.new("RGB", (grid_w, header_h + grid_h + legend_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    header_font = get_font(28, bold=True)
    legend_font = get_font(21)
    row_font = get_font(20, bold=True)

    for out_col, source_col in enumerate(SOURCE_COLS):
        x = out_col * tile_w
        label = COL_LABELS[out_col]
        tw, th = text_size(draw, label, header_font)
        draw.text((x + (tile_w - tw) / 2, 22), label, fill=(20, 20, 20), font=header_font)
        draw.line((x, header_h - 5, x + tile_w, header_h - 5), fill=(210, 210, 210), width=2)

    for row in range(N_ROWS):
        for out_col, source_col in enumerate(SOURCE_COLS):
            panel = crop_source_panel(src, source_col, row, source_tile_w, source_tile_h)
            panel = enhance_panel(panel, out_col in XOR_PANEL_INDICES)
            canvas.paste(panel, (out_col * tile_w, header_h + row * tile_h))

        y = header_h + row * tile_h + 12
        draw.rounded_rectangle((8, y, 62, y + 30), radius=4, fill=(255, 255, 255), outline=(185, 185, 185), width=1)
        draw.text((18, y + 4), "S%d" % (row + 1), fill=(30, 30, 30), font=row_font)

    draw.rectangle((0, header_h, grid_w - 1, header_h + grid_h - 1), outline=(35, 35, 35), width=2)

    legend_y = header_h + grid_h + 18
    draw.rectangle((44, legend_y + 5, 84, legend_y + 25), fill=(220, 40, 40))
    draw.text((96, legend_y), "FP (red): predicted foreground outside GT", fill=(25, 25, 25), font=legend_font)
    draw.rectangle((760, legend_y + 5, 800, legend_y + 25), fill=(45, 80, 220))
    draw.text((812, legend_y), "FN (blue): missed GT region", fill=(25, 25, 25), font=legend_font)

    return canvas, source_tile_w, source_tile_h, header_h


def draw_roi_boxes(main, tile_w, tile_h, header_h):
    draw = ImageDraw.Draw(main)
    roi_color = (245, 210, 95)
    shadow = (70, 70, 70)
    for row, _, crop in ROI_SPECS:
        crop = scale_roi(crop, tile_h)
        for out_col in XOR_PANEL_INDICES:
            x0 = out_col * tile_w + crop[0]
            y0 = header_h + row * tile_h + crop[1]
            x1 = out_col * tile_w + crop[2]
            y1 = header_h + row * tile_h + crop[3]
            draw.rectangle((x0 + 1, y0 + 1, x1 + 1, y1 + 1), outline=shadow, width=2)
            draw.rectangle((x0, y0, x1, y1), outline=roi_color, width=3)


def add_zoom_strip(main, tile_w, tile_h, header_h):
    w, h = main.size
    zoom_h = 500
    canvas = Image.new("RGB", (w, h + zoom_h), (255, 255, 255))
    canvas.paste(main, (0, 0))
    draw = ImageDraw.Draw(canvas)

    title_font = get_font(27, bold=True)
    label_font = get_font(20, bold=True)
    method_font = get_font(18)

    top = h + 18
    draw.line((0, h, w, h), fill=(45, 45, 45), width=3)
    draw.text((42, top), "Local magnified XOR regions", fill=(20, 20, 20), font=title_font)

    margin_x = 24
    group_gap = 16
    patch_gap = 8
    patch_w = 276
    patch_h = 245
    group_w = patch_w * 3 + patch_gap * 2
    y0 = h + 92

    for group_idx, (row, roi_name, crop) in enumerate(ROI_SPECS):
        crop = scale_roi(crop, tile_h)
        gx = margin_x + group_idx * (group_w + group_gap)
        draw.text((gx, y0 - 32), roi_name, fill=(20, 20, 20), font=label_font)
        for k, out_col in enumerate(XOR_PANEL_INDICES):
            x = gx + k * (patch_w + patch_gap)
            sx0 = out_col * tile_w + crop[0]
            sy0 = header_h + row * tile_h + crop[1]
            sx1 = out_col * tile_w + crop[2]
            sy1 = header_h + row * tile_h + crop[3]
            patch = main.crop((sx0, sy0, sx1, sy1)).resize((patch_w, patch_h), Image.Resampling.LANCZOS)
            canvas.paste(patch, (x, y0))
            draw.rectangle((x, y0, x + patch_w, y0 + patch_h), outline=(245, 210, 95), width=3)
            label = COL_LABELS[out_col]
            tw, _ = text_size(draw, label, method_font)
            draw.text((x + (patch_w - tw) / 2, y0 + patch_h + 12), label, fill=(25, 25, 25), font=method_font)

    return canvas


def main():
    src = Image.open(SRC).convert("RGB")
    if src.width % N_SOURCE_COLS != 0 or src.height % N_ROWS != 0:
        raise ValueError("Unexpected source grid size: %dx%d" % src.size)

    figure, tile_w, tile_h, header_h = assemble_main_grid(src)
    draw_roi_boxes(figure, tile_w, tile_h, header_h)
    figure = add_zoom_strip(figure, tile_w, tile_h, header_h)

    figure.save(OUT_PNG)
    figure.save(OUT_TIFF, dpi=(600, 600), compression="tiff_lzw")
    figure.save(OUT_PDF, resolution=600.0)
    print("saved:", OUT_PNG)
    print("saved:", OUT_TIFF)
    print("saved:", OUT_PDF)


if __name__ == "__main__":
    main()
