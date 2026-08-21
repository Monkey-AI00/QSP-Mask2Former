import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont


SRC = Path(r"/home/users1/sjw/cursor/workspace/outputs/gangkou/output/xor_zoom_table_0605/xor_zoom_grid.png")
OUT_DIR = Path(r"/home/users1/sjw/cursor/workspace/outputs/gangkou/output/xor_zoom_table_0605")

SOURCE_COLS = [0, 1, 2, 5, 8]
COL_LABELS = ["Raw", "GT", "Mask R-CNN", "Mask2Former", "QSP-Mask2Former"]
XOR_PANEL_INDICES = [2, 3, 4]
XOR_SOURCE_COLS = {SOURCE_COLS[i] for i in XOR_PANEL_INDICES}

N_SOURCE_COLS = 9
N_ROWS = 5
# visualize_xor_zoom_table.py uses a 34-pixel title bar. Crop the complete
# strip so standalone panels do not retain the lower edge of the source label.
SOURCE_LABEL_STRIP = 34
# The XOR source panels also contain "FP:red FN:blue" at the lower-left corner.
SOURCE_BOTTOM_LABEL_STRIP = 20

# row, ROI name, crop inside the original 520 x 416 tile: x0, y0, x1, y1
ROI_SPECS = [
    (1, "Saturated boundary", (115, 125, 315, 300)),
    (3, "Reflective cap", (260, 145, 455, 325)),
    (4, "Occluded edge", (175, 15, 390, 210)),
]


def get_font(size, bold=False):
    candidates = []
    # Linux 常见字体路径（优先，避免回退到 PIL 默认小字体）
    if bold:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            ]
        )
    candidates.extend(
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ]
    )
    # Windows 字体作为兼容后备
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
    sy1 = (row + 1) * source_tile_h
    if source_col in XOR_SOURCE_COLS:
        sy1 -= SOURCE_BOTTOM_LABEL_STRIP
    panel = src.crop((sx0, sy0, sx0 + source_tile_w, sy1)).convert("RGB")
    return panel.resize((source_tile_w, source_tile_h), Image.Resampling.LANCZOS)


def scale_roi(crop, tile_h):
    content_h = tile_h - SOURCE_LABEL_STRIP - SOURCE_BOTTOM_LABEL_STRIP
    scale = tile_h / float(content_h)
    x0, y0, x1, y1 = crop
    return (
        x0,
        max(0, int(round((y0 - SOURCE_LABEL_STRIP) * scale))),
        x1,
        max(0, int(round((y1 - SOURCE_LABEL_STRIP) * scale))),
    )


def assemble_main_grid(
    src,
    *,
    show_header=True,
    show_row_tags=True,
    show_legend=True,
):
    source_tile_w = src.width // N_SOURCE_COLS
    source_tile_h = src.height // N_ROWS
    tile_w = source_tile_w
    tile_h = source_tile_h

    header_h = 74 if show_header else 0
    legend_h = 96 if show_legend else 0
    grid_w = tile_w * len(SOURCE_COLS)
    grid_h = tile_h * N_ROWS
    canvas = Image.new("RGB", (grid_w, header_h + grid_h + legend_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    header_font = get_font(30, bold=True)
    legend_font = get_font(25)
    row_font = get_font(21, bold=True)

    if show_header:
        for out_col, source_col in enumerate(SOURCE_COLS):
            x = out_col * tile_w
            label = COL_LABELS[out_col]
            tw, th = text_size(draw, label, header_font)
            draw.text((x + (tile_w - tw) / 2, 21), label, fill=(20, 20, 20), font=header_font)
            draw.line((x, header_h - 5, x + tile_w, header_h - 5), fill=(210, 210, 210), width=2)

    for row in range(N_ROWS):
        for out_col, source_col in enumerate(SOURCE_COLS):
            panel = crop_source_panel(src, source_col, row, source_tile_w, source_tile_h)
            panel = enhance_panel(panel, out_col in XOR_PANEL_INDICES)
            canvas.paste(panel, (out_col * tile_w, header_h + row * tile_h))

        if show_row_tags:
            y = header_h + row * tile_h + 12
            draw.rounded_rectangle(
                (8, y, 68, y + 32),
                radius=5,
                fill=(255, 255, 255),
                outline=(185, 185, 185),
                width=1,
            )
            draw.text((18, y + 4), "S%d" % (row + 1), fill=(30, 30, 30), font=row_font)

    draw.rectangle((0, header_h, grid_w - 1, header_h + grid_h - 1), outline=(35, 35, 35), width=2)

    if show_legend:
        legend_y = header_h + grid_h + 26
        draw.rectangle((44, legend_y + 4, 92, legend_y + 30), fill=(220, 40, 40))
        draw.text(
            (106, legend_y),
            "FP (red): predicted foreground outside GT",
            fill=(25, 25, 25),
            font=legend_font,
        )
        draw.rectangle((920, legend_y + 4, 968, legend_y + 30), fill=(45, 80, 220))
        draw.text((982, legend_y), "FN (blue): missed GT region", fill=(25, 25, 25), font=legend_font)

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


def add_zoom_strip(main, tile_w, tile_h, header_h, *, show_local_labels=True):
    w, h = main.size
    zoom_h = 560 if show_local_labels else 330
    canvas = Image.new("RGB", (w, h + zoom_h), (255, 255, 255))
    canvas.paste(main, (0, 0))
    draw = ImageDraw.Draw(canvas)

    title_font = get_font(32, bold=True)
    label_font = get_font(23, bold=True)
    method_font = get_font(20)

    margin_x = 24
    group_gap = 16
    patch_gap = 8
    patch_w = 276
    patch_h = 290
    group_w = patch_w * 3 + patch_gap * 2
    y0 = h + (108 if show_local_labels else 20)

    for group_idx, (row, roi_name, crop) in enumerate(ROI_SPECS):
        crop = scale_roi(crop, tile_h)
        gx = margin_x + group_idx * (group_w + group_gap)
        if show_local_labels:
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
            if show_local_labels:
                label = COL_LABELS[out_col]
                tw, _ = text_size(draw, label, method_font)
                draw.text((x + (patch_w - tw) / 2, y0 + patch_h + 14), label, fill=(25, 25, 25), font=method_font)

    if show_local_labels:
        draw.text(
            (42, h + 20),
            "Local magnified XOR regions",
            fill=(20, 20, 20),
            font=title_font,
        )
        draw.text(
            (42, h + 58),
            "FP: red (over-seg), FN: blue (missed GT); region auto-selected by union XOR.",
            fill=(45, 45, 45),
            font=get_font(18),
        )

    return canvas


def parse_args():
    parser = argparse.ArgumentParser(
        description="Enhance XOR source grid and export complete/standalone paper figures."
    )
    parser.add_argument("--src", default=str(SRC), help="基础 XOR 网格图片路径")
    parser.add_argument("--out-dir", default=str(OUT_DIR), help="输出目录")
    parser.add_argument("--out-prefix", default="xor_zoom_grid_5col_sci", help="输出文件名前缀")
    parser.add_argument(
        "--out-png",
        default="",
        help="完整图 PNG 输出路径；相对路径默认相对于 --out-dir。",
    )
    parser.add_argument(
        "--out-tiff",
        default="",
        help="完整图 TIFF 输出路径；相对路径默认相对于 --out-dir。",
    )
    parser.add_argument(
        "--out-pdf",
        default="",
        help="完整图 PDF 输出路径；相对路径默认相对于 --out-dir。",
    )
    parser.add_argument(
        "--no-header",
        "--hide-header",
        dest="no_header",
        action="store_true",
        help="去掉完整图和顶部五栏图的列图注；默认保留。",
    )
    parser.add_argument(
        "--no-row-tags",
        "--hide-row-tags",
        dest="no_row_tags",
        action="store_true",
        help="去掉完整图和顶部五栏图的 S1/S2/... 标签；默认保留。",
    )
    parser.add_argument(
        "--no-legend",
        "--hide-legend",
        dest="no_legend",
        action="store_true",
        help="去掉 FP/FN 颜色图例；默认保留。",
    )
    parser.add_argument(
        "--no-local-labels",
        "--hide-local-labels",
        dest="no_local_labels",
        action="store_true",
        help="去掉局部放大图的标题、区域名、方法名和说明文字；默认保留。",
    )
    parser.add_argument(
        "--no-all-labels",
        "--hide-all-labels",
        dest="no_all_labels",
        action="store_true",
        help="去掉所有新增图注、S1标签、FP/FN图例和局部放大文字。",
    )
    parser.add_argument(
        "--export-parts",
        action="store_true",
        help="额外导出顶部五栏图和底部局部放大图。",
    )
    return parser.parse_args()


def _resolve_output_path(value, out_dir, default_name):
    text = str(value).strip()
    path = Path(text).expanduser() if text else Path(out_dir) / default_name
    if not path.is_absolute():
        path = Path(out_dir) / path
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main():
    args = parse_args()
    src_path = Path(args.src).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = str(args.out_prefix).strip() or "xor_zoom_grid_5col_sci"

    if not src_path.is_file():
        raise FileNotFoundError(f"source grid not found: {src_path}")

    no_all = bool(args.no_all_labels)
    show_header = not (bool(args.no_header) or no_all)
    show_row_tags = not (bool(args.no_row_tags) or no_all)
    show_legend = not (bool(args.no_legend) or no_all)
    show_local_labels = not (bool(args.no_local_labels) or no_all)

    src = Image.open(src_path).convert("RGB")
    if src.width % N_SOURCE_COLS != 0 or src.height % N_ROWS != 0:
        raise ValueError("Unexpected source grid size: %dx%d" % src.size)

    figure, tile_w, tile_h, header_h = assemble_main_grid(
        src,
        show_header=show_header,
        show_row_tags=show_row_tags,
        show_legend=show_legend,
    )
    draw_roi_boxes(figure, tile_w, tile_h, header_h)
    top_figure = figure.copy()
    figure = add_zoom_strip(
        figure,
        tile_w,
        tile_h,
        header_h,
        show_local_labels=show_local_labels,
    )

    combined_png = _resolve_output_path(args.out_png, out_dir, f"{out_prefix}.png")
    combined_tiff = _resolve_output_path(args.out_tiff, out_dir, f"{out_prefix}.tiff")
    combined_pdf = _resolve_output_path(args.out_pdf, out_dir, f"{out_prefix}.pdf")
    local_top = top_figure.height
    local_figure = figure.crop((0, local_top, figure.width, figure.height))

    figure.save(combined_png)
    figure.save(combined_tiff, dpi=(600, 600), compression="tiff_lzw")
    figure.save(combined_pdf, resolution=600.0)
    print("saved:", combined_png)
    print("saved:", combined_tiff)
    print("saved:", combined_pdf)

    if args.export_parts:
        top_png = out_dir / f"{out_prefix}_top_5col.png"
        local_png = out_dir / f"{out_prefix}_local_zoom.png"
        top_figure.save(top_png)
        local_figure.save(local_png)
        print("saved:", top_png)
        print("saved:", local_png)


if __name__ == "__main__":
    main()
