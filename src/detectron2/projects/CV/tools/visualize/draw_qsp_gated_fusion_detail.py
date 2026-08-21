#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""绘制与当前 QSP 代码一致的 Gated residual fusion(detail) 简图。"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


def _box(ax, xy, wh, text, face, edge="#5f6b76", fontsize=11, lw=1.8, radius=0.04):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.015,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def _tile(ax, x, y, w, h, label, face="#253a91", text_color="white"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor="#263238", linewidth=1.2))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9, color=text_color)


def _arrow(ax, start: Tuple[float, float], end: Tuple[float, float], color="#27333d", lw=1.7):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=lw,
            color=color,
            shrinkA=3,
            shrinkB=3,
        )
    )


def _circle(ax, center, radius, text, face="#ffffff", edge="#27333d", fontsize=16):
    ax.add_patch(Circle(center, radius, facecolor=face, edgecolor=edge, linewidth=1.8))
    ax.text(center[0], center[1], text, ha="center", va="center", fontsize=fontsize)


def _draw(out_path: str, dpi: int = 220) -> None:
    fig, ax = plt.subplots(figsize=(16, 9), dpi=dpi)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        8,
        8.65,
        "Gated residual fusion (detail)",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color="#177a88",
    )

    # Fusion stack X_i.
    stack_x, stack_y, stack_w, stack_h = 0.35, 1.65, 2.65, 6.25
    ax.add_patch(
        FancyBboxPatch(
            (stack_x, stack_y),
            stack_w,
            stack_h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor="#f8fbff",
            edgecolor="#263b9b",
            linestyle=(0, (6, 4)),
            linewidth=1.8,
        )
    )
    ax.text(1.68, 8.05, r"$X_i\in\mathbb{R}^{4\times H\times W}$", ha="center", fontsize=13)
    tile_x, tile_w, tile_h = 0.65, 1.15, 0.92
    tile_ys = [6.45, 5.05, 3.65, 2.25]
    tile_labels = [
        r"$\sigma(M_i^{raw})$" + "\nprobability",
        r"$P_i^{align}$",
        r"$\sigma(M_i^{raw})-P_i^{align}$",
        r"$V_i$" + "\nsigned visual map",
    ]
    for y, label in zip(tile_ys, tile_labels):
        _tile(ax, tile_x, y, tile_w, tile_h, label)

    # Head boxes.
    _box(
        ax,
        (3.75, 6.62),
        (1.35, 0.78),
        r"$G(\cdot)$" + "\nGate head",
        "#dff1dc",
        "#86b879",
        fontsize=10,
    )
    _box(
        ax,
        (3.75, 5.05),
        (1.35, 0.78),
        r"$C(\cdot)$" + "\nOccluder head",
        "#fff0d5",
        "#e3a947",
        fontsize=10,
    )
    _box(
        ax,
        (3.75, 3.35),
        (1.35, 0.78),
        r"$D(\cdot)$" + "\nDelta head",
        "#f9dfe0",
        "#d8898c",
        fontsize=10,
    )

    # Shared X_i fan-out; one vertical bus makes the shared input explicit.
    bus_x = 3.25
    ax.plot([bus_x, bus_x], [2.65, 7.0], color="#263b9b", linewidth=1.5)
    for y in [6.99, 5.42, 3.72]:
        _arrow(ax, (bus_x, y), (3.75, y))
    _arrow(ax, (stack_x + stack_w, 6.9), (bus_x, 6.9))

    # Gate path: G(X)+beta -> sigmoid -> g_raw.
    _circle(ax, (5.62, 7.01), 0.23, "+", face="#ffffff")
    ax.text(5.62, 7.48, r"$\beta_i$ (bias)", ha="center", fontsize=11)
    _arrow(ax, (5.10, 7.01), (5.39, 7.01))
    _arrow(ax, (5.85, 7.01), (6.28, 7.01))
    _box(ax, (6.28, 6.62), (1.2, 0.78), r"$\sigma(\cdot)$", "#eef2f6")
    _arrow(ax, (7.48, 7.01), (8.05, 7.01))
    ax.text(8.25, 7.01, r"$g_i^{raw}$", va="center", fontsize=12)

    # Occluder path: C(X) -> c -> 1-c.
    _arrow(ax, (5.10, 5.44), (6.0, 5.44))
    ax.text(6.22, 5.44, r"$c_i$", va="center", fontsize=12)
    _arrow(ax, (6.42, 5.44), (6.82, 5.44))
    _box(ax, (6.82, 5.05), (1.05, 0.78), r"$1-c_i$", "#e7eef8", "#597ca8")
    _arrow(ax, (7.87, 5.44), (9.55, 5.44))

    # Delta path: D(X) + signed prior -> tanh -> delta.
    _circle(ax, (5.62, 3.74), 0.23, "+", face="#ffffff")
    _arrow(ax, (5.10, 3.74), (5.39, 3.74))
    _box(
        ax,
        (4.88, 2.25),
        (1.48, 0.72),
        r"$2P_i^{align}-1$" + "\nSigned prior",
        "#eee8f8",
        "#8c73ad",
        fontsize=10,
    )
    _arrow(ax, (5.62, 2.97), (5.62, 3.49))
    _arrow(ax, (5.85, 3.74), (6.25, 3.74))
    _box(ax, (6.25, 3.35), (0.95, 0.78), r"$\tanh$", "#f0f0f0")
    _arrow(ax, (7.2, 3.74), (8.05, 3.74))
    ax.text(8.25, 3.74, r"$\delta_i$", va="center", fontsize=12)

    # Three-factor correction path.
    _circle(ax, (9.95, 5.44), 0.23, r"$\times$", face="#ffffff", fontsize=12)
    _arrow(ax, (8.45, 7.01), (9.95, 5.72))
    _arrow(ax, (9.55, 5.44), (10.18, 5.44))
    ax.text(10.55, 5.44, r"$g_i^{raw}(1-c_i)$", va="center", fontsize=11)

    _circle(ax, (11.15, 4.25), 0.23, r"$\times$", face="#ffffff", fontsize=12)
    _arrow(ax, (10.15, 5.28), (11.02, 4.42))
    _arrow(ax, (8.45, 3.74), (11.02, 4.12))
    ax.text(11.15, 3.73, r"$\times\,\alpha$", ha="center", fontsize=11, color="#5b3b8e")
    _arrow(ax, (11.15, 3.98), (11.15, 3.42))

    # Raw-logit skip connection and final addition.
    _circle(ax, (12.25, 3.05), 0.23, "+", face="#ffffff")
    _arrow(ax, (11.15, 3.42), (12.25, 3.28))
    _box(
        ax,
        (12.0, 6.9),
        (1.25, 0.78),
        r"$M_i^{raw}$" + "\nlogits",
        "#edf2f7",
        "#687782",
        fontsize=10,
    )
    _arrow(ax, (12.62, 6.9), (12.62, 3.38), color="#4c5963")
    _arrow(ax, (12.62, 3.38), (12.45, 3.18), color="#4c5963")
    ax.text(12.95, 5.25, "skip", fontsize=10, color="#4c5963", rotation=90, va="center")

    _arrow(ax, (12.48, 3.05), (13.15, 3.05))
    _box(
        ax,
        (13.15, 2.66),
        (1.35, 0.78),
        r"$M_i^{fuse}$" + "\nlogits",
        "#dce9ff",
        "#567aa8",
        fontsize=10,
    )
    _arrow(ax, (14.5, 3.05), (14.95, 3.05))
    _box(ax, (14.95, 2.66), (0.75, 0.78), r"$\sigma$", "#eef2f6", fontsize=11)
    ax.text(15.33, 2.2, r"$\sigma(M_i^{fuse})$", ha="center", fontsize=11)

    # Bottom exact equation and note.
    ax.text(
        8.0,
        0.85,
        r"$M_i^{fuse}=M_i^{raw}+\alpha\,g_i^{raw}(1-c_i)\,\delta_i$",
        ha="center",
        fontsize=17,
        color="#20252b",
    )
    ax.text(
        8.0,
        0.35,
        "Raw mask and fused mask are logits; sigmoid is applied for probability visualization.",
        ha="center",
        fontsize=10,
        color="#5b6570",
    )

    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    svg_path = os.path.splitext(out_path)[0] + ".svg"
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out_path}")
    print(f"saved: {svg_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw QSP gated residual fusion detail.")
    parser.add_argument(
        "--out",
        default="qsp_gated_residual_fusion_detail.png",
        help="输出 PNG 路径；同时生成同名 SVG",
    )
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    _draw(out_path, dpi=max(120, int(args.dpi)))


if __name__ == "__main__":
    main()

