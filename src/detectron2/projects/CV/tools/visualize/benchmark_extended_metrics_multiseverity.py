#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_extended_metrics 的多 severity 副本入口：

1) 调用原始 benchmark_extended_metrics.py 执行评测
2) 默认固定 severities = 0/0.25/0.5/0.75/1.0
3) 额外导出“表格形式”的多档结果：
   - table_multiseverity.csv
   - table_multiseverity.md
4) 可选评估各模型推理时长：
   - speed_fps.csv
   - speed_fps.json
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple


def _strip_existing_severities(argv: List[str]) -> List[str]:
    out: List[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--severities":
            i += 1
            while i < len(argv) and (not argv[i].startswith("--")):
                i += 1
            continue
        out.append(tok)
        i += 1
    return out


def _get_opt_value(argv: List[str], key: str) -> Optional[str]:
    for i, tok in enumerate(argv):
        if tok == key and i + 1 < len(argv):
            return argv[i + 1]
    return None


def _sev_tag(s: float) -> str:
    return f"{float(s):.2f}"


def _to_float(v: str) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def _fmt(v: Optional[float], nd: int = 2) -> str:
    if v is None:
        return ""
    return f"{v:.{nd}f}"


def _build_method_row_maps(rows: List[dict]) -> Dict[str, Dict[float, dict]]:
    by_method: Dict[str, Dict[float, dict]] = {}
    for r in rows:
        m = str(r.get("method", "")).strip()
        s = _to_float(str(r.get("severity", "")))
        if not m or s is None:
            continue
        by_method.setdefault(m, {})[float(s)] = r
    return by_method


def _pick_method(by_method: Dict[str, Dict[float, dict]], aliases: List[str]) -> Tuple[str, Dict[float, dict]]:
    for a in aliases:
        if a in by_method:
            return a, by_method[a]
    return aliases[0], {}


def _write_multisev_tables(out_root: str, severities: List[float]) -> Tuple[str, str]:
    tables_dir = os.path.join(out_root, "tables")
    metrics_csv = os.path.join(tables_dir, "metrics_raw.csv")
    if not os.path.isfile(metrics_csv):
        raise FileNotFoundError(f"metrics_raw.csv not found: {metrics_csv}")

    rows: List[dict] = []
    with open(metrics_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    by_method = _build_method_row_maps(rows)

    specs = [
        ("Feature-based", "Mask R-CNN", "R-50", ["MaskRCNN", "Mask R-CNN"]),
        ("Feature-based", "PointRend", "R-50", ["BASE", "PointRend(Base)", "PointRend"]),
        ("Feature-based", "Mask2Former", "R-50", ["Mask2Former"]),
        ("Geometry-aware", "M2F+Geo.Loss", "R-50", ["Mask2Former+GeoLoss", "M2F+Geo.Loss"]),
        ("Geometry-aware", "M2F+SDF", "R-50", ["Mask2Former+SDF", "M2F+SDF"]),
        ("Geometry-aware", "Ours (QSP-Mask2Former)", "R-50", ["Mask2Former+QSP", "QSP-M2F", "QSP-Mask2Former (ours)"]),
    ]

    out_csv = os.path.join(tables_dir, "table_multiseverity.csv")
    out_md = os.path.join(tables_dir, "table_multiseverity.md")

    fieldnames: List[str] = ["group", "method", "backbone"]
    for s in severities:
        tag = _sev_tag(s)
        fieldnames.extend([f"segm_AP@{tag}", f"boundary_iou@{tag}", f"hd95@{tag}"])

    table_rows: List[dict] = []
    for group, display_name, backbone, aliases in specs:
        _, mrows = _pick_method(by_method, aliases)
        item = {"group": group, "method": display_name, "backbone": backbone}
        for s in severities:
            rr = mrows.get(float(s), {})
            segm = _to_float(str(rr.get("segm_AP", "")))
            biou = _to_float(str(rr.get("boundary_iou", "")))
            hd = _to_float(str(rr.get("hd95", "")))
            tag = _sev_tag(s)
            item[f"segm_AP@{tag}"] = _fmt(segm, 2)
            item[f"boundary_iou@{tag}"] = _fmt(biou, 3)
            item[f"hd95@{tag}"] = _fmt(hd, 2)
        table_rows.append(item)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in table_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    md_lines: List[str] = []
    head = ["Group", "Method", "Backbone"]
    for s in severities:
        tag = _sev_tag(s)
        head.extend([f"SegmAP@{tag}", f"BIoU@{tag}", f"95HD@{tag}"])
    md_lines.append("| " + " | ".join(head) + " |")
    md_lines.append("|" + "|".join(["---"] * len(head)) + "|")
    for r in table_rows:
        vals = [str(r.get("group", "")), str(r.get("method", "")), str(r.get("backbone", ""))]
        for s in severities:
            tag = _sev_tag(s)
            vals.extend(
                [
                    str(r.get(f"segm_AP@{tag}", "")),
                    str(r.get(f"boundary_iou@{tag}", "")),
                    str(r.get(f"hd95@{tag}", "")),
                ]
            )
        md_lines.append("| " + " | ".join(vals) + " |")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    return out_csv, out_md


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-severity wrapper for benchmark_extended_metrics.py",
        add_help=True,
    )
    parser.add_argument(
        "--benchmark-script",
        default=os.path.join(os.path.dirname(__file__), "benchmark_extended_metrics.py"),
        help="原始 benchmark_extended_metrics.py 路径",
    )
    parser.add_argument(
        "--severities",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="多档 severity（默认 0/0.25/0.5/0.75/1.0）",
    )
    parser.add_argument(
        "--eval-speed",
        action="store_true",
        help="额外评估各模型推理时长；未指定 --speed-severity 时遍历所有 severity。",
    )
    parser.add_argument(
        "--speed-severity",
        type=float,
        default=None,
        help="仅评估指定的单个 severity；留空时遍历 --severities。",
    )
    parser.add_argument(
        "--speed-all-severities",
        action="store_true",
        help="明确要求速度评估遍历 --severities 中的所有 severity。",
    )
    parser.add_argument(
        "--speed-num-images",
        type=int,
        default=None,
        help="速度评估使用的图像数量；留空时由底层脚本默认使用 200 张。",
    )
    parser.add_argument(
        "--speed-warmup",
        type=int,
        default=None,
        help="速度评估 warmup 次数；留空时由底层脚本默认使用 20 次。",
    )
    parser.add_argument(
        "--speed-cuda-sync",
        action="store_true",
        help="速度评估计时前后同步 CUDA，GPU 计时更准确但略慢。",
    )

    args, passthrough = parser.parse_known_args()
    bench_script = os.path.abspath(str(args.benchmark_script))
    if not os.path.isfile(bench_script):
        raise FileNotFoundError(f"benchmark script not found: {bench_script}")

    passthrough = _strip_existing_severities(list(passthrough))
    speed_requested = bool(
        args.eval_speed
        or args.speed_severity is not None
        or args.speed_num_images is not None
        or args.speed_warmup is not None
        or args.speed_cuda_sync
        or args.speed_all_severities
    )
    if args.speed_all_severities and args.speed_severity is not None:
        parser.error("--speed-all-severities 与 --speed-severity 不能同时使用。")
    speed_all_severities = bool(args.speed_all_severities or args.speed_severity is None)
    cmd = [sys.executable, "-u", bench_script] + passthrough
    if speed_requested:
        cmd.append("--eval-speed")
        if speed_all_severities:
            cmd.append("--speed-all-severities")
        if args.speed_severity is not None:
            cmd += ["--speed-severity", str(float(args.speed_severity))]
        if args.speed_num_images is not None:
            cmd += ["--speed-num-images", str(int(args.speed_num_images))]
        if args.speed_warmup is not None:
            cmd += ["--speed-warmup", str(int(args.speed_warmup))]
        if args.speed_cuda_sync:
            cmd.append("--speed-cuda-sync")
    cmd += ["--severities"] + [str(float(s)) for s in args.severities]

    print("[RUN] " + " ".join(cmd))
    cp = subprocess.run(cmd)
    if cp.returncode != 0:
        raise SystemExit(cp.returncode)

    out_root = _get_opt_value(passthrough, "--out-root")
    if not out_root:
        raise ValueError("需要在参数中提供 --out-root，才能导出多档表格。")
    out_root = os.path.abspath(out_root)

    out_csv, out_md = _write_multisev_tables(out_root=out_root, severities=[float(s) for s in args.severities])
    print(f"[TABLE] written: {out_csv}")
    print(f"[TABLE] written: {out_md}")
    if speed_requested:
        speed_csv = os.path.join(out_root, "tables", "speed_fps.csv")
        speed_json = os.path.join(out_root, "tables", "speed_fps.json")
        print(f"[SPEED] written: {speed_csv}")
        print(f"[SPEED] written: {speed_json}")


if __name__ == "__main__":
    main()

