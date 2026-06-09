#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Table S1: Severity 客观定义与分档统计（双口径）

视图A（正文口径）:
- 将图像转为 YCrCb，取亮度通道 Y
- 在 ROI 内统计高光比例 r = |{Y > tau}| / |ROI|
- 依据 bin_edges 将 r 分箱到 severities（默认 5 档）

视图B（附录口径）:
- 读取 benchmark 生成的 highlight_s* 离线目录
- 统计每档图像数/实例数，并可计算该目录下样本的 r 分布
- 输出 A/B 档位映射摘要（pipeline_severity vs y_assigned_severity）
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _fmt_float(x: float, nd: int = 6) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return ""
    return f"{float(x):.{nd}f}"


def _parse_float_list(raw: str) -> List[float]:
    vals: List[float] = []
    for t in str(raw).split(","):
        tt = t.strip()
        if not tt:
            continue
        vals.append(float(tt))
    return vals


def _parse_bin_edges(raw: str, severities: Sequence[float]) -> List[float]:
    if str(raw).strip():
        edges = _parse_float_list(raw)
    else:
        sev = sorted(float(x) for x in severities)
        mids = [(sev[i] + sev[i + 1]) / 2.0 for i in range(len(sev) - 1)]
        edges = [0.0] + mids + [1.0]
    if len(edges) != len(severities) + 1:
        raise ValueError(
            f"bin_edges 数量错误：期望 {len(severities) + 1}，实际 {len(edges)}"
        )
    edges = [float(max(0.0, min(1.0, e))) for e in edges]
    for i in range(len(edges) - 1):
        if edges[i + 1] < edges[i]:
            raise ValueError(f"bin_edges 非递增：{edges}")
    return edges


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _valid_categories(cats: List[dict]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for c in cats:
        cid = int(c.get("id", -1))
        name = str(c.get("name", "")).strip()
        if cid >= 0 and name and name != "_background_":
            out[cid] = name
    return out


def _ann_to_mask(ann: dict, h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=np.uint8)
    seg = ann.get("segmentation", None)
    # polygon
    if isinstance(seg, list) and len(seg) > 0:
        for poly in seg:
            if not isinstance(poly, list) or len(poly) < 6:
                continue
            pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
            cv2.fillPoly(out, [pts.astype(np.int32)], color=1)
        if np.any(out > 0):
            return out
    # bbox fallback
    bbox = ann.get("bbox", None)
    if isinstance(bbox, list) and len(bbox) >= 4:
        x, y, bw, bh = [float(v) for v in bbox[:4]]
        x1 = int(max(0, min(w - 1, round(x))))
        y1 = int(max(0, min(h - 1, round(y))))
        x2 = int(max(0, min(w - 1, round(x + bw - 1))))
        y2 = int(max(0, min(h - 1, round(y + bh - 1))))
        if x2 >= x1 and y2 >= y1:
            out[y1 : y2 + 1, x1 : x2 + 1] = 1
    return out


def _build_roi_mask(
    anns: List[dict],
    h: int,
    w: int,
    roi_source: str,
    valid_cats: Optional[Dict[int, str]] = None,
) -> np.ndarray:
    roi = np.zeros((h, w), dtype=np.uint8)
    for ann in anns:
        if valid_cats is not None:
            cid = int(ann.get("category_id", -1))
            if cid not in valid_cats:
                continue
        if str(roi_source) == "bbox":
            bbox = ann.get("bbox", None)
            if isinstance(bbox, list) and len(bbox) >= 4:
                x, y, bw, bh = [float(v) for v in bbox[:4]]
                x1 = int(max(0, min(w - 1, round(x))))
                y1 = int(max(0, min(h - 1, round(y))))
                x2 = int(max(0, min(w - 1, round(x + bw - 1))))
                y2 = int(max(0, min(h - 1, round(y + bh - 1))))
                if x2 >= x1 and y2 >= y1:
                    roi[y1 : y2 + 1, x1 : x2 + 1] = 1
        else:
            roi = np.maximum(roi, _ann_to_mask(ann, h=h, w=w))
    return roi


def _calc_r_y_ratio(img_bgr: np.ndarray, roi01: np.ndarray, tau: int) -> float:
    y = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    roi = roi01 > 0
    denom = int(roi.sum())
    if denom <= 0:
        return float("nan")
    sat = (y > int(tau)) & roi
    return float(sat.sum()) / float(denom)


def _assign_severity(r: float, severities: Sequence[float], edges: Sequence[float]) -> float:
    if not np.isfinite(r):
        return float(severities[0])
    rr = float(max(0.0, min(1.0, r)))
    if rr >= float(edges[-1]):
        return float(severities[-1])
    idx = int(np.searchsorted(np.asarray(edges), rr, side="right") - 1)
    idx = int(max(0, min(len(severities) - 1, idx)))
    return float(severities[idx])


@dataclass
class ImageStat:
    file_name: str
    image_id: int
    num_instances: int
    r: float
    severity_a: float
    pipeline_severity: Optional[float] = None


def _collect_stats_for_dataset(
    dataset_root: str,
    json_file: str,
    tau: int,
    roi_source: str,
    severities: Sequence[float],
    edges: Sequence[float],
) -> Tuple[List[ImageStat], Dict[str, int]]:
    root = os.path.abspath(dataset_root)
    coco = _read_json(os.path.join(root, json_file))
    valid_cats = _valid_categories(list(coco.get("categories", [])))
    images = list(coco.get("images", []))
    anns = list(coco.get("annotations", []))
    id_to_anns: Dict[int, List[dict]] = {}
    for a in anns:
        iid = int(a.get("image_id", -1))
        id_to_anns.setdefault(iid, []).append(a)
    stats: List[ImageStat] = []
    missing = 0
    for im in images:
        iid = int(im.get("id", -1))
        fn = str(im.get("file_name", ""))
        p = os.path.join(root, fn)
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            missing += 1
            continue
        h, w = img.shape[:2]
        ann_list = id_to_anns.get(iid, [])
        roi = _build_roi_mask(ann_list, h=h, w=w, roi_source=roi_source, valid_cats=valid_cats)
        r = _calc_r_y_ratio(img, roi, tau=tau)
        sev = _assign_severity(r, severities=severities, edges=edges)
        n_inst = sum(1 for a in ann_list if int(a.get("category_id", -1)) in valid_cats)
        stats.append(ImageStat(file_name=fn, image_id=iid, num_instances=n_inst, r=r, severity_a=sev))
    meta = {"num_images_json": len(images), "num_annotations_json": len(anns), "num_missing_images": missing}
    return stats, meta


def _aggregate_by_severity(rows: List[ImageStat], severities: Sequence[float]) -> List[dict]:
    out: List[dict] = []
    for s in severities:
        ss = [r for r in rows if float(r.severity_a) == float(s)]
        out.append(
            {
                "severity": float(s),
                "num_images": int(len(ss)),
                "num_instances": int(sum(r.num_instances for r in ss)),
                "r_mean": float(np.nanmean([r.r for r in ss])) if ss else float("nan"),
                "r_median": float(np.nanmedian([r.r for r in ss])) if ss else float("nan"),
            }
        )
    return out


def _discover_pipeline_dirs(pipeline_root: str) -> List[Tuple[float, str]]:
    root = os.path.abspath(pipeline_root)
    out: List[Tuple[float, str]] = []
    if not os.path.isdir(root):
        return out
    patt = re.compile(r"^highlight_s(\d+)p(\d+)$")
    for n in sorted(os.listdir(root)):
        m = patt.match(str(n))
        if not m:
            continue
        s = float(f"{int(m.group(1))}.{int(m.group(2))}")
        out.append((s, os.path.join(root, n)))
    return out


def _write_csv(path: str, rows: List[dict], fieldnames: Sequence[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _write_md_table_s1(path: str, view_a: List[dict], view_b: List[dict], mapping: List[dict], edges: Sequence[float], tau: int, roi_source: str) -> None:
    lines: List[str] = []
    lines.append("# Table S1: Severity 定义与分档统计")
    lines.append("")
    lines.append("## 协议")
    lines.append(f"- 亮度阈值: `tau={int(tau)}`（Y 通道）")
    lines.append(f"- ROI来源: `{roi_source}`")
    lines.append(f"- 分箱边界: `{[round(float(e), 6) for e in edges]}`")
    lines.append("")
    lines.append("## 视图A（Y通道高光比例 r 分箱）")
    lines.append("")
    lines.append("| Severity | 图像数 | 实例数 | r_mean | r_median |")
    lines.append("|---:|---:|---:|---:|---:|")
    for r in view_a:
        lines.append(
            f"| {r['severity']:.2f} | {int(r['num_images'])} | {int(r['num_instances'])} | {_fmt_float(r['r_mean'],4)} | {_fmt_float(r['r_median'],4)} |"
        )
    if view_b:
        lines.append("")
        lines.append("## 视图B（pipeline highlight_s* 目录统计）")
        lines.append("")
        lines.append("| pipeline_severity | 图像数 | 实例数 | r_mean_in_dir |")
        lines.append("|---:|---:|---:|---:|")
        for r in view_b:
            lines.append(
                f"| {r['pipeline_severity']:.2f} | {int(r['num_images'])} | {int(r['num_instances'])} | {_fmt_float(r['r_mean_in_dir'],4)} |"
            )
    if mapping:
        lines.append("")
        lines.append("## 双口径映射摘要")
        lines.append("")
        lines.append("| pipeline_severity | y_assigned_severity | 图像数 | 实例数 |")
        lines.append("|---:|---:|---:|---:|")
        for r in mapping:
            lines.append(
                f"| {r['pipeline_severity']:.2f} | {r['y_assigned_severity']:.2f} | {int(r['num_images'])} | {int(r['num_instances'])} |"
            )
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Table S1 severity stats.")
    p.add_argument("--dataset-root", required=True, help="COCO 数据集目录（图片+json）")
    p.add_argument("--json-file", default="plug_train.json", help="COCO 标注文件名（相对 dataset-root）")
    p.add_argument("--out-dir", required=True, help="输出目录")
    p.add_argument("--tau", type=int, default=245, help="Y 通道高光阈值，判定条件为 Y > tau")
    p.add_argument("--roi-source", choices=["mask", "bbox"], default="mask", help="ROI 计算来源")
    p.add_argument("--severities", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0], help="Severity 离散档位")
    p.add_argument("--bin-edges", default="", help="可选：逗号分隔的分箱边界（长度=severity数+1）")
    p.add_argument("--pipeline-root", default="", help="可选：benchmark datasets_cache 目录（含 highlight_s* 子目录）")
    p.add_argument("--seed", type=int, default=0, help="记录到 manifest 的随机种子")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = os.path.abspath(args.out_dir)
    _ensure_dir(out_dir)
    severities = sorted(float(s) for s in args.severities)
    edges = _parse_bin_edges(args.bin_edges, severities=severities)

    # 视图A：正文口径
    view_a_rows, meta_a = _collect_stats_for_dataset(
        dataset_root=args.dataset_root,
        json_file=args.json_file,
        tau=int(args.tau),
        roi_source=str(args.roi_source),
        severities=severities,
        edges=edges,
    )
    agg_a = _aggregate_by_severity(view_a_rows, severities=severities)

    # 视图B+映射：附录口径
    agg_b: List[dict] = []
    mapping_rows: List[dict] = []
    pipeline_dirs = _discover_pipeline_dirs(str(args.pipeline_root)) if str(args.pipeline_root).strip() else []
    for ps, droot in pipeline_dirs:
        rows_b, _ = _collect_stats_for_dataset(
            dataset_root=droot,
            json_file=args.json_file,
            tau=int(args.tau),
            roi_source=str(args.roi_source),
            severities=severities,
            edges=edges,
        )
        for r in rows_b:
            r.pipeline_severity = float(ps)
        agg_b.append(
            {
                "pipeline_severity": float(ps),
                "num_images": int(len(rows_b)),
                "num_instances": int(sum(r.num_instances for r in rows_b)),
                "r_mean_in_dir": float(np.nanmean([r.r for r in rows_b])) if rows_b else float("nan"),
            }
        )
        # pipeline_severity vs y_assigned_severity
        for s in severities:
            ss = [r for r in rows_b if float(r.severity_a) == float(s)]
            if not ss:
                continue
            mapping_rows.append(
                {
                    "pipeline_severity": float(ps),
                    "y_assigned_severity": float(s),
                    "num_images": int(len(ss)),
                    "num_instances": int(sum(r.num_instances for r in ss)),
                }
            )

    # 导出
    csv_a = os.path.join(out_dir, "table_s1_severity_viewA.csv")
    csv_b = os.path.join(out_dir, "table_s1_severity_viewB.csv")
    csv_m = os.path.join(out_dir, "table_s1_severity_mapping.csv")
    _write_csv(csv_a, agg_a, ["severity", "num_images", "num_instances", "r_mean", "r_median"])
    _write_csv(csv_b, agg_b, ["pipeline_severity", "num_images", "num_instances", "r_mean_in_dir"])
    _write_csv(csv_m, mapping_rows, ["pipeline_severity", "y_assigned_severity", "num_images", "num_instances"])
    md_path = os.path.join(out_dir, "table_s1_severity.md")
    _write_md_table_s1(md_path, agg_a, agg_b, mapping_rows, edges=edges, tau=int(args.tau), roi_source=str(args.roi_source))

    manifest = {
        "timestamp": int(time.time()),
        "dataset_root": os.path.abspath(args.dataset_root),
        "json_file": str(args.json_file),
        "out_dir": out_dir,
        "tau": int(args.tau),
        "roi_source": str(args.roi_source),
        "severities": [float(x) for x in severities],
        "bin_edges": [float(x) for x in edges],
        "pipeline_root": os.path.abspath(args.pipeline_root) if str(args.pipeline_root).strip() else "",
        "seed": int(args.seed),
        "meta_viewA": meta_a,
        "num_viewA_image_rows": int(len(view_a_rows)),
        "num_viewB_dirs": int(len(pipeline_dirs)),
    }
    manifest_path = os.path.join(out_dir, "severity_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[S1] written: {csv_a}")
    print(f"[S1] written: {csv_b}")
    print(f"[S1] written: {csv_m}")
    print(f"[S1] written: {md_path}")
    print(f"[S1] written: {manifest_path}")


if __name__ == "__main__":
    main()

