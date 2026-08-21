#!/usr/bin/env bash
set -u

# One-click evaluation for Table-3 K ablation: K={1,4,6,8,16,24,32}
# It runs benchmark_extended_metrics.py per K and merges results into one CSV.

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/home/users1/sjw/cursor/workspace}"
MASK2FORMER_ROOT="${MASK2FORMER_ROOT:-${WORKSPACE_ROOT}/src/Mask2Former}"
GPU_ID="${GPU_ID:-1}"

# Checkpoint paths (fill manually if needed).
# You can pass them from command line env, e.g.:
#   CKPT_K1=/path/to/k1/model_final.pth CKPT_K4=/path/to/k4/model_final.pth CKPT_K6=/path/to/k6/model_final.pth CKPT_K24=/path/to/k24/model_final.pth ...
#
# Suggested defaults (can be replaced one by one):
CKPT_K1="${CKPT_K1:-${WORKSPACE_ROOT}/outputs/gangkou/output/table3_k_ablation_0619/k1/model_final.pth}"
CKPT_K4="${CKPT_K4:-${WORKSPACE_ROOT}/outputs/gangkou/output/table3_k_ablation_0622/k4/model_final.pth}"
CKPT_K6="${CKPT_K6:-${WORKSPACE_ROOT}/outputs/gangkou/output/table3_k_ablation_0623/k6/model_final.pth}"
CKPT_K8="${CKPT_K8:-${WORKSPACE_ROOT}/outputs/gangkou/output/table3_k_ablation_0620/k8/model_final.pth}"
CKPT_K16="${CKPT_K16:-${WORKSPACE_ROOT}/outputs/gangkou/output/table3_k_ablation_0621/k16/model_final.pth}"
CKPT_K24="${CKPT_K24:-${WORKSPACE_ROOT}/outputs/gangkou/output/table3_k_ablation_0624/k24/model_final.pth}"
CKPT_K32="${CKPT_K32:-${WORKSPACE_ROOT}/outputs/gangkou/output/table3_k_ablation_0621/k32/model_final.pth}"

# Evaluation output root:
#   ${EVAL_ROOT}/k1/...
#   ${EVAL_ROOT}/k4/...
EVAL_ROOT="${EVAL_ROOT:-${WORKSPACE_ROOT}/outputs/gangkou/output/table3_k_ablation_0622_eval}"

VAL_ROOT="${VAL_ROOT:-${WORKSPACE_ROOT}/datasets/gangkou/plug_test_80}"
VAL_JSON="${VAL_JSON:-plug_test.json}"
SEVERITY="${SEVERITY:-1.0}"
SCORE_THR="${SCORE_THR:-0.5}"

BENCH_PY="${WORKSPACE_ROOT}/src/detectron2/projects/CV/tools/visualize/benchmark_extended_metrics.py"
CFG_DIR="${WORKSPACE_ROOT}/src/detectron2/projects/CV/configs/InstanceSegmentation"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${WORKSPACE_ROOT}/src/detectron2:${WORKSPACE_ROOT}/src/detectron2/projects/CV:${MASK2FORMER_ROOT}"

mkdir -p "${EVAL_ROOT}"

echo "[INFO] WORKSPACE_ROOT=${WORKSPACE_ROOT}"
echo "[INFO] MASK2FORMER_ROOT=${MASK2FORMER_ROOT}"
echo "[INFO] EVAL_ROOT=${EVAL_ROOT}"
echo "[INFO] VAL_ROOT=${VAL_ROOT}, VAL_JSON=${VAL_JSON}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] CKPT_K1=${CKPT_K1}"
echo "[INFO] CKPT_K4=${CKPT_K4}"
echo "[INFO] CKPT_K6=${CKPT_K6}"
echo "[INFO] CKPT_K8=${CKPT_K8}"
echo "[INFO] CKPT_K16=${CKPT_K16}"
echo "[INFO] CKPT_K24=${CKPT_K24}"
echo "[INFO] CKPT_K32=${CKPT_K32}"

for k in 1 4 6 8 16 24 32; do
  cfg="${CFG_DIR}/mask2former_R50_plug_k${k}.yaml"
  ckpt=""
  case "${k}" in
    1) ckpt="${CKPT_K1}" ;;
    4) ckpt="${CKPT_K4}" ;;
    6) ckpt="${CKPT_K6}" ;;
    8) ckpt="${CKPT_K8}" ;;
    16) ckpt="${CKPT_K16}" ;;
    24) ckpt="${CKPT_K24}" ;;
    32) ckpt="${CKPT_K32}" ;;
  esac
  out_k="${EVAL_ROOT}/k${k}"

  if [[ ! -f "${cfg}" ]]; then
    echo "[WARN] Skip k=${k}: config not found: ${cfg}"
    continue
  fi
  if [[ -z "${ckpt}" || ! -f "${ckpt}" ]]; then
    echo "[WARN] Skip k=${k}: checkpoint not found: ${ckpt}"
    continue
  fi

  echo "[RUN] Evaluate k=${k}"
  python -u "${BENCH_PY}" \
    --clean-root "${VAL_ROOT}" \
    --json-file "${VAL_JSON}" \
    --out-root "${out_k}" \
    --mask2former-root "${MASK2FORMER_ROOT}" \
    --config-file-mask2former "${cfg}" \
    --weights-mask2former "${ckpt}" \
    --severities "${SEVERITY}" \
    --score-thr "${SCORE_THR}" \
    --overwrite-cache
done

TABLE3_CSV="${EVAL_ROOT}/table3_k_ablation.csv"
TABLE3_MD="${EVAL_ROOT}/table3_k_ablation.md"

python - "${EVAL_ROOT}" "${SEVERITY}" "${TABLE3_CSV}" "${TABLE3_MD}" <<'PY'
import csv
import os
import sys
from typing import Dict, Optional

eval_root = sys.argv[1]
target_sev = float(sys.argv[2])
out_csv = sys.argv[3]
out_md = sys.argv[4]

ks = [1, 4, 6, 8, 16, 24, 32]

def pick_row(metrics_csv: str, sev: float) -> Optional[Dict[str, str]]:
    if not os.path.isfile(metrics_csv):
        return None
    rows = []
    with open(metrics_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        return None
    exact = []
    for r in rows:
        try:
            s = float(r.get("severity", "nan"))
        except Exception:
            continue
        if abs(s - sev) < 1e-9:
            exact.append(r)
    if exact:
        return exact[0]
    # fallback: nearest severity
    cand = []
    for r in rows:
        try:
            s = float(r.get("severity", "nan"))
        except Exception:
            continue
        cand.append((abs(s - sev), r))
    if not cand:
        return None
    cand.sort(key=lambda x: x[0])
    return cand[0][1]

table_rows = []
for k in ks:
    metrics_csv = os.path.join(eval_root, f"k{k}", "tables", "metrics_raw.csv")
    row = pick_row(metrics_csv, target_sev)
    canonical = "yes" if k == 1 else "no"
    prior_bank = "no" if k == 1 else "yes"
    method = "K=1 canonical prior" if k == 1 else f"K={k} prior-bank"
    if row is None:
        table_rows.append({
            "method": method,
            "k": str(k),
            "canonical_prior": canonical,
            "prior_bank": prior_bank,
            "boundary_iou": "",
            "hd95": "",
            "segm_AP": "",
            "source_metrics_csv": metrics_csv,
            "status": "missing",
        })
        continue
    table_rows.append({
        "method": method,
        "k": str(k),
        "canonical_prior": canonical,
        "prior_bank": prior_bank,
        "boundary_iou": row.get("boundary_iou", ""),
        "hd95": row.get("hd95", ""),
        "segm_AP": row.get("segm_AP", ""),
        "source_metrics_csv": metrics_csv,
        "status": "ok",
    })

fieldnames = [
    "method",
    "k",
    "canonical_prior",
    "prior_bank",
    "boundary_iou",
    "hd95",
    "segm_AP",
    "source_metrics_csv",
    "status",
]
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in table_rows:
        w.writerow(r)

lines = []
lines.append("| Method | Canonical Prior | Prior-Bank | Boundary IoU | 95HD | Segm AP |")
lines.append("|---|---:|---:|---:|---:|---:|")
for r in table_rows:
    cp = "✓" if r["canonical_prior"] == "yes" else ""
    pb = "✓" if r["prior_bank"] == "yes" else ""
    lines.append(
        f"| {r['method']} | {cp} | {pb} | {r['boundary_iou']} | {r['hd95']} | {r['segm_AP']} |"
    )
with open(out_md, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print(f"[TABLE3] written csv: {out_csv}")
print(f"[TABLE3] written md : {out_md}")
PY

echo "[DONE] Table-3 summary:"
echo "  - ${TABLE3_CSV}"
echo "  - ${TABLE3_MD}"

