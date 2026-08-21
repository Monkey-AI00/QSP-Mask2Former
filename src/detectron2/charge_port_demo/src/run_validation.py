import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import __version__ as pillow_version

from config import OUTPUT_DIR, RANDOM_SEED, ROOT
from main import run_demo


def _hash_manifest(project_root: Path, output_root: Path) -> dict[str, Any]:
    excluded_parts = {"__pycache__", ".pytest_cache", ".git"}
    files: dict[str, str] = {}
    candidates = list((project_root / "src").rglob("*.py"))
    candidates += list((project_root / "tests").rglob("*.py"))
    candidates += list((project_root / "tools").rglob("*.py"))
    candidates += list((project_root / "data").rglob("*.json"))
    candidates += [project_root / "README.md", project_root / "requirements.txt"]
    candidates += [output_root / "validation_summary.json"]
    candidates += list((output_root / "cases").rglob("*.json"))
    candidates += [
        project_root.parent / "权利要求支持矩阵.md",
        project_root.parent / "专利提交前技术风险提示.md",
        project_root.parent / "研发佐证材料_专利对齐仿真版.md",
        project_root.parent / "研发佐证材料_专利对齐仿真版.docx",
    ]
    for path in sorted(set(candidates), key=lambda item: item.as_posix()):
        if not path.is_file() or path.name == "manifest_sha256.json" or excluded_parts.intersection(path.parts):
            continue
        try:
            relative = path.relative_to(project_root).as_posix()
        except ValueError:
            try:
                relative = f"outputs/{path.relative_to(output_root).as_posix()}"
            except ValueError:
                relative = f"deliverables/{path.name}"
        files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"algorithm": "SHA-256", "file_count": len(files), "files": files}


def _case_summary(result: dict[str, Any], project_root: Path, output_root: Path) -> dict[str, Any]:
    final_step = result["steps"][-1] if result["steps"] else {}
    confidence = final_step.get("confidence", {})
    artifacts = dict(result.get("artifacts", {}))

    def relative_path(value: str | None) -> str | None:
        if not value:
            return None
        path = Path(value)
        try:
            return path.relative_to(project_root).as_posix()
        except ValueError:
            return path.relative_to(output_root).as_posix()

    return {
        "status": result["status"],
        "view_sequence": [step["view_id"] for step in result["steps"]],
        "step_count": len(result["steps"]),
        "final_confidence": float(confidence.get("final_conf", 0.0)),
        "classification_confidence": float(confidence.get("cls_conf", 0.0)),
        "localization_confidence": float(confidence.get("kpt_conf", 0.0)),
        "prediction_method": result["final_prediction"].get("method"),
        "port_2d": result["final_prediction"].get("port_2d"),
        "port_3d_robot": result["final_prediction"].get("port_3d"),
        "evaluation": result["evaluation"],
        "log_path": relative_path(artifacts.get("log")),
        "frames_path": relative_path(artifacts.get("frames")),
        "animation_path": relative_path(artifacts.get("animation")),
    }


def run_all_cases(output_root: Path | None = None) -> dict[str, Any]:
    destination = Path(output_root) if output_root is not None else OUTPUT_DIR
    destination.mkdir(parents=True, exist_ok=True)
    cases: dict[str, Any] = {}
    for car_id in ("car_A", "car_B", "model3"):
        case_root = destination / "cases" / car_id
        result = run_demo(car_id, "view_far", output_root=case_root)
        cases[car_id] = _case_summary(result, ROOT, destination)

    summary = {
        "scope": "reproducible_simulation_engineering_prototype",
        "accuracy_claim": "not_a_real_world_accuracy_report",
        "seed": RANDOM_SEED,
        "command": "python src/run_validation.py",
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pillow": pillow_version,
        },
        "cases": cases,
    }
    summary_path = destination / "validation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = _hash_manifest(ROOT, destination)
    (destination / "manifest_sha256.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    summary = run_all_cases()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not all(case["status"] == "localized" for case in summary["cases"].values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
