"""One-time deterministic migration of replay JSON into inference/evaluation partitions."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def migrate(path: Path) -> None:
    original = json.loads(path.read_text(encoding="utf-8"))
    if "replay_output" in original:
        return
    replay = dict(original)
    view_id = replay.get("view_id", path.stem.removesuffix("_kpts"))
    true_port_2d = replay.pop("true_port_2d", None)
    replay.pop("true_port_score", None)
    ground_truth = {}
    if true_port_2d is not None:
        ground_truth = {
            "port_2d": true_port_2d,
            "source": "manual_simulation_reference",
            "coordinate_frame": "image_pixel",
        }
    migrated = {
        "schema_version": 1,
        "view_id": view_id,
        "replay_output": replay,
        "ground_truth": ground_truth,
    }
    path.write_text(json.dumps(migrated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    for path in sorted((ROOT / "data").glob("*/view_*_kpts.json")):
        migrate(path)


if __name__ == "__main__":
    main()
