import argparse
import json
from pathlib import Path

from active_view import select_next_best_view
from confidence import compute_confidence
from config import (
    ALPHA,
    BETA,
    CAMERA_INTRINSICS,
    CONF_THRESHOLD,
    DEFAULT_CAR_ID,
    DEFAULT_INIT_VIEW,
    FRAME_DIR,
    LOG_DIR,
    OUTPUT_DIR,
    USE_PSEUDO_3D,
)
from infer_port import infer_port_2d, infer_port_3d
from load_data import list_views, load_graph, load_meta, load_view
from perception import run_perception
from planner import build_explore_action, build_work_action
from semantic_graph import SemanticGraph
from visualize import build_video_from_frames, draw_action_panel, draw_confidence_panel, draw_keypoints, draw_port_result, save_frame


def make_output_dirs() -> None:
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for path in FRAME_DIR.glob("*.png"):
        path.unlink()


def run_demo(car_id: str, init_view_id: str) -> dict:
    make_output_dirs()
    meta = load_meta(car_id)
    graph = SemanticGraph(load_graph(car_id))
    all_views = list_views(car_id)
    current_view_id = init_view_id
    step_idx = 0
    finished = False
    run_log = {"car_id": car_id, "init_view_id": init_view_id, "meta": meta, "steps": []}

    while not finished:
        step_idx += 1
        sample = load_view(car_id, current_view_id)
        perception_result = run_perception(sample)
        conf_result = compute_confidence(perception_result, alpha=ALPHA, beta=BETA)

        frame = sample["rgb"].copy()
        frame = draw_keypoints(frame, perception_result["keypoints_2d"], perception_result["keypoint_scores"])
        frame = draw_confidence_panel(frame, conf_result, perception_result)

        if conf_result["final_conf"] < CONF_THRESHOLD:
            explore_result = select_next_best_view(car_id, current_view_id, all_views, load_view)
            action = build_explore_action(explore_result["next_view_id"])
            frame = draw_action_panel(frame, action, title="explore")
            save_frame(frame, f"{step_idx:03d}_{current_view_id}_explore")
            run_log["steps"].append(
                {
                    "view_id": current_view_id,
                    "cls_conf": conf_result["cls_conf"],
                    "kpt_conf": conf_result["kpt_conf"],
                    "final_conf": conf_result["final_conf"],
                    "mode": "explore",
                    "next_view_id": explore_result["next_view_id"],
                }
            )
            if not explore_result["need_explore"] or explore_result["next_view_id"] == current_view_id:
                finished = True
            else:
                current_view_id = explore_result["next_view_id"]
            continue

        if USE_PSEUDO_3D:
            port_result = infer_port_3d(perception_result, sample, graph, CAMERA_INTRINSICS)
        else:
            port_result = infer_port_2d(perception_result, graph)

        action = build_work_action(port_result)
        frame = draw_port_result(frame, port_result)
        frame = draw_action_panel(frame, action, title="work")
        save_frame(frame, f"{step_idx:03d}_{current_view_id}_work")
        run_log["steps"].append(
            {
                "view_id": current_view_id,
                "cls_conf": conf_result["cls_conf"],
                "kpt_conf": conf_result["kpt_conf"],
                "final_conf": conf_result["final_conf"],
                "mode": "work",
                "port_2d": port_result.get("port_2d"),
                "port_3d": port_result.get("port_3d"),
                "port_method": port_result.get("method"),
            }
        )
        finished = True

    build_video_from_frames(FRAME_DIR, OUTPUT_DIR / "demo_video.mp4")
    log_path = LOG_DIR / f"run_{car_id}_{init_view_id}.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(run_log, f, ensure_ascii=False, indent=2)
    return run_log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--car_id", default=DEFAULT_CAR_ID)
    parser.add_argument("--init_view_id", default=DEFAULT_INIT_VIEW)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_demo(args.car_id, args.init_view_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("Demo finished.")


if __name__ == "__main__":
    main()
