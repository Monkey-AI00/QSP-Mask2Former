"""New entrypoint for online grasp pipeline."""

from __future__ import annotations

from online_grasp.config.args import parse_args
from online_grasp.runtime.pipeline import OnlineGraspPipeline


def main():
    args = parse_args()
    # 保持 legacy 参数兼容行为
    if int(args.max_icp_stage2) == 80 and int(args.max_icp) != 80:
        args.max_icp_stage2 = int(args.max_icp)
    if str(args.source).strip():
        args.handle_target_sample_path = str(args.source).strip()
        if not bool(args.handle_target_sample):
            args.handle_target_sample = True

    pipeline = OnlineGraspPipeline(args)
    try:
        pipeline.loop()
    finally:
        pipeline.close()
        print("pipeline stopped")


if __name__ == "__main__":
    main()

