#!/usr/bin/env python3
"""
Mask R-CNN 在 plug 数据集上的训练脚本（对齐 train_plug.py 的使用习惯）

你需要提供：
- Mask R-CNN 的 config yaml（例如 detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml）
- 数据集目录（dataset_root，包含图片与 COCO json）
- （推荐）COCO 预训练权重作为初始化：detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl

说明：
- 这个脚本不会调用 add_pointrend_config，也不会设置 POINT_HEAD.*（Mask R-CNN 不需要）。
- 默认用标准 Detectron2 DatasetMapper；可用 --use-highlight-aug 复用你项目里的 HighlightMapper 做强光增强训练。
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Optional

# -------------------------------
# 控制台降噪：屏蔽与训练结果无关的 warning
# -------------------------------
# 1) PyTorch AMP 旧接口弃用提示（detectron2 旧版本内部仍在用 torch.cuda.amp.*）
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch\.cuda\.amp\.(autocast|GradScaler).*deprecated.*",
)
# 2) torch.meshgrid indexing 参数提示（不影响训练）
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*torch\.meshgrid.*indexing argument.*",
)
# 3) lr_scheduler.step/optimizer.step 顺序提示（detectron2 旧实现的提示，不影响收敛）
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"Detected call of `lr_scheduler\.step\(\)` before `optimizer\.step\(\)`\..*",
)

# 添加 detectron2 到路径（必须指向 detectron2 仓库根目录）
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.utils.file_io import PathManager
import detectron2.data.transforms as T

# 复用你的数据集注册逻辑（过滤 _background_）
from train_plug import register_plug_dataset

# 可选：复用强光增强 mapper（如果你希望 Mask R-CNN 也在强光增强数据上训练）
try:  # pragma: no cover
    from highlight_mapper import HighlightMapper
except Exception:  # pragma: no cover
    HighlightMapper = None  # type: ignore


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        # 保持和 train_plug.py 一致的 Resize/Flip 基础增强（其余由 HighlightMapper 决定）
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            ),
            T.RandomFlip(horizontal=True),
        ]
        # cfg.INPUT 是 CfgNode，不是 dict；用 getattr 读取更稳
        use_hl = bool(getattr(cfg.INPUT, "USE_HIGHLIGHT_MAPPER", False))
        if use_hl and HighlightMapper is not None:
            mapper = HighlightMapper(cfg, is_train=True, augmentations=augs)
        else:
            mapper = DatasetMapper(cfg, is_train=True, augmentations=augs)
        return build_detection_train_loader(cfg, mapper=mapper)


def _read_last_checkpoint_path(output_dir: str) -> Optional[str]:
    try:
        p = os.path.join(str(output_dir), "last_checkpoint")
        if not os.path.exists(p):
            return None
        with open(p, "r") as f:
            name = f.read().strip()
        if not name:
            return None
        if os.path.isabs(name):
            return name
        return os.path.abspath(os.path.join(str(output_dir), name))
    except Exception:
        return None


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.defrost()
    # 由注册数据集后填入
    cfg.DATASETS.TRAIN = (str(args.dataset_name),)
    cfg.DATASETS.TEST = (str(args.dataset_name),)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = int(args.num_classes)

    # 初始化权重（推荐用 COCO 预训练 Mask R-CNN）
    if str(getattr(args, "weights", "")).strip():
        cfg.MODEL.WEIGHTS = str(args.weights).strip()

    # 输出目录
    if str(getattr(args, "output_dir", "")).strip():
        cfg.OUTPUT_DIR = str(args.output_dir).strip()
    else:
        cfg.OUTPUT_DIR = "./output/plug_maskrcnn"

    # 设备
    import torch

    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 是否使用 HighlightMapper（写到 cfg.INPUT 下，方便 Trainer.build_train_loader 读取）
    cfg.INPUT.USE_HIGHLIGHT_MAPPER = bool(getattr(args, "use_highlight_aug", False))

    # 训练“看起来没反应”时，很多时候是 dataloader worker 在起进程/卡住或占用资源；
    # 小数据集 + 单卡时用 0 最稳（也便于定位问题）。
    cfg.DATALOADER.NUM_WORKERS = int(getattr(cfg.DATALOADER, "NUM_WORKERS", 0))
    if cfg.DATALOADER.NUM_WORKERS > 0:
        cfg.DATALOADER.NUM_WORKERS = 0

    # --- logs (early) ---
    try:
        print("=" * 70)
        print("[maskrcnn][setup] config_file:", str(args.config_file))
        print("[maskrcnn][setup] dataset_name:", str(args.dataset_name))
        print("[maskrcnn][setup] num_classes:", int(args.num_classes))
        print("[maskrcnn][setup] device:", str(cfg.MODEL.DEVICE))
        print("[maskrcnn][setup] output_dir:", str(cfg.OUTPUT_DIR))
        print("[maskrcnn][setup] highlight_aug:", bool(cfg.INPUT.USE_HIGHLIGHT_MAPPER))
        print("[maskrcnn][setup] num_workers:", int(cfg.DATALOADER.NUM_WORKERS))
        print("[maskrcnn][setup] ims_per_batch:", int(cfg.SOLVER.IMS_PER_BATCH))
        print("[maskrcnn][setup] base_lr:", float(cfg.SOLVER.BASE_LR))
        print("[maskrcnn][setup] max_iter:", int(cfg.SOLVER.MAX_ITER))
        print("[maskrcnn][setup] steps:", tuple(cfg.SOLVER.STEPS))
        print("=" * 70)
    except Exception:
        pass

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    print("[maskrcnn] starting training entry...")
    dataset_name, num_classes = register_plug_dataset(args.dataset_root, args.dataset_name, args.json_file)
    args.dataset_name = dataset_name
    args.num_classes = num_classes

    # 解析/下载权重（如果是 detectron2://... 会自动下载到本机缓存）
    if str(getattr(args, "weights", "")).strip():
        try:
            w = str(args.weights).strip()
            local_w = PathManager.get_local_path(w)
            print(f"[maskrcnn][weights] {w}")
            print(f"[maskrcnn][weights][local] {local_w}")
        except Exception as e:
            print(f"[maskrcnn][weights] failed to resolve weights: {e}")

    cfg = setup(args)

    if args.eval_only:
        from detectron2.checkpoint import DetectionCheckpointer

        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)

    # 结构变化不大，但保留与 train_plug.py 一致的“resume 失败回退”逻辑
    from detectron2.checkpoint import DetectionCheckpointer

    if args.resume:
        try:
            trainer.resume_or_load(resume=True)
        except Exception as e:
            print("⚠️  resume 失败，退化为只加载模型权重并从 iter=0 开始。")
            print(f"    错误信息: {e}")
            ckpt = _read_last_checkpoint_path(cfg.OUTPUT_DIR) or cfg.MODEL.WEIGHTS
            trainer = Trainer(cfg)
            DetectionCheckpointer(trainer.model).load(ckpt)
            trainer.start_iter = 0
    else:
        if str(cfg.MODEL.WEIGHTS).strip():
            DetectionCheckpointer(trainer.model).load(cfg.MODEL.WEIGHTS)

    try:
        return trainer.train()
    except RuntimeError as e:
        # 你遇到的“训练没反应”典型根因其实是 OOM 在 iter=0 直接爆掉；
        # 这里补充更直观的提示，避免只看到 conda run failed。
        msg = str(e)
        if "CUDA out of memory" in msg or "OutOfMemoryError" in msg:
            print("\n" + "=" * 70)
            print("❌ [maskrcnn][OOM] CUDA 显存不足，训练在 iter=0 退出。")
            print("建议：")
            print("  - 将 batch 降到 1：SOLVER.IMS_PER_BATCH 1")
            print("  - 开 AMP：SOLVER.AMP.ENABLED True")
            print("  - 仍 OOM 则降低输入分辨率：INPUT.MIN_SIZE_TRAIN / INPUT.MAX_SIZE_TRAIN")
            print("  - 如显存出现“幽灵占用”，用 nvidia-smi 找 PID 后 kill -9 <pid> 或重启释放。")
            print("=" * 70 + "\n")
        raise


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--gpu-id", type=int, default=None, help="指定使用的 GPU ID")
    parser.add_argument(
        "--dataset-root",
        default="/home/user/sjw/Yolo_pointrend/detectron2/plug_train1",
        help="数据集目录（包含图片与 COCO json）",
    )
    parser.add_argument("--dataset-name", default="plug_train1", help="注册到 detectron2 的数据集名称")
    parser.add_argument("--json-file", default="plug_train.json", help="COCO 标注文件名（相对 dataset-root）")
    parser.add_argument(
        "--weights",
        default="",
        help="初始化权重（推荐 COCO 预训练 Mask R-CNN；也可留空从 yaml 的 MODEL.WEIGHTS 读取）",
    )
    parser.add_argument("--output-dir", default="", help="训练输出目录（留空则默认 ./output/plug_maskrcnn）")
    parser.add_argument("--use-highlight-aug", action="store_true", help="训练时使用 HighlightMapper 做强光模拟增强")

    args = parser.parse_args()

    print("=" * 70)
    print("[maskrcnn] command line args:", args)
    print("=" * 70)

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        print(f"✓ 已设置使用 GPU {args.gpu_id}")

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


