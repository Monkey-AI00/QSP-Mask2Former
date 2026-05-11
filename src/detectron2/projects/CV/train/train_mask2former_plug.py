#!/usr/bin/env python3
"""
Mask2Former 在 plug 数据集上的训练脚本（风格对齐 train_maskrcnn_plug.py / train_plug.py）

你需要准备：
1) 安装依赖：
   - Detectron2（建议源码安装）
   - Mask2Former（默认与 Detectron2 仓库同父目录下的 `Mask2Former/`；也可用环境变量 `MASK2FORMER_ROOT`）
   - 顶层工程目录可用 `WORKSPACE_ROOT` 指定（与默认数据集路径一致）
   - pip install -r ../Mask2Former/requirements.txt
2) 编译 MSDeformAttn CUDA 扩展（必须，否则无法训练）：
   cd ../Mask2Former/mask2former/modeling/pixel_decoder/ops
   sh make.sh
3) 选一个 Mask2Former instance 配置作为 base（推荐 R50 instance config）：
   src/detectron2/projects/PointRend/configs/InstanceSegmentation/mask2former_R50_plug.yaml
4) 初始化权重（强烈建议用 Mask2Former Model Zoo 的 COCO instance 预训练）：
   https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_R50_bs16_50ep/model_final_3c8ec9.pkl

说明：
- Mask2Former 使用 AdamW + (常见) MSDeformAttn pixel decoder，单卡需要自己调 batch/LR/分辨率。
- 本脚本会复用你现有的 register_plug_dataset（过滤 _background_），并自动把 NUM_CLASSES 同步为 thing_classes 的长度。
"""

from __future__ import annotations
import torch
import importlib.util
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

# 控制台降噪（与结果无关的 warning）
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch\.cuda\.amp\.(autocast|GradScaler).*deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch\.load.*weights_only=False.*",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*torch\.meshgrid.*indexing argument.*",
)

# 本脚本位于 projects/CV/train/，向上三级才是 Detectron2 仓库根（含 detectron2/ 与 projects/）。
_TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
_D2_REPO_ROOT = os.path.abspath(os.path.join(_TRAIN_DIR, "..", "..", ".."))
if _D2_REPO_ROOT not in sys.path:
    sys.path.insert(0, _D2_REPO_ROOT)


def _resolve_workspace_root() -> str:
    env = os.environ.get("WORKSPACE_ROOT", "").strip()
    if env:
        return os.path.abspath(env)
    p = Path(_D2_REPO_ROOT).resolve()
    parts = p.parts
    if len(parts) >= 2 and parts[-2] == "src" and parts[-1] == "detectron2":
        return str(p.parent.parent)
    if parts and parts[-1] == "detectron2":
        return str(p.parent)
    return str(p.parent)


def _resolve_mask2former_root() -> str:
    env = os.environ.get("MASK2FORMER_ROOT", "").strip()
    if env:
        return os.path.abspath(env)
    return os.path.abspath(os.path.join(_D2_REPO_ROOT, "..", "Mask2Former"))


_WORKSPACE_ROOT = _resolve_workspace_root()
_M2F_ROOT = _resolve_mask2former_root()
if _M2F_ROOT not in sys.path:
    sys.path.insert(0, _M2F_ROOT)

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.projects.deeplab import add_deeplab_config

# 复用你的数据集注册逻辑
from train_plug import register_plug_dataset


def _load_mask2former_train_net() -> object:
    """
    通过路径加载 Mask2Former 的 train_net.py，避免与 detectron2/projects/PointRend/train_net.py 名字冲突。
    """
    train_net_path = os.path.join(_M2F_ROOT, "train_net.py")
    if not os.path.isfile(train_net_path):
        raise FileNotFoundError(f"Mask2Former/train_net.py not found: {train_net_path}")
    spec = importlib.util.spec_from_file_location("mask2former_train_net", train_net_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load spec for mask2former_train_net")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


_m2f_train_net = _load_mask2former_train_net()
Mask2FormerTrainer = getattr(_m2f_train_net, "Trainer")
add_maskformer2_config = getattr(__import__("mask2former", fromlist=["add_maskformer2_config"]), "add_maskformer2_config")


def setup(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.defrost()
    # 数据集与类别数：由 register_plug_dataset 决定
    cfg.DATASETS.TRAIN = (str(args.train_dataset_name),)
    cfg.DATASETS.TEST = (str(args.val_dataset_name),)
    # Mask2Former instance config 用 SEM_SEG_HEAD.NUM_CLASSES 表示 thing 类别数（COCO=80）
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = int(args.num_classes)

    # 初始化权重
    if str(getattr(args, "weights", "")).strip():
        cfg.MODEL.WEIGHTS = str(args.weights).strip()

    # 输出目录
    if str(getattr(args, "output_dir", "")).strip():
        cfg.OUTPUT_DIR = str(args.output_dir).strip()
    else:
        cfg.OUTPUT_DIR = "./output/plug_mask2former"

    # 单卡/小数据集：worker=0 更稳
    cfg.DATALOADER.NUM_WORKERS = 0

    # 设备
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 训练/推理任务开关（确保是 instance seg）
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True

    cfg.freeze()
    default_setup(cfg, args)

    # 训练前打印关键信息
    print("=" * 70)
    print("[mask2former][setup] config_file:", str(args.config_file))
    print("[mask2former][setup] train_dataset_name:", str(args.train_dataset_name))
    print("[mask2former][setup] val_dataset_name:", str(args.val_dataset_name))
    print("[mask2former][setup] train_dataset_root:", str(args.train_dataset_root))
    print("[mask2former][setup] val_dataset_root:", str(args.val_dataset_root))
    print("[mask2former][setup] train_json_file:", str(args.train_json_file))
    print("[mask2former][setup] val_json_file:", str(args.val_json_file))
    print("[mask2former][setup] num_classes:", int(args.num_classes))
    print("[mask2former][setup] device:", str(cfg.MODEL.DEVICE))
    print("[mask2former][setup] backbone:", str(getattr(cfg.MODEL.BACKBONE, "NAME", "")))
    if str(getattr(cfg.MODEL.BACKBONE, "NAME", "")) == "D2SwinTransformer":
        print("[mask2former][setup] swin_embed_dim:", int(getattr(cfg.MODEL.SWIN, "EMBED_DIM", 0)))
        print("[mask2former][setup] swin_depths:", list(getattr(cfg.MODEL.SWIN, "DEPTHS", [])))
        print("[mask2former][setup] swin_num_heads:", list(getattr(cfg.MODEL.SWIN, "NUM_HEADS", [])))
    print("[mask2former][setup] output_dir:", str(cfg.OUTPUT_DIR))
    print("[mask2former][setup] ims_per_batch:", int(cfg.SOLVER.IMS_PER_BATCH))
    print("[mask2former][setup] base_lr:", float(cfg.SOLVER.BASE_LR))
    print("[mask2former][setup] max_iter:", int(cfg.SOLVER.MAX_ITER))
    print("[mask2former][setup] steps:", tuple(cfg.SOLVER.STEPS))
    print("[mask2former][setup] prior_on:", bool(getattr(cfg.MODEL.MASK_FORMER, "PRIOR_ON", False)))
    print("[mask2former][setup] prior_path:", str(getattr(cfg.MODEL.MASK_FORMER, "PRIOR_PATH", "")))
    print("[mask2former][setup] prior_alpha:", float(getattr(cfg.MODEL.MASK_FORMER, "PRIOR_ALPHA", 1.0)))
    print(
        "[mask2former][setup] prior_loss_weight:",
        float(getattr(cfg.MODEL.MASK_FORMER, "PRIOR_LOSS_WEIGHT", 0.0)),
    )
    print("=" * 70)
    return cfg


def main(args):
    # 注册训练集
    train_dataset_name, train_num_classes = register_plug_dataset(
        args.train_dataset_root,
        args.train_dataset_name,
        args.train_json_file,
    )
    args.train_dataset_name = train_dataset_name

    # 注册验证集；必须显式提供，不再回退到训练集
    val_dataset_root = str(getattr(args, "val_dataset_root", "")).strip()
    val_dataset_name = str(getattr(args, "val_dataset_name", "")).strip()
    val_json_file = str(getattr(args, "val_json_file", "")).strip()
    if not val_dataset_root or not val_dataset_name or not val_json_file:
        raise ValueError(
            "未找到完整的验证集参数：--val-dataset-root / --val-dataset-name / --val-json-file 必须全部显式提供；"
            "不再支持留空后自动回退到训练集。"
        )
    val_dataset_name, val_num_classes = register_plug_dataset(
        val_dataset_root,
        val_dataset_name,
        val_json_file,
    )
    args.val_dataset_root = val_dataset_root
    args.val_dataset_name = val_dataset_name
    args.val_json_file = val_json_file

    if int(train_num_classes) != int(val_num_classes):
        raise ValueError(
            f"Train/val num_classes mismatch: train={train_num_classes}, val={val_num_classes}"
        )
    args.num_classes = train_num_classes

    cfg = setup(args)

    if args.eval_only:
        model = Mask2FormerTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Mask2FormerTrainer.test(cfg, model)
        return res

    trainer = Mask2FormerTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--gpu-id", type=int, default=None, help="指定使用的 GPU ID")
    parser.add_argument(
        "--train-dataset-root",
        default=os.path.join(_WORKSPACE_ROOT, "datasets", "plug_train1"),
        help="训练集目录（包含图片与 COCO json）",
    )
    parser.add_argument("--train-dataset-name", default="plug_train1", help="训练集注册名称")
    parser.add_argument("--train-json-file", default="plug_train.json", help="训练集 COCO 标注文件名")
    parser.add_argument("--val-dataset-root", default="", help="验证集目录（必须显式提供）")
    parser.add_argument("--val-dataset-name", default="", help="验证集注册名称（必须显式提供）")
    parser.add_argument("--val-json-file", default="", help="验证集 COCO 标注文件名（必须显式提供）")
    # 向后兼容旧参数名：若提供则覆盖 train 参数
    parser.add_argument("--dataset-root", default="", help="(兼容) 等价于 --train-dataset-root")
    parser.add_argument("--dataset-name", default="", help="(兼容) 等价于 --train-dataset-name")
    parser.add_argument("--json-file", default="", help="(兼容) 等价于 --train-json-file")
    parser.add_argument("--weights", default="", help="初始化权重（建议用 Mask2Former COCO instance 预训练）")
    parser.add_argument("--output-dir", default="", help="训练输出目录")

    args = parser.parse_args()

    if str(getattr(args, "dataset_root", "")).strip():
        args.train_dataset_root = str(args.dataset_root).strip()
    if str(getattr(args, "dataset_name", "")).strip():
        args.train_dataset_name = str(args.dataset_name).strip()
    if str(getattr(args, "json_file", "")).strip():
        args.train_json_file = str(args.json_file).strip()

    # 常见失误：shell 里未 export DATASET_NAME，导致 ${DATASET_NAME}_train 变成字面量 _train
    _tn = str(getattr(args, "train_dataset_name", "")).strip()
    _vn = str(getattr(args, "val_dataset_name", "")).strip()
    if _tn == "_train" or _vn == "_val":
        raise ValueError(
            "训练/验证集名称异常（_train / _val），通常是未设置环境变量 DATASET_NAME，"
            "命令行里的 ${DATASET_NAME}_train 与 .../gangkou/${DATASET_NAME} 被展开为空。\n"
            "请先执行：export DATASET_NAME=<数据子目录名>  "
            "（例如 plug_train_merged_0422，对应 datasets/gangkou/plug_train_merged_0422/plug_train.json）。"
        )

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        print(f"✓ 已设置使用 GPU {args.gpu_id}")

    # 提醒用户检查 ops 编译产物（避免“训练没反应/直接报错”）
    ops_so = Path(_M2F_ROOT) / "mask2former" / "modeling" / "pixel_decoder" / "ops" / "MultiScaleDeformableAttention.cpython-39-x86_64-linux-gnu.so"
    if not ops_so.exists():
        print("⚠️  [mask2former] 未检测到 MSDeformAttn 编译产物：")
        print(f"    期望存在: {ops_so}")
        print("    请先执行：")
        print(f"      cd {Path(_M2F_ROOT) / 'mask2former' / 'modeling' / 'pixel_decoder' / 'ops'} && sh make.sh")

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


