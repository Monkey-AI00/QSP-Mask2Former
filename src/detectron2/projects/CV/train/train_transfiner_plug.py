#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在 plug 数据集上训练 Mask Transfiner（基于 transfiner 仓库自带的 detectron2 fork）。

为什么要单独脚本？
- transfiner 自带 detectron2 fork + 自定义 mask head/损失（见 transfiner/detectron2/modeling/roi_heads/mask_head.py）
- 直接用你主工程的 detectron2 会产生包冲突/权重不兼容
- 本脚本在 transfiner 代码树内运行，确保 import 的是 transfiner 的 detectron2

用法示例（单卡）：
  PYTHONPATH=/home/user/sjw/Yolo_pointrend/transfiner \\
  python -u /home/user/sjw/Yolo_pointrend/transfiner/tools/train_transfiner_plug.py \\
    --config-file /home/user/sjw/Yolo_pointrend/transfiner/configs/transfiner/mask_rcnn_R_50_FPN_3x.yaml \\
    --dataset-root /home/user/sjw/Yolo_pointrend/detectron2/plug_train1 \\
    --json-file plug_train.json \\
    --dataset-name plug_train1 \\
    --output-dir /home/user/sjw/Yolo_pointrend/transfiner/output/plug_transfiner_r50_3x \\
    --num-gpus 1

注意：覆写 config 的方式与 Detectron2 一致，是在命令末尾直接追加：
  KEY VALUE KEY VALUE ...
而不是使用 `--opts` flag。

多卡训练可参考 transfiner/scripts/*.sh，把入口脚本换成本文件即可。
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from collections import OrderedDict
from typing import Dict, List, Tuple
from pathlib import Path

import torch

# Reduce noisy deprecation/runtime warnings that do not affect training correctness.
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch\.cuda\.amp\.autocast\(args\.\.\.\) is deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch\.cuda\.amp\.GradScaler\(args\.\.\.\) is deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"detectron2\.engine\.train_loop",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*Detected call of `lr_scheduler\.step\(\)` before `optimizer\.step\(\)`.*",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*torch\.meshgrid: in an upcoming release.*",
)

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json, register_coco_instances
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOPanopticEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA


_TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
_D2_REPO_ROOT = os.path.abspath(os.path.join(_TRAIN_DIR, "..", "..", ".."))


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


_WORKSPACE_ROOT = _resolve_workspace_root()


def _read_coco_categories(json_path: str) -> List[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    return list(coco.get("categories", []))


def _is_valid_cat_name(name: object) -> bool:
    n = str(name or "").strip()
    return bool(n) and n != "_background_"


def register_plug_dataset_transfiner(dataset_root: str, dataset_name: str, json_file: str) -> Tuple[str, int]:
    """
    注册 plug 数据集（COCO 实例分割）：
    - 过滤 `_background_` 类别（若存在）
    - 将 category_id 映射为连续 id（从 0 开始）

    返回：(dataset_name, num_classes)
    """
    dataset_root = os.path.abspath(str(dataset_root))
    json_path = os.path.join(dataset_root, str(json_file))
    image_root = dataset_root

    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"COCO json not found: {json_path}")
    if not os.path.isdir(image_root):
        raise FileNotFoundError(f"image root not found: {image_root}")

    # remove existing registrations
    if dataset_name in DatasetCatalog.list():
        DatasetCatalog.remove(dataset_name)
    if dataset_name in MetadataCatalog.list():
        MetadataCatalog.remove(dataset_name)

    # build mapping from dataset category id -> contiguous id
    cats = sorted(_read_coco_categories(json_path), key=lambda x: int(x.get("id", 0)))
    valid_cats = [c for c in cats if _is_valid_cat_name(c.get("name"))]
    thing_classes = [str(c.get("name", "")).strip() for c in valid_cats]
    id_map: Dict[int, int] = {int(c["id"]): i for i, c in enumerate(valid_cats)}

    def loader():
        dicts = load_coco_json(json_path, image_root, dataset_name=None)
        out = []
        for d in dicts:
            anns = []
            for ann in d.get("annotations", []):
                cid = int(ann.get("category_id", -1))
                if cid in id_map:
                    ann["category_id"] = int(id_map[cid])
                    anns.append(ann)
            d["annotations"] = anns
            if anns:
                out.append(d)
        return out

    DatasetCatalog.register(dataset_name, loader)
    MetadataCatalog.get(dataset_name).set(
        json_file=json_path,
        image_root=image_root,
        evaluator_type="coco",
        thing_classes=thing_classes,
        thing_dataset_id_to_contiguous_id=id_map,
    )

    num_classes = len(thing_classes)
    print(f"✓ 数据集 {dataset_name} 注册成功: num_classes={num_classes}, classes={thing_classes}")
    return dataset_name, int(num_classes)


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    直接复用 transfiner/tools/train_net.py 的 evaluator 逻辑，保持一致。
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        assert torch.cuda.device_count() > comm.get_rank()
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        assert torch.cuda.device_count() > comm.get_rank()
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(f"no Evaluator for dataset {dataset_name} with type {evaluator_type}")
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA"))
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.defrost()
    cfg.DATASETS.TRAIN = (str(args.train_dataset_name),)
    cfg.DATASETS.TEST = (str(args.val_dataset_name),)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = int(args.num_classes)
    cfg.DATALOADER.NUM_WORKERS = 0

    if str(getattr(args, "weights", "")).strip():
        cfg.MODEL.WEIGHTS = str(args.weights).strip()
    if str(getattr(args, "output_dir", "")).strip():
        cfg.OUTPUT_DIR = str(args.output_dir).strip()

    # device auto
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    # 注册训练集
    train_dataset_name, train_num_classes = register_plug_dataset_transfiner(
        dataset_root=args.train_dataset_root,
        dataset_name=args.train_dataset_name,
        json_file=args.train_json_file,
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
    val_dataset_name, val_num_classes = register_plug_dataset_transfiner(
        dataset_root=val_dataset_root,
        dataset_name=val_dataset_name,
        json_file=val_json_file,
    )
    args.val_dataset_root = val_dataset_root
    args.val_dataset_name = val_dataset_name
    args.val_json_file = val_json_file

    if int(train_num_classes) != int(val_num_classes):
        raise ValueError(
            f"Train/val num_classes mismatch: train={train_num_classes}, val={val_num_classes}"
        )
    args.num_classes = int(train_num_classes)

    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks([hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))])
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
    parser.add_argument("--weights", default="", help="初始化权重（可选）")
    parser.add_argument("--output-dir", default="", help="输出目录（可选）")
    args = parser.parse_args()

    if str(getattr(args, "dataset_root", "")).strip():
        args.train_dataset_root = str(args.dataset_root).strip()
    if str(getattr(args, "dataset_name", "")).strip():
        args.train_dataset_name = str(args.dataset_name).strip()
    if str(getattr(args, "json_file", "")).strip():
        args.train_json_file = str(args.json_file).strip()

    _tn = str(getattr(args, "train_dataset_name", "")).strip()
    _vn = str(getattr(args, "val_dataset_name", "")).strip()
    if _tn == "_train" or _vn == "_val":
        raise ValueError(
            "训练/验证集名称异常（_train / _val），通常是未设置环境变量 DATASET_NAME，"
            "命令行里的 ${DATASET_NAME}_train 与 .../gangkou/${DATASET_NAME} 被展开为空。"
        )

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        print(f"✓ 已设置使用 GPU {args.gpu_id}")

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


