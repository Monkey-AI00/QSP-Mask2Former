import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.projects.deeplab import add_deeplab_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPOSITORY_ROOT / "configs"

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from qsp_mask2former.data import register_plug_train_val_datasets


def _resolve_mask2former_root():
    configured = os.environ.get("MASK2FORMER_ROOT", "").strip()
    root = Path(configured).expanduser() if configured else REPOSITORY_ROOT / "third_party" / "Mask2Former"
    root = root.resolve()
    required = [root / "train_net.py", root / "mask2former" / "__init__.py"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "QSP-modified Mask2Former checkout is incomplete: " + ", ".join(missing)
        )
    return root


def _load_mask2former_components(mask2former_root):
    root_string = str(mask2former_root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    train_path = mask2former_root / "train_net.py"
    spec = importlib.util.spec_from_file_location("qsp_external_train_net", train_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Mask2Former trainer from {train_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mask2former_package = __import__("mask2former", fromlist=["add_maskformer2_config"])
    return module.Trainer, mask2former_package.add_maskformer2_config


def _materialize_config(config_file, mask2former_root, destination):
    source = Path(config_file).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Configuration file not found: {source}")
    try:
        relative = source.relative_to(CONFIG_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"Configuration must be located under {CONFIG_ROOT}: {source}") from error
    copied_root = Path(destination) / "configs"
    shutil.copytree(CONFIG_ROOT, copied_root)
    replacement = mask2former_root.as_posix()
    for yaml_path in copied_root.rglob("*.yaml"):
        content = yaml_path.read_text(encoding="utf-8")
        yaml_path.write_text(
            content.replace("${MASK2FORMER_ROOT}", replacement),
            encoding="utf-8",
        )
    return copied_root / relative


def _validate_optional_file(value, label):
    text = str(value).strip()
    if not text or "://" in text:
        return text
    path = Path(text).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return str(path)


def setup(args, trainer_class, add_maskformer2_config, mask2former_root):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    with tempfile.TemporaryDirectory(prefix="qsp_mask2former_config_") as temporary:
        materialized = _materialize_config(
            args.config_file,
            mask2former_root,
            temporary,
        )
        cfg.merge_from_file(str(materialized))
    cfg.merge_from_list(args.opts)
    cfg.defrost()
    cfg.DATASETS.TRAIN = (str(args.train_dataset_name),)
    cfg.DATASETS.TEST = (str(args.val_dataset_name),)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = int(args.num_classes)
    cfg.DATALOADER.NUM_WORKERS = int(args.num_workers)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True

    weights = _validate_optional_file(args.weights, "Initialization weights")
    if weights:
        cfg.MODEL.WEIGHTS = weights
    if str(args.output_dir).strip():
        cfg.OUTPUT_DIR = str(Path(args.output_dir).expanduser().resolve())

    prior_on = bool(getattr(cfg.MODEL.MASK_FORMER, "PRIOR_ON", False))
    prior_path = str(getattr(cfg.MODEL.MASK_FORMER, "PRIOR_PATH", "")).strip()
    if prior_on and not prior_path:
        raise ValueError(
            "MODEL.MASK_FORMER.PRIOR_PATH must be supplied when PRIOR_ON is True"
        )
    if prior_path:
        cfg.MODEL.MASK_FORMER.PRIOR_PATH = _validate_optional_file(
            prior_path,
            "Shape-prior file",
        )

    cfg.freeze()
    default_setup(cfg, args)
    return cfg, trainer_class


def main(args):
    train_name, val_name, num_classes = register_plug_train_val_datasets(
        train_dataset_root=args.train_dataset_root,
        val_dataset_root=args.val_dataset_root,
        train_dataset_name=args.train_dataset_name,
        val_dataset_name=args.val_dataset_name,
        train_json_file=args.train_json_file,
        val_json_file=args.val_json_file,
    )
    args.train_dataset_name = train_name
    args.val_dataset_name = val_name
    args.num_classes = num_classes

    mask2former_root = _resolve_mask2former_root()
    trainer_class, add_maskformer2_config = _load_mask2former_components(
        mask2former_root
    )
    cfg, trainer_class = setup(
        args,
        trainer_class,
        add_maskformer2_config,
        mask2former_root,
    )

    if args.eval_only:
        model = trainer_class.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS,
            resume=args.resume,
        )
        return trainer_class.test(cfg, model)

    trainer = trainer_class(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


def build_parser():
    parser = default_argument_parser()
    parser.add_argument("--train-dataset-root", required=True)
    parser.add_argument("--train-dataset-name", default="plug_train")
    parser.add_argument("--train-json-file", default="plug_train.json")
    parser.add_argument("--val-dataset-root", required=True)
    parser.add_argument("--val-dataset-name", default="plug_val")
    parser.add_argument("--val-json-file", default="plug_val.json")
    parser.add_argument("--weights", default="")
    parser.add_argument("--output-dir", default="outputs/qsp_mask2former")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gpu-id", type=int)
    return parser


def run():
    args = build_parser().parse_args()
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


if __name__ == "__main__":
    run()
