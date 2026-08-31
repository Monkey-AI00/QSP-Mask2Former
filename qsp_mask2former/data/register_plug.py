import json
from pathlib import Path

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json


def load_plug_json(json_file, image_root, dataset_name):
    json_path = Path(json_file).expanduser().resolve()
    image_path = Path(image_root).expanduser().resolve()
    with json_path.open("r", encoding="utf-8") as handle:
        coco_data = json.load(handle)

    categories = sorted(coco_data["categories"], key=lambda item: item["id"])
    valid_categories = [
        item
        for item in categories
        if str(item.get("name", "")).strip() and item.get("name") != "_background_"
    ]
    category_mapping = {
        item["id"]: index for index, item in enumerate(valid_categories)
    }
    records = load_coco_json(str(json_path), str(image_path), dataset_name=None)
    filtered_records = []

    for record in records:
        filtered_annotations = []
        for annotation in record.get("annotations", []):
            category_id = annotation["category_id"]
            if category_id in category_mapping:
                mapped_annotation = dict(annotation)
                mapped_annotation["category_id"] = category_mapping[category_id]
                filtered_annotations.append(mapped_annotation)
        if filtered_annotations:
            mapped_record = dict(record)
            mapped_record["annotations"] = filtered_annotations
            filtered_records.append(mapped_record)

    metadata = MetadataCatalog.get(dataset_name)
    metadata.thing_classes = [item["name"] for item in valid_categories]
    metadata.thing_dataset_id_to_contiguous_id = category_mapping
    return filtered_records


def _validate_dataset_paths(dataset_root, json_filename):
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    json_path = root / str(json_filename)
    if not json_path.is_file():
        raise FileNotFoundError(f"COCO annotation file not found: {json_path}")
    return root, json_path


def _remove_catalog_entry(dataset_name):
    if dataset_name in DatasetCatalog.list():
        DatasetCatalog.remove(dataset_name)
    if dataset_name in MetadataCatalog.list():
        MetadataCatalog.remove(dataset_name)


def register_plug_dataset(dataset_root, dataset_name, json_filename):
    root, json_path = _validate_dataset_paths(dataset_root, json_filename)
    name = str(dataset_name).strip()
    if not name:
        raise ValueError("dataset_name must not be empty")

    _remove_catalog_entry(name)
    DatasetCatalog.register(
        name,
        lambda: load_plug_json(str(json_path), str(root), name),
    )
    MetadataCatalog.get(name).set(
        json_file=str(json_path),
        image_root=str(root),
        evaluator_type="coco",
    )
    records = DatasetCatalog.get(name)
    num_classes = len(MetadataCatalog.get(name).thing_classes)
    if num_classes < 1:
        _remove_catalog_entry(name)
        raise ValueError(f"No valid foreground categories found in {json_path}")
    if not records:
        _remove_catalog_entry(name)
        raise ValueError(f"No annotated foreground images found in {json_path}")
    return name, num_classes


def register_plug_train_val_datasets(
    train_dataset_root,
    val_dataset_root,
    train_dataset_name,
    val_dataset_name,
    train_json_file,
    val_json_file,
):
    train_name, train_classes = register_plug_dataset(
        train_dataset_root,
        train_dataset_name,
        train_json_file,
    )
    val_name, val_classes = register_plug_dataset(
        val_dataset_root,
        val_dataset_name,
        val_json_file,
    )
    if train_classes != val_classes:
        _remove_catalog_entry(train_name)
        _remove_catalog_entry(val_name)
        raise ValueError(
            f"Train/validation class-count mismatch: {train_classes} != {val_classes}"
        )
    return train_name, val_name, train_classes
