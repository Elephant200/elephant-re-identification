"""
Module to create distilled datasets from SAM3 predictions.
"""

import json
import os
import shutil
from typing import Literal

import cv2
import numpy as np
from pycocotools import mask as mask_util

from roboflow.sam3 import segment_image_batch

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def _collect_image_paths(data_path: str) -> list[str]:
    """Return sorted list of image file paths under data_path."""
    paths = []
    for root, _, files in os.walk(data_path):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                paths.append(os.path.join(root, f))
    return paths


def _rle_to_polygons(rle: dict) -> list[list[float]]:
    """Convert an RLE mask to a list of polygon coordinate arrays."""
    binary = mask_util.decode(rle)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) >= 3:
            polygons.append(contour.flatten().tolist())
    return polygons


def _build_annotation(
    annotation_id: int,
    image_id: int,
    rle: dict,
    mode: Literal["bbox", "mask", "polygon"],
) -> dict:
    """Build a single COCO annotation dict from an RLE prediction."""
    ann = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,
        "bbox": mask_util.toBbox(rle).tolist(),
        "area": float(mask_util.area(rle)),
        "iscrowd": 0,
    }
    if mode == "mask":
        ann["segmentation"] = rle
    elif mode == "polygon":
        ann["segmentation"] = _rle_to_polygons(rle)
    return ann


def distill_dataset(
    data_path: str,
    output_path: str,
    query: str,
    mode: Literal["bbox", "mask", "polygon"] = "mask",
):
    """Run SAM3 on all images in data_path and save a COCO dataset to output_path.

    Copies images into output_path preserving subdirectory structure, and
    writes an annotations.json in COCO format.

    Args:
        data_path: Root directory containing source images.
        output_path: Destination directory for the distilled dataset.
        query: Text prompt to use for SAM3 segmentation.
        mode: Annotation mode. "bbox" saves bounding boxes only, "mask"
            adds RLE segmentation masks, "polygon" adds polygon segmentation.
    """
    image_paths = _collect_image_paths(data_path)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {data_path}")

    queries = [query] * len(image_paths)
    print(f"Running SAM3 on {len(image_paths)} images...")
    all_predictions = segment_image_batch(image_paths, queries)

    coco_images = []
    coco_annotations = []
    annotation_id = 1

    for image_id, (image_path, predictions) in enumerate(
        zip(image_paths, all_predictions), start=1
    ):
        rel_path = os.path.relpath(image_path, data_path)
        dest_path = os.path.join(output_path, rel_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(image_path, dest_path)

        img = cv2.imread(image_path)
        h, w = img.shape[:2]

        coco_images.append({
            "id": image_id,
            "file_name": rel_path,
            "width": w,
            "height": h,
        })

        for pred in predictions:
            ann = _build_annotation(annotation_id, image_id, pred["rle_mask"], mode)
            coco_annotations.append(ann)
            annotation_id += 1

    coco = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [{"id": 1, "name": query}],
    }

    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "annotations.json"), "w") as f:
        json.dump(coco, f)

    print(f"Saved {len(coco_annotations)} annotations for {len(coco_images)} images to {output_path}")