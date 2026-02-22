"""
Auto-annotate images using pretrained models and output COCO-format JSON.

Supports two model backends:
  - "sam3": prompt-based segmentation via the SAM3 Roboflow workflow.
    Each category name is used directly as the SAM3 query prompt.
  - Any other model string (e.g. "model_name/5"): treated as a Roboflow-
    hosted model ID, accessed via the InferenceHTTPClient.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import cv2
from pycocotools import mask as mask_util
from tqdm import tqdm

from roboflow.sam3 import segment_image
from roboflow.model import infer

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ── Helpers ──────────────────────────────────────────────────────────

def _scan_images(image_dir: str) -> list[str]:
    """Return a sorted list of image file paths found in *image_dir*."""
    directory = Path(image_dir)
    paths = [
        str(p) for p in sorted(directory.iterdir())
        if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
    ]
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return paths


def _center_to_topleft(x: float, y: float, w: float, h: float) -> list[float]:
    """Convert a center-format bounding box to COCO top-left [x, y, w, h]."""
    return [x - w / 2, y - h / 2, w, h]


def _make_image_entry(image_path: str, image_id: int) -> dict | None:
    """Build a COCO ``images`` entry by reading the file for its dimensions."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    return {
        "id": image_id,
        "file_name": os.path.basename(image_path),
        "width": w,
        "height": h,
    }


def _rle_to_json_safe(rle: dict) -> dict:
    """Ensure an RLE dict is JSON-serializable (decode bytes counts)."""
    rle = dict(rle)
    if isinstance(rle.get("counts"), bytes):
        rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def _polygon_to_rle(points: list[dict], height: int, width: int) -> dict:
    """Convert Roboflow polygon points to a COCO compressed-RLE mask.

    Args:
        points: List of {"x": float, "y": float} dicts from the prediction.
        height: Image height in pixels.
        width: Image width in pixels.
    """
    # Flatten polygon points into [x1, y1, x2, y2, ...] for pycocotools
    polygon = []
    for pt in points:
        polygon.extend([pt["x"], pt["y"]])

    # Encode via pycocotools: polygon -> binary mask -> compressed RLE
    rle_list = mask_util.frPyObjects([polygon], height, width)
    rle = mask_util.merge(rle_list)
    return _rle_to_json_safe(rle)


# ── SAM3 backend ─────────────────────────────────────────────────────

def _annotate_with_sam3(
    image_paths: list[str],
    categories: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Run SAM3 on every image for each category and collect COCO entries.

    SAM3 uses the category name directly as its text query prompt.
    Predictions always include RLE segmentation masks.
    """
    category_name_to_id = {c["name"]: c["id"] for c in categories}
    images_list: list[dict] = []
    annotations: list[dict] = []
    annotation_id = 1
    errors = 0

    progress = tqdm(image_paths, desc="SAM3", unit="img")
    for image_index, image_path in enumerate(progress):
        image_id = image_index + 1
        image_entry_added = False
        image_annotation_count = 0

        try:
            # Query SAM3 once per category (category name = query prompt)
            for category_name, category_id in category_name_to_id.items():
                response = segment_image(
                    image_path, category_name, output_type="rle",
                )
                # segment_image returns {"image": {...}, "predictions": [...]}
                predictions = response.get("predictions", [])

                # Build image entry from the API response dimensions
                # (avoids reading the file with cv2 just for width/height)
                if not image_entry_added and "image" in response:
                    images_list.append({
                        "id": image_id,
                        "file_name": os.path.basename(image_path),
                        "width": response["image"]["width"],
                        "height": response["image"]["height"],
                    })
                    image_entry_added = True

                for prediction in predictions:
                    has_rle = "rle_mask" in prediction
                    has_bbox = all(
                        k in prediction
                        for k in ("x", "y", "width", "height")
                    )
                    if not has_rle and not has_bbox:
                        continue

                    # Use the bbox from the prediction when available;
                    # only fall back to deriving it from the RLE mask
                    if has_bbox:
                        bbox = _center_to_topleft(
                            prediction["x"], prediction["y"],
                            prediction["width"], prediction["height"],
                        )
                        area = prediction["width"] * prediction["height"]
                    else:
                        rle = prediction["rle_mask"]
                        bbox = mask_util.toBbox(rle).tolist()
                        area = float(mask_util.area(rle))

                    annotation: dict = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": 0,
                    }

                    if has_rle:
                        annotation["segmentation"] = _rle_to_json_safe(
                            prediction["rle_mask"],
                        )

                    annotations.append(annotation)
                    annotation_id += 1
                    image_annotation_count += 1

            # Fallback: read dimensions from file if the API didn't provide them
            if not image_entry_added:
                image_entry = _make_image_entry(image_path, image_id)
                if image_entry is not None:
                    images_list.append(image_entry)

        except Exception:
            errors += 1
            logger.exception("Failed to annotate %s", os.path.basename(image_path))

        progress.set_postfix(annotations=len(annotations), errors=errors)

    if errors:
        logger.warning("Skipped %d images due to errors", errors)

    return images_list, annotations


# ── Roboflow model backend ───────────────────────────────────────────

def _annotate_with_roboflow_model(
    image_paths: list[str],
    model_id: str,
    categories: list[dict],
    confidence: float,
    iou: float,
) -> tuple[list[dict], list[dict]]:
    """Run a Roboflow-hosted model on every image and collect COCO entries.

    Predictions are filtered to only include classes listed in *categories*.
    If the model returns polygon segmentation data, it is converted to RLE.
    """
    category_name_to_id = {c["name"]: c["id"] for c in categories}
    images_list: list[dict] = []
    annotations: list[dict] = []
    annotation_id = 1
    errors = 0

    progress = tqdm(image_paths, desc=model_id, unit="img")
    for image_index, image_path in enumerate(progress):
        image_id = image_index + 1

        try:
            image_entry = _make_image_entry(image_path, image_id)
            if image_entry is None:
                logger.warning("Could not read %s, skipping", os.path.basename(image_path))
                continue
            images_list.append(image_entry)
            height = image_entry["height"]
            width = image_entry["width"]

            predictions = infer(
                image_path, model_id, confidence=confidence, iou=iou,
            )

            for prediction in predictions:
                # Match prediction class to a requested category
                class_name = prediction.get("class", prediction.get("class_name"))
                if class_name not in category_name_to_id:
                    continue

                # Every prediction must have a bounding box
                if not all(
                    k in prediction for k in ("x", "y", "width", "height")
                ):
                    continue

                bbox = _center_to_topleft(
                    prediction["x"], prediction["y"],
                    prediction["width"], prediction["height"],
                )

                # Area comes directly from the prediction's bbox dimensions
                annotation: dict = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_name_to_id[class_name],
                    "bbox": bbox,
                    "area": prediction["width"] * prediction["height"],
                    "iscrowd": 0,
                }

                # Convert polygon segmentation to RLE when available
                if "points" in prediction:
                    annotation["segmentation"] = _polygon_to_rle(
                        prediction["points"], height, width,
                    )

                annotations.append(annotation)
                annotation_id += 1

        except Exception:
            errors += 1
            logger.exception("Failed to annotate %s", os.path.basename(image_path))

        progress.set_postfix(annotations=len(annotations), errors=errors)

    if errors:
        logger.warning("Skipped %d images due to errors", errors)

    return images_list, annotations


# ── Public API ───────────────────────────────────────────────────────

def distill(
    image_dir: str,
    output_path: str,
    model: str,
    classes: list[str],
    *,
    confidence: float = 0.5,
    iou: float = 0.5,
) -> None:
    """Auto-annotate a directory of images and write COCO-format JSON.

    Args:
        image_dir: Directory containing images to annotate.
        output_path: Where to write the COCO JSON file.
        model: Model to use. Pass ``"sam3"`` for the SAM3 segmentation
            workflow, or a Roboflow model ID (e.g. ``"model_name/5"``)
            for a hosted model.
        classes: Category names for the output dataset. For SAM3, each
            name doubles as the query prompt sent to the model.
        confidence: (Roboflow models only) Minimum confidence threshold.
        iou: (Roboflow models only) IoU threshold for NMS.
    """
    image_paths = _scan_images(image_dir)
    logger.info("Found %d images in %s", len(image_paths), image_dir)

    # Build COCO categories list
    categories = [
        {"id": i + 1, "name": name, "supercategory": ""}
        for i, name in enumerate(classes)
    ]

    # Dispatch to the appropriate backend
    if model == "sam3":
        images_list, annotations = _annotate_with_sam3(
            image_paths, categories,
        )
    else:
        images_list, annotations = _annotate_with_roboflow_model(
            image_paths, model, categories, confidence, iou,
        )

    # Assemble the full COCO dataset structure
    coco = {
        "info": {
            "description": "Auto-annotated dataset",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [],
        "images": images_list,
        "annotations": annotations,
        "categories": categories,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)

    logger.info(
        "Wrote %d annotations across %d images to %s",
        len(annotations), len(images_list), output_path,
    )


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Auto-annotate images using a pretrained model.",
    )
    parser.add_argument(
        "image_dir",
        help="Directory containing images to annotate.",
    )
    parser.add_argument(
        "model",
        help='Model to use: "sam3" or a Roboflow model ID (e.g. "model_name/5").',
    )
    parser.add_argument(
        "classes",
        nargs="+",
        help="Category names for the output dataset.",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5,
        help="(Roboflow models only) Minimum confidence threshold (default: 0.5).",
    )
    parser.add_argument(
        "--iou", type=float, default=0.5,
        help="(Roboflow models only) IoU threshold for NMS (default: 0.5).",
    )

    args = parser.parse_args()
    output_path = os.path.join(args.image_dir, "annotations.coco.json")

    distill(
        image_dir=args.image_dir,
        output_path=output_path,
        model=args.model,
        classes=args.classes,
        confidence=args.confidence,
        iou=args.iou,
    )
