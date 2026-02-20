import os

import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
from pycocotools import mask as mask_util

BG_CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY"),
)


def smooth_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Keep only the largest connected component and close small holes via morphological closing.

    Args:
        mask: Binary mask as a numpy array.
        kernel_size: Size of the kernel for morphological closing.

    Returns:
        The smoothed mask.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    cleaned = (labels == largest).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)


def remove_background(image: np.ndarray) -> np.ndarray:
    """Remove the background from an elephant image using a SAM3 Roboflow workflow.

    Segments the elephant body and tusks, merges overlapping tusk masks into the
    body mask, then applies the combined mask to isolate the foreground.

    Args:
        image: BGR image as a numpy array.

    Returns:
        The image with background pixels zeroed out.
    """
    response = BG_CLIENT.run_workflow(
        workspace_name="elephantidentificationresearch",
        workflow_id="sam3-with-prompts",
        images={"image": image},
        use_cache=True,
    )

    if not response:
        return image

    body_preds = response[0].get("sam_body", {}).get("predictions", [])
    tusk_preds = response[0].get("sam_tusk", {}).get("predictions", [])

    body_rles = [p["rle_mask"] for p in body_preds if "rle_mask" in p]
    tusk_rles = [p["rle_mask"] for p in tusk_preds if "rle_mask" in p]

    if not body_rles:
        return image

    body_merged = mask_util.merge(body_rles, intersect=0)

    if tusk_rles:
        tusk_merged = mask_util.merge(tusk_rles, intersect=0)
        if mask_util.iou([body_merged], [tusk_merged], [0])[0, 0] > 0:
            body_merged = mask_util.merge([body_merged, tusk_merged], intersect=0)

    mask = smooth_mask(mask_util.decode(body_merged))
    return cv2.bitwise_and(image, image, mask=mask)