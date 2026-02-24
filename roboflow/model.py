"""
Module wrapping fine-tuned Roboflow-hosted models through InferenceHTTPClient.
"""

import os

import numpy as np
from inference_sdk import InferenceHTTPClient, InferenceConfiguration


_client_cache: dict[tuple[float, float], InferenceHTTPClient] = {}


def _get_client(
    confidence: float = 0.5,
    iou: float = 0.5,
) -> InferenceHTTPClient:
    key = (confidence, iou)
    if key not in _client_cache:
        client = InferenceHTTPClient(
            api_url="https://outline.roboflow.com",
            api_key=os.getenv("ROBOFLOW_API_KEY"),
        )
        client.configure(InferenceConfiguration(
            confidence_threshold=confidence,
            iou_threshold=iou,
        ))
        _client_cache[key] = client
    return _client_cache[key]


def infer(
    image: np.ndarray | str,
    model_id: str,
    confidence: float = 0.5,
    iou: float = 0.5,
) -> list[dict]:
    """Run inference on a single image.

    Args:
        image: Path to an image file or a numpy array.
        model_id: Roboflow model identifier (e.g. "model_name/5").
        confidence: Minimum confidence threshold.
        iou: IoU threshold for NMS.

    Returns:
        List of prediction dicts from the model.
    """
    client = _get_client(confidence, iou)
    response = client.infer(image, model_id=model_id)
    return response["predictions"]


def infer_batch(
    images: list[np.ndarray | str],
    model_id: str,
    confidence: float = 0.5,
    iou: float = 0.5,
) -> list[list[dict]]:
    """Run inference on a list of images.

    Args:
        images: List of image paths or numpy arrays.
        model_id: Roboflow model identifier (e.g. "model_name/5").
        confidence: Minimum confidence threshold.
        iou: IoU threshold for NMS.

    Returns:
        List of prediction lists, one per input image.
    """
    client = _get_client(confidence, iou)
    results = []
    for image in images:
        response = client.infer(image, model_id=model_id)
        results.append(response["predictions"])
    return results
