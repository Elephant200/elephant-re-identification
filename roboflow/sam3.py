"""
Module wrapping the SAM3 Roboflow workflow.
"""

import os
from typing import Literal

import numpy as np
from inference_sdk import InferenceHTTPClient
from tqdm import tqdm

BATCH_SIZE = 8

_client: InferenceHTTPClient | None = None


def _get_client() -> InferenceHTTPClient:
    global _client
    if _client is None:
        _client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=os.getenv("ROBOFLOW_API_KEY"),
        )
    return _client


def segment_image(
        image: np.ndarray | str,
        query: str,
        force: bool = False
    ):
    """Segment an image using the SAM3 Roboflow workflow."""
    response = _get_client().run_workflow(
        workspace_name="elephantidentificationresearch",
        workflow_id="sam3",
        images={"image": image},
        parameters={"query": query},
        use_cache=not force,
    )
    return response[0]["predictions"]


def segment_image_batch(
        images: list[np.ndarray | str],
        queries: list[str],
        force: bool = False
    ):
    """Segment a batch of images using the SAM3 Roboflow workflow.

    Images are sent in batches of 8 to match the workflow's slot layout.

    Args:
        images: List of images to segment (paths or arrays).
        queries: List of queries to use for segmentation.
        force: Whether to bypass the workflow cache.

    Returns:
        List of prediction dicts, one per input image.
    """
    if len(images) != len(queries):
        raise ValueError("Images and queries must have the same length")

    total = len(images)
    results = []

    client = _get_client()
    progress = tqdm(total=total, desc="SAM3", unit="img")
    for start in range(0, total, BATCH_SIZE):
        batch_images = images[start:start + BATCH_SIZE]
        batch_queries = queries[start:start + BATCH_SIZE]

        images_dict = {f"image{i}": img for i, img in enumerate(batch_images)}
        params_dict = {f"query{i}": q for i, q in enumerate(batch_queries)}

        response = client.run_workflow(
            workspace_name="elephantidentificationresearch",
            workflow_id="sam3-batch-8",
            images=images_dict,
            parameters=params_dict,
            use_cache=not force,
        )

        for key in sorted(response[0]):
            results.append(response[0][key])

        progress.update(len(batch_images))
    progress.close()

    assert len(results) == len(images)
    return results
