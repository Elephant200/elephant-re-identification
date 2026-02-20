"""
Module wrapping the SAM3 Roboflow workflow.
"""

import json
import os
from typing import Literal
from inference_sdk import InferenceHTTPClient
import numpy as np

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY"),
)

WORKSPACE_NAME = "elephant-re-id"
WORKFLOW_ID = "sam3"


def segment_image(
        image: np.ndarray | str,
        query: str,
        output_type: Literal["rle", "polygon"],
        force: bool = False
    ):
    """
    Segment an image using the SAM3 Roboflow workflow.

    """
    response = CLIENT.run_workflow(
        workspace_name=WORKSPACE_NAME,
        workflow_id=WORKFLOW_ID,
        images={"image": image},
        parameters={"query": query},
        use_cache=not force,
    )
    response = response[0]
    if response:
        return response["predictions"]

def segment_image_batch(
        images: list[np.ndarray | str],
        queries: list[str],
        output_type: Literal["rle", "polygon"] = "rle",
        force: bool = False
    ):
    """
    Segment a batch of images using the SAM3 Roboflow workflow.
    """
    if len(images) != len(queries):
        raise ValueError("Images and queries must have the same length")

    # Workflow uses batches of 8 images
    BATCH_SIZE = 8
    
    # Split images into batches
    batches = zip(
        [images[i:i+BATCH_SIZE] for i in range(0, len(images), BATCH_SIZE)], 
        [queries[i:i+BATCH_SIZE] for i in range(0, len(queries), BATCH_SIZE)],
    )

    results = []
    for images, queries in batches:
        response = CLIENT.run_workflow(
            workspace_name=WORKSPACE_NAME,
            workflow_id=WORKFLOW_ID,
            images={"image0": images[0], "image1": images[1], "image2": images[2], "image3": images[3], "image4": images[4], "image5": images[5], "image6": images[6], "image7": images[7]},
            parameters={"query0": queries[0], "query1": queries[1], "query2": queries[2], "query3": queries[3], "query4": queries[4], "query5": queries[5], "query6": queries[6], "query7": queries[7]},
            use_cache=not force,
        )
        response = response[0]
        if response:
            for r in response.values():
                results.append(r["predictions"])
    
    assert len(results) == len(images)

    return results
