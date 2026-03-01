"""
Generates tusk part of SEEK code
"""

from typing import Literal
import numpy as np

from roboflow.model import infer

def get_tusk_code(elephant_image: np.ndarray, view: Literal["front", "left", "right", "back"]) -> str:
    """
    Generate the tusk segment of the SEEK code from an input elephant image.

    Args:
        elephant_image (np.ndarray): An image of one elephant; background not removed.
        view (Literal["front", "left", "right", "back"]): The view of the elephant.

    Returns:
        A two-character string of zeroes and ones representing the tusk portion of the SEEK code.
    """
    predictions: list[dict] = infer(elephant_image, "elephant-re-id/tusk-detection/1")

    if len(predictions) > 2:
        raise ValueError(f"Expected at most 2 tusk predictions, got {len(predictions)}")
    
    if len(predictions) == 0:
        return "00"
    
    if len(predictions) == 1:
        if view == "front":
            return "__"
        elif view == "left":
            return "01"
        elif view == "right":
            return "10"
        elif view == "back":
            return "__"
    
    if len(predictions) == 2:
        return "11"
    
    if len(predictions) > 2:
        print("Found more than 2 tusks")
        return "11"