"""
Generates ears, extreme features, and special features parts of SEEK code
"""

from typing import Tuple
import numpy as np

from roboflow.model import infer

def get_ear_contour(elephant_image: np.ndarray) -> np.ndarray:
    """
    Generate the ear contour from an input elephant image.

    Args:
        elephant_image (np.ndarray): An image of one elephant; no 
            background removal required.

    Returns:
        A numpy array representing the ear contour.
    """
    predictions: list[dict] = infer(elephant_image, "elephant-re-id/ear-contour/1")
    for prediction in predictions:
        pass
    

def get_ears_code(elephant_image: np.ndarray) -> Tuple[str, str, str, str]:
    """
    Generate the ears, extreme features, and special features segments 
    of the SEEK code from an input elephant image. Extracts ear contours 
    and locates ear tears and holes by sector.

    Args:
        elephant_image (np.ndarray): An image of one elephant; no 
            background removal required.

    Returns:
        A 4-tuple of a 4-character string representing the left ear, a 
        4-character string representing the right ear, a 2-character 
        string representing presense / absense of extreme features, and 
        a 2-character string representing presense / absense of special 
        features on the ears of the elephant.
    """
    left_ear_code = "llll"
    right_ear_code = "rrrr"
    extreme_features_code = "xx"
    special_features_code = "ss"

    return left_ear_code, right_ear_code, extreme_features_code, special_features_code