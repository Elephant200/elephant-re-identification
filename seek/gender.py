"""
Generates gender part of SEEK code
"""

from typing import Literal
import numpy as np

from roboflow.model import infer

def get_gender_code(elephant_image: np.ndarray) -> Literal["B", "C"]:
    """
    Generate the gender segment of the SEEK code from an input elephant image.

    Args:
        elephant_image (np.ndarray): An image of one elephant; background should be removed.

    Returns:
        A one-character string representing the gender portion of the SEEK code.
            "B" for bull (male), "C" for cow (female)
    """
    return "g"