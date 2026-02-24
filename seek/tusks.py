"""
Generates tusk part of SEEK code
"""

import numpy as np

from roboflow.model import infer

def get_tusk_code(elephant_image: np.ndarray) -> str:
    """
    Generate the tusk segment of the SEEK code from an input elephant image.

    Args:
        elephant_image (np.ndarray): An image of one elephant; background not removed.

    Returns:
        A two-character string of zeroes and ones representing the gender portion of the SEEK code.
    """
    predictions = infer(elephant_image, "elephant-re-id/tusks")

    return "tt"