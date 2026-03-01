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
        A two-character string of zeroes and ones representing the tusk portion of the SEEK code.
    """
    predictions: list[dict] = infer(elephant_image, "elephant-re-id/tusk-detection/1")

    if len(predictions) > 2:
        raise ValueError(f"Expected at most 2 tusk predictions, got {len(predictions)}")
    
    match len(predictions):
        case 0: # No tusks
            return "00"
        case 1: # Only one tusk is present
            # TODO: Determine which tusk is left or right
            return "__" # not sure which tusk is left or right
        case 2: # Both tusks are present
            return "11"