"""
Generates age part of SEEK code
"""

from datetime import datetime
import numpy as np

from roboflow.model import infer

def get_age_code(elephant_image: np.ndarray) -> str:
    """
    Generate the age segment of the SEEK code from an input elephant image. Estimates elephant age and inserts into birth year brackets.

    Args:
        elephant_image (np.ndarray): An image of one elephant; background should be removed.

    Returns:
        A two-character string representing the age portion of the SEEK code.
    """
    current_year = datetime.now().year
    return "ee"