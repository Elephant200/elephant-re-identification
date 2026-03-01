from typing import Literal
import numpy as np

def get_view(elephant_image: np.ndarray) -> Literal["front", "left", "right", "back"]:
    return "front"