"""
Generates a SEEK code for one elephant
"""

import numpy as np

from preprocess.background import remove_background
from roboflow.model import infer
from seek.gender import get_gender_code
from seek.age import get_age_code
from seek.tusks import get_tusk_code
from seek.ears import get_ears_code, get_ear_contour
from seek.view import get_view

def get_seek_code(elephant_image: np.ndarray) -> str:
    """
    Given an image of one elephant, output the corresponding SEEK code in the form:
    _ _ _ T _ _ E _ _ _ _ - _ _ _ _ X _ _ S _ _ _
    Following the form of https://elephantsalive.org/wp-content/uploads/2021/07/Bedetti-et-al-2020.pdf#page=7
    """
    ear_contour = get_ear_contour(elephant_image)
    no_background_image = remove_background(elephant_image)
    view = get_view(no_background_image)

    gender_code = get_gender_code(no_background_image)
    age_code = get_age_code(no_background_image)
    tusk_code = get_tusk_code(elephant_image, view)
    (
        left_ear_code,
        right_ear_code,
        extreme_features_code,
        special_features_ear_code,
    ) = get_ears_code(elephant_image)
    return f"{gender_code}{age_code}T{tusk_code}E{left_ear_code}-{right_ear_code}X{extreme_features_code}S{special_features_ear_code}_"
    