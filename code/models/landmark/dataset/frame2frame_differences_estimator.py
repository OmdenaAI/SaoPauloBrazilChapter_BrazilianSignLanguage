import numpy as np
from typing import Union, Dict, List, Tuple, Iterable
from utils import check_mode, check_landmark_type, check_difference_type, load_config

def difference(p1, p2, mode: str = '3D', diff_type: str = 'diff') -> Union[Tuple[float, float], Tuple[float, float, float]]:
    """
    Computes the difference vector (movement) between two points.

    Parameters:
    ----------
    p1, p2 : objects
        Landmarks with attributes x, y, and optionally z.
    mode : str
        '2D' or '3D' — whether to include the z-dimension.
    diff_type : str
        - 'diff'            : raw delta vector (dx, dy, dz)
        - 'normalized_diff' : difference rescaled to [-1, 1] based on max possible shift (2.0)

    Returns:
    -------
    Tuple of float(s) representing the movement vector between p1 and p2.
    """
    check_mode(mode)
    check_difference_type(diff_type)

    if mode == '3D':
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        dz = p1.z - p2.z
        if diff_type == 'normalized_diff':
            return dx / 2.0, dy / 2.0, dz / 2.0  # range [-1, 1]
        else:
            return dx, dy, dz
    else:
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        if diff_type == 'normalized_diff':
            return dx / 2.0, dy / 2.0
        else:
            return dx, dy

class DifferencesEstimator:
    """
    Estimates movement vectors (differences) between the same landmarks in two frames.
    """

    def __init__(self,
                 difference_points: Union[str, Dict],
                 mode: str = '3D'):
        """
        Parameters:
        ----------
        differences_config : str or dict
            Path to YAML file or dictionary with named landmark indices.
        mode : str
            '2D' or '3D' — determines whether to compute (dx, dy) or (dx, dy, dz).
        """
        check_mode(mode)
        self.mode = mode

        self.difference_points = load_config(difference_points, "difference_points")

        # Cache keys and index list
        self.difference_names = list(self.difference_points.keys())
        self.difference_indices = list(self.difference_points.values())

    def __compute_differences(self,
                              prev_landmarks: Iterable,
                              next_landmarks: Iterable,
                              diff_type: str) -> List[Union[Tuple[float, float], Tuple[float, float, float]]]:
        return [
            difference(next_landmarks[idx], prev_landmarks[idx], self.mode, diff_type)
            for idx in self.difference_indices
        ]

    def compute_differences(self,
                            prev_landmarks: Iterable,
                            next_landmarks: Iterable,
                            diff_type: str = 'normalized_diff') -> List[Union[Tuple[float, float], Tuple[float, float, float]]]:
        """
        Compute raw or normalized movement vectors between frames.

        Parameters:
        ----------
        prev_landmarks : Iterable
            List of landmarks from the previous frame.
        next_landmarks : Iterable
            List of landmarks from the next frame.
        diff_type : str
            'diff' or 'normalized_diff'

        Returns:
        -------
        List of movement vectors.
        """
        return self.__compute_differences(prev_landmarks, next_landmarks, diff_type)

    def compute_annotated_differences(self,
                                      prev_landmarks: Iterable,
                                      next_landmarks: Iterable,
                                      diff_type: str = 'normalized_diff') -> Dict[str, Union[Tuple[float, float], Tuple[float, float, float]]]:
        """
        Compute named movement vectors between frames.

        Parameters:
        ----------
        prev_landmarks : Iterable
            Landmarks from the previous frame.
        next_landmarks : Iterable
            Landmarks from the next frame.
        diff_type : str
            'diff' or 'normalized_diff'

        Returns:
        -------
        Dict[str, Tuple] : Named mapping of point label to movement vector.
        """
        diffs = self.__compute_differences(prev_landmarks, next_landmarks, diff_type)
        return dict(zip(self.difference_names, diffs))