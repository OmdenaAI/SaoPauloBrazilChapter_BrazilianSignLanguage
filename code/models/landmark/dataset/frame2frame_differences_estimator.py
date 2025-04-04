from typing import Union, Dict, List, Tuple, Iterable
from utils import check_mode, check_difference_type, check_landmark_type, load_config


def difference(
    p1, p2, mode: str = "3D", diff_type: str = "diff"
) -> Union[Tuple[float, float], Tuple[float, float, float]]:
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

    if mode == "3D":
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        dz = p1.z - p2.z
        if diff_type == "normalized_diff":
            return dx / 2.0, dy / 2.0, dz / 2.0  # range [-1, 1]
        else:
            return dx, dy, dz
    else:
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        if diff_type == "normalized_diff":
            return dx / 2.0, dy / 2.0
        else:
            return dx, dy


class DifferencesEstimator:
    """
    Estimates frame-to-frame movement vectors (differences) for pose and hand landmarks.
    """

    def __init__(
        self,
        hand_differences: Union[str, Dict],
        pose_differences: Union[str, Dict],
    ):
        """
        Parameters:
        ----------
        hand_differences : str or dict
            Path to YAML file or dictionary with hand landmark indices.
        pose_differences : str or dict
            Path to YAML file or dictionary with pose landmark indices.
        """

        self.hand_differences = load_config(hand_differences, "hand_differences")
        self.pose_differences = load_config(pose_differences, "pose_differences")

        self.hand_difference_names = list(self.hand_differences.keys())
        self.hand_difference_indices = list(self.hand_differences.values())

        self.pose_difference_names = list(self.pose_differences.keys())
        self.pose_difference_indices = list(self.pose_differences.values())

    def __compute_differences(
        self,
        landmark_indices: List[int],
        prev_landmarks: Iterable,
        next_landmarks: Iterable,
        mode: str,
        diff_type: str,
    ) -> List[Union[Tuple[float, float], Tuple[float, float, float]]]:
        return [
            difference(next_landmarks[idx], prev_landmarks[idx], mode, diff_type)
            for idx in landmark_indices
        ]

    def compute_differences(
        self,
        prev_landmarks: Iterable,
        next_landmarks: Iterable,
        mode: str,
        landmark_type: str,
        diff_type: str = "normalized_diff",
    ) -> List[Union[Tuple[float, float], Tuple[float, float, float]]]:
        """
        Compute raw or normalized difference vectors for specified landmark type.

        Parameters:
        ----------
        prev_landmarks : Iterable
            Landmarks from the previous frame.
        next_landmarks : Iterable
            Landmarks from the next frame.
        mode : str
        '2D' or '3D' — defines whether to use 2 or 3 dimensions for distance.
        landmark_type : str
            Either 'pose' or 'hand'.
        diff_type : str
            'diff' or 'normalized_diff'

        Returns:
        -------
        List of movement vectors between frames.
        """
        check_landmark_type(landmark_type)
        check_difference_type(diff_type)
        check_mode(mode)

        if landmark_type == "pose":
            return self.__compute_differences(
                self.pose_difference_indices,
                prev_landmarks,
                next_landmarks,
                mode,
                diff_type,
            )
        else:
            return self.__compute_differences(
                self.hand_difference_indices,
                prev_landmarks,
                next_landmarks,
                mode,
                diff_type,
            )

    def compute_annotated_differences(
        self,
        prev_landmarks: Iterable,
        next_landmarks: Iterable,
        mode: str,
        landmark_type: str,
        diff_type: str = "normalized_diff",
    ) -> Dict[str, Union[Tuple[float, float], Tuple[float, float, float]]]:
        """
        Compute named difference vectors for visualization or debugging.

        Parameters:
        ----------
        prev_landmarks : Iterable
            Previous frame's landmarks.
        next_landmarks : Iterable
            Next frame's landmarks.
        mode : str
        '2D' or '3D' — defines whether to use 2 or 3 dimensions for distance.
        landmark_type : str
            Either 'pose' or 'hand'.
        diff_type : str
            'diff' or 'normalized_diff'

        Returns:
        -------
        Dict[str, Tuple] : Mapping from name to movement vector.
        """
        check_landmark_type(landmark_type)
        check_difference_type(diff_type)

        if landmark_type == "pose":
            diffs = self.__compute_differences(
                self.pose_difference_indices,
                prev_landmarks,
                next_landmarks,
                mode,
                diff_type,
            )
            names = self.pose_difference_names
        else:
            diffs = self.__compute_differences(
                self.hand_difference_indices,
                prev_landmarks,
                next_landmarks,
                mode,
                diff_type,
            )
            names = self.hand_difference_names

        return dict(zip(names, diffs))
