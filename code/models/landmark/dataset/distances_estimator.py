import numpy as np
from typing import Union, Dict, List, Tuple, Iterable
from utils import check_mode, check_landmark_type, check_distance_type, load_config



def distance(p1, p2, mode: str = "3D", distance_type: str = "shifted_dist") -> float:
    """
    Computes distance between two points with support for normalized and shifted variants.

    Parameters:
    ----------
    p1, p2 : objects
        Points with attributes x, y, and optionally z (for 3D).
    mode : str
        '2D' or '3D' — defines whether to use 2 or 3 dimensions for distance.
    distance_type : str
        One of:
        - 'dist'            : Raw Euclidean distance
        - 'normalized_dist' : Distance scaled to [0, 1] based on max possible distance (√12)
        - 'shifted_dist'    : Normalized then shifted to [-1, 1] range

    Returns:
    -------
    float : Distance in the selected representation
    """
    check_mode(mode)
    check_distance_type(distance_type)

    if mode == "3D":
        dist = np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)
        max_dist = np.sqrt(12)  # max distance in 3D normalized space [-1,1]
    else:
        dist = np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)
        max_dist = np.sqrt(8)  # max distance in 2D normalized space [-1,1]

    if distance_type == "dist":
        return dist
    elif distance_type == "normalized_dist":
        return dist / max_dist
    elif distance_type == "shifted_dist":
        return (dist / max_dist) * 2 - 1  # scale to [-1, 1]


class DistancesEstimator:
    """
    Estimates distances between pairs of landmarks, with support for normalized and shifted distance types.
    """

    def __init__(
        self,
        hand_distances: Union[str, Dict],
        pose_distances: Union[str, Dict],
        mode: str = "3D",
    ):
        """
        Parameters:
        ----------
        hand_distances : str or dict
            Path to YAML file or dictionary with hand landmark index pairs.
        pose_distances : str or dict
            Path to YAML file or dictionary with pose landmark index pairs.
        mode : str
            '2D' or '3D' — determines how distances are computed.
        """
        check_mode(mode)
        self.mode = mode

        self.hand_distances = load_config(hand_distances, "hand_distances")
        self.pose_distances = load_config(pose_distances, "pose_distances")

        # Cache landmark index pairs and their names
        self.pose_distance_pairs = list(self.pose_distances.values())
        self.pose_distance_names = list(self.pose_distances.keys())

        self.hand_distance_pairs = list(self.hand_distances.values())
        self.hand_distance_names = list(self.hand_distances.keys())

    def __compute_distances(
        self, landmark_pairs: List[Tuple[int]], landmarks: Iterable, distance_type: str
    ) -> List[float]:
        return [
            distance(landmarks[start], landmarks[end], self.mode, distance_type)
            for start, end in landmark_pairs
        ]

    def compute_distances(
        self,
        landmarks: Iterable,
        landmark_type: str,
        distance_type: str = "shifted_dist",
    ) -> List[float]:
        """
        Compute distances between defined landmark pairs.

        Parameters:
        ----------
        landmarks : Iterable
            List of landmark objects.
        landmark_type : str
            Either 'pose' or 'hand'.
        distance_type : str
            Type of distance: 'dist', 'normalized_dist', or 'shifted_dist'.

        Returns:
        -------
        List[float] : Computed distance values.
        """
        check_landmark_type(landmark_type)
        check_distance_type(distance_type)

        if landmark_type == "pose":
            return self.__compute_distances(
                self.pose_distance_pairs, landmarks, distance_type
            )
        else:
            return self.__compute_distances(
                self.hand_distance_pairs, landmarks, distance_type
            )

    def compute_annotated_distances(
        self,
        landmarks: Iterable,
        landmark_type: str,
        distance_type: str = "shifted_dist",
    ) -> Dict[str, float]:
        """
        Compute named distances for use in analysis or visualization.

        Parameters:
        ----------
        landmarks : Iterable
            List of landmark objects.
        landmark_type : str
            Either 'pose' or 'hand'.
        distance_type : str
            Type of distance to compute.

        Returns:
        -------
        Dict[str, float] : Mapping of distance name to computed value.
        """
        check_landmark_type(landmark_type)
        check_distance_type(distance_type)

        if landmark_type == "pose":
            distances = self.__compute_distances(
                self.pose_distance_pairs, landmarks, distance_type
            )
            names = self.pose_distance_names
        elif landmark_type == "hand":
            distances = self.__compute_distances(
                self.hand_distance_pairs, landmarks, distance_type
            )
            names = self.hand_distance_names
        else:
            raise ValueError(
                "Parameter 'landmark_type' must be either 'pose' or 'hand'."
            )

        return dict(zip(names, distances))
