import numpy as np
from typing import Union, Dict, List, Tuple, Iterable
from utils import check_mode, check_landmark_type, check_angle_type, load_config
import numpy as np



def angle(a, b, c, mode: str = '3D', angle_type: str = 'func') -> Union[float, Tuple[float]]:
    """
    Computes the angle between vectors AB and BC, with various output formats.

    Parameters:
    ----------
    a, b, c : objects
        Points with attributes x, y, and optionally z (for 3D). 
        The angle is calculated at point `b` between vectors AB and CB.
    
    mode : str, optional (default='3D')
        Coordinate mode: '2D' uses only x and y, '3D' uses x, y, z.
    
    anlge_type : str, optional (default='func')
        Output format of the angle. Supported options:
        - 'rad'            : Returns the angle in radians [0, π].
        - 'grad'           : Returns the angle in degrees [0, 180].
        - 'normalized_rad' : Returns the angle normalized to [0, 1] as angle_rad / π.
        - 'shifted_rad'    : Returns the angle normalized to [-1, 1] as (angle_rad / π) * 2 - 1.
        - 'func'           : Returns a tuple of (sin(angle_rad), cos(angle_rad)), both ∈ [-1, 1].

    Returns:
    -------
    float or tuple of float
        Depending on `angle_type`, returns a float (angle in selected format) or a tuple (sin, cos).

    Notes:
    -----
    - If the magnitude of either vector AB or CB is zero, returns 0 (or (0, 1) for 'func') to avoid division by zero
    
    Examples:
    --------
    angle(a, b, c, mode='2D', angle_type='normalized_rad')  # → 0.5 for 90 degrees
    angle(a, b, c, angle_type='func')                       # → (0.707, 0.707) for 45 degrees
    """
    
    check_mode(mode)
    check_angle_type(angle_type)
    
    if mode == '3D':
        ab = [a.x - b.x, a.y - b.y, a.z - b.z]
        cb = [c.x - b.x, c.y - b.y, c.z - b.z]
    else:
        ab = [a.x - b.x, a.y - b.y]
        cb = [c.x - b.x, c.y - b.y]
    
    dot_product = sum(ai * ci for ai, ci in zip(ab, cb))
    norm_ab = np.linalg.norm(ab)
    norm_cb = np.linalg.norm(cb)
    
    if norm_ab * norm_cb == 0:
        return 0.0 if angle_type != 'func' else (0.0, 1.0)  # avoid division by zero
    
    cos_theta = dot_product / (norm_ab * norm_cb)
    cos_theta = max(min(cos_theta, 1), -1)  # clamp for stability
    angle_rad = np.arccos(cos_theta)
    
    if angle_type == 'rad':
        return angle_rad
    elif angle_type == 'grad':
        return np.degrees(angle_rad)
    elif angle_type == 'normalized_rad':
        return angle_rad / np.pi  # [0, 1]
    elif angle_type == 'shifted_rad':
        return (angle_rad / np.pi) * 2 - 1  # [-1, 1]
    elif angle_type == 'func':
        return np.sin(angle_rad), np.cos(angle_rad)

class AnglesEstimator:
    """
    Estimates angles between triplets of landmarks.
    """

    def __init__(self, 
                 hand_angles: Union[str, Dict], 
                 pose_angles: Union[str, Dict], 
                 mode: str = '3D'):
        """
        Parameters:
        ----------
       hand_angles : str or dict
            Path to YAML file or dictionary with hand landmark triplets.
        pose_angles : str or dict
            Path to YAML file or dictionary with pose landmark triplets.
        mode : str
            '2D' or '3D' coordinate mode for angle computation.
        """
        self.hand_angles = load_config(hand_angles, "hand_angles")
        self.pose_angles = load_config(pose_angles, "pose_angles")

        # cached to speed up computing
        self.pose_angle_triplets = list(self.pose_angles.values())
        self.pose_angle_names = list(self.pose_angles.keys())
        self.hand_angle_triplets = list(self.hand_angles.values())
        self.hand_angle_names = list(self.hand_angles.keys())
        check_mode(mode)
        self.mode = mode


    def __compute_angles(self, 
                         landmark_triplets: Dict[str, Tuple[int]], 
                         landmarks: Iterable, 
                         angle_type: str) -> List[float]:
        return [
            angle(landmarks[start], landmarks[middle], landmarks[end], self.mode, angle_type) 
            for start, middle, end in landmark_triplets.values()
        ]

    def compute_angles(self, 
                       landmarks: Iterable, 
                       landmark_type: str, 
                       angle_type: str) -> List[float]:
        """
        Compute angles (as features) for either 'pose' or 'hand' landmarks.

        Parameters:
        ----------
        landmarks : Iterable
            List of landmark objects.
        landmark_type : str
            Either 'pose' or 'hand'.
        angle_type : str
            Angle representation to use.

        Returns:
        -------
        List[float] : List of computed angles.
        """
        check_landmark_type(landmark_type)
        check_angle_type(angle_type)

        if landmark_type == 'pose':
            return self.__compute_angles(self.pose_angles, landmarks, angle_type)
        else:
            return self.__compute_angles(self.hand_angles, landmarks, angle_type)

    def compute_annotated_angles(self, 
                                 landmarks: Iterable, 
                                 landmark_type: str, 
                                 angle_type: str) -> Dict[str, float]:
        """
        Compute angles and return a dictionary with named features.

        Parameters:
        ----------
        landmarks : Iterable
            List of landmark objects.
        landmark_type : str
            Either 'pose' or 'hand'.
        angle_type : str
            Angle representation to use.

        Returns:
        -------
        Dict[str, float] : Mapping from feature name to computed angle value. For visualization/debug.
        """
        check_landmark_type(landmark_type)
        check_angle_type(angle_type)

        if landmark_type == 'pose':
            angles = self.__compute_angles(self.pose_angles, landmarks, angle_type)
            keys = self.pose_angles.keys()
        else:
            angles = self.__compute_angles(self.hand_angles, landmarks, angle_type)
            keys = self.hand_angles.keys()
        
        return dict(zip(keys, angles))
