import yaml
from typing import Union, Dict


def check_mode(mode: str):
    if mode not in ["2D", "3D"]:
        raise Exception("Parameter 'mode' must be equal '2D' or '3D'")


def check_landmark_type(landmark_type: str):
    if landmark_type not in ["hand", "pose"]:
        raise Exception("Parameter 'landmark_type' must be equal 'hand' or 'pose'")


def check_angle_type(angle_type: str):
    if angle_type not in ["rad", "grad", "func", "normalized_rad", "shifted_rad"]:
        raise Exception(
            "Parameter 'ange_type' should be one of ['rad', 'grad', 'func', 'normalized_rad', 'shifted_rad']"
        )


def check_distance_type(distance_type: str):
    if distance_type not in ["dist", "normalized_dist", "shifted_dist"]:
        raise Exception(
            "Parameter 'distance_type' should be one of ['dist', 'normalized_dist', 'shifted_dist']"
        )


def check_difference_type(diff_type: str):
    if diff_type not in ["diff", "normalized_diff"]:
        raise ValueError("Parameter 'diff_type' must be 'diff' or 'normalized_diff'")


def load_config(config: Union[str, Dict], config_name: str) -> Dict:
    if isinstance(config, str):
        with open(config, "r") as f:
            return yaml.safe_load(f)
    elif isinstance(config, dict):
        return config
    else:
        raise ValueError(
            f"Parameter {config_name} must be either a file path or dictionary."
        )
