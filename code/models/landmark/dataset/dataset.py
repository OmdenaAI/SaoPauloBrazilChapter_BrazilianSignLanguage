from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch 
from utils import load_config
from typing import Dict, List, Union
from angles_estimator import AnglesEstimator
from distances_estimator import DistancesEstimator
from frame2frame_differences_estimator import DifferencesEstimator

class LandmarkFeatureTorchJoiner:
    def __init__(self, landmark_order: List[str]):
        self.landmark_order = landmark_order

    def forward(self, landmark_features: Dict):
        feature_vector = []
        for landmark_type in self.landmark_order:
            feature_vector.extend(landmark_features[landmark_type])
        return torch.tensor(feature_vector, dtype=torch.float)
    
class LandmarkDataset(Dataset):
    def __init__(self, config:Union[str, Dict]):
        config = load_config(config, "dataset_config")
        self.data_dir = config["data_dir"]
        self.augmentations = config["augmentations"]
        self.data = pd.read_csv(config["data_path"])
        self.data = self.data[self.data["dataset_split"] == config["dataset_split"]]
        self.landmark_order = config["landmark_order"]
        self.landmark_feature_list = config["landmark_feature_list"]

        self.angle_estimator = AnglesEstimator(
            hand_angles=config["hand_angle_triplets"],
            pose_angles=config["pose_angle_triplets"],
            mode=config.get("mode", "3D"),
        )
        self.distance_estimator = DistancesEstimator(
            hand_distances=config["hand_distance_pairs"],
            pose_distances=config["pose_distance_pairs"],
            mode=config.get("mode", "3D"),
        )
        self.diff_estimator = DifferencesEstimator(
            difference_points=config["difference_points"],
            mode=config.get("mode", "3D"),
        )

        self.joiner = LandmarkFeatureTorchJoiner(self.landmark_feature_list)


    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int):
        landmark_path = self.data_dir / self.data.loc[idx, "filename"]
        landmarks = np.load(landmark_path, allow_pickle=True)
        for aug in self.augmentations:
            if np.random.uniform() <= aug["p"]:
                landmarks = aug["augmentation"](landmarks)
        
        # Ensure at least 2 frames for frame-to-frame diff
        if len(landmarks) < 2:
            raise ValueError(f"Landmark file {landmark_path} has less than 2 frames.")

        # Use the last frame as current, previous one for difference
        prev_frame = landmarks[-2]
        curr_frame = landmarks[-1]

        for aug in self.augmentations:
            if np.random.uniform() <= aug["p"]:
                curr_frame = aug["augmentation"](curr_frame)
                prev_frame = aug["augmentation"](prev_frame)

        features = {}

        if "angles" in self.landmark_feature_list:
            features["angles"] = []
            if curr_frame["pose_landmarks"]:
                features["angles"] += self.angle_estimator.compute_angles(
                    curr_frame["pose_landmarks"], "pose", angle_type="func"
                )
            if curr_frame["left_hand_landmarks"]:
                features["angles"] += self.angle_estimator.compute_angles(
                    curr_frame["left_hand_landmarks"], "hand", angle_type="func"
                )
            if curr_frame["right_hand_landmarks"]:
                features["angles"] += self.angle_estimator.compute_angles(
                    curr_frame["right_hand_landmarks"], "hand", angle_type="func"
                )

        if "distances" in self.landmark_feature_list:
            features["distances"] = []
            if curr_frame["pose_landmarks"]:
                features["distances"] += self.distance_estimator.compute_distances(
                    curr_frame["pose_landmarks"], "pose", distance_type="shifted_dist"
                )
            if curr_frame["left_hand_landmarks"]:
                features["distances"] += self.distance_estimator.compute_distances(
                    curr_frame["left_hand_landmarks"], "hand", distance_type="shifted_dist"
                )
            if curr_frame["right_hand_landmarks"]:
                features["distances"] += self.distance_estimator.compute_distances(
                    curr_frame["right_hand_landmarks"], "hand", distance_type="shifted_dist"
                )

        if "differences" in self.landmark_feature_list:
            diffs = self.diff_estimator.compute_differences(
                prev_landmarks=prev_frame["pose_landmarks"] + 
                               prev_frame["left_hand_landmarks"] + 
                               prev_frame["right_hand_landmarks"],
                next_landmarks=curr_frame["pose_landmarks"] + 
                               curr_frame["left_hand_landmarks"] + 
                               curr_frame["right_hand_landmarks"],
                diff_type="normalized_diff"
            )
            features["differences"] = [x for vec in diffs for x in vec]

        return self.joiner.forward(features)