from typing import Dict
import numpy as np


class RotateLandmarks:
    def __init__(self, angle_range: float = 10.0):
        self.angle_range = angle_range

    def __call__(self, landmarks: Dict) -> Dict:
        angle_rad = np.radians(np.random.uniform(-self.angle_range, self.angle_range))
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)

        def rotate(lms):
            for lm in lms:
                x_new = cos_theta * lm.x - sin_theta * lm.y
                y_new = sin_theta * lm.x + cos_theta * lm.y
                lm.x, lm.y = x_new, y_new
            return lms

        for key in ["pose_landmarks", "left_hand_landmarks", "right_hand_landmarks"]:
            if key in landmarks and landmarks[key]:
                landmarks[key] = rotate(landmarks[key])

        return landmarks


class LandmarksNoise:
    def __init__(self, noise_std: float = 0.01):
        self.noise_std = noise_std

    def __call__(self, landmarks: Dict) -> Dict:
        def add_noise(lms):
            for lm in lms:
                lm.x += np.random.normal(0, self.noise_std)
                lm.y += np.random.normal(0, self.noise_std)
                if hasattr(lm, "z"):
                    lm.z += np.random.normal(0, self.noise_std)
            return lms

        for key in ["pose_landmarks", "left_hand_landmarks", "right_hand_landmarks"]:
            if key in landmarks and landmarks[key]:
                landmarks[key] = add_noise(landmarks[key])

        return landmarks


AUGMENTATIONS = {
    "train": [
        {"augmentation": RotateLandmarks, "p": 0.5},
        {"augmentation": LandmarksNoise, "p": 0.5},
    ],
    "val": [],
    "test": [],
}
