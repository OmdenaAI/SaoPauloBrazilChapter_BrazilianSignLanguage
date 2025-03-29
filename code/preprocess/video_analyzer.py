import os
import json
import time
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any, Union
import datetime

import motion_detection as md

try:
    import mediapipe_holistic as mph
except ImportError:
    print("MediaPipeHolistic not available. Pose estimation functions will not work.")


class VideoAnalyzer:
    """
    VideoAnalyzer class for processing and analyzing videos with motion detection and pose estimation.
    
    This class handles:
    - Motion detection and analysis
    - Pose estimation using MediaPipe Holistic
    - Result caching and loading
    - Parameter management
    """
    
    def __init__(
        self,
        metadata_row: Dict,
        timestamp: str,
        path_to_root: str,
        params_dict: Dict = None,
        verbose: bool = True,
        motion_detection_version: str = "versionA",
        pose_detection_version: str = "versionA",
        ):
        """
        Initialize the VideoAnalyzer with metadata and parameters.
        
        Args:
            metadata_row: Dictionary containing video metadata (filename, fps, width, height, etc.)
            timestamp: Timestamp for this analysis run
            path_to_root: Path to the root directory of the project
            params_dict: Dictionary of parameters for analysis (thresholds, etc.)
            verbose: Whether to print details about parameter sources
            motion_detection_version: Version of motion detection to use (default: "versionA")
            pose_detection_version: Version of pose detection to use (default: "versionA")
        """
        # Set required parameters as attributes
        self.timestamp = timestamp
        self.path_to_root = path_to_root
        self.motion_detection_version = motion_detection_version
        self.pose_detection_version = pose_detection_version
        
        # Set up base directories
        self.interim_dir = os.path.join(self.path_to_root, "data", "interim")
        self.motion_detection_dir = os.path.join(self.interim_dir, "RawMotionMeasurements")
        self.pose_estimation_dir = os.path.join(self.interim_dir, "RawPoseLandmarks")
        self.videos_dir = os.path.join(self.interim_dir, "Videos")
        self.results_dir = os.path.join(self.interim_dir, "Analysis", self.timestamp, "individual_json")
        
        # Create directories if they don't exist
        for directory in [
            os.path.join(self.interim_dir),
            os.path.join(self.motion_detection_dir),
            os.path.join(self.motion_detection_dir, self.motion_detection_version),
            os.path.join(self.pose_estimation_dir),
            os.path.join(self.pose_estimation_dir, self.pose_detection_version), 
            os.path.join(self.videos_dir),
            os.path.join(self.results_dir),
        ]:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                print(f"Created directory: {directory}")

        # Set attributes from metadata
        self.attributes_from_metadata(metadata_row)
        
        # Set parameters from params_dict or use defaults
        self.params = {}
        self.check_params(params_dict or {}, metadata_row, verbose=verbose)
        
        # Initialize results containers
        self.motion_data = {}
        self.pose_data = {}
        self.start_frame = None
        self.end_frame = None
        
        print(f"Initialized VideoAnalyzer for {self.filename}")
        print(f"Video properties: {self.fps} fps, {self.width}x{self.height}, {self.duration_sec:.2f} seconds")
    
    def attributes_from_metadata(self, metadata: Dict) -> None:
        """
        Set object attributes from metadata dictionary.
        
        Args:
            metadata: Dictionary containing video metadata
        """
        for key, value in metadata.items():
            setattr(self, key, value)
        
        # Ensure required attributes exist
        required_attrs = ["filename", "data_source", "fps", "width", "height", "duration_sec", "frame_count"]
        missing_attrs = [attr for attr in required_attrs if not hasattr(self, attr)]
        
        if missing_attrs:
            raise ValueError(f"Missing required metadata attributes: {', '.join(missing_attrs)}")
    
    def check_params(self, params: Dict, metadata: Dict, verbose: bool = True) -> None:
        """
        Check and set parameters from params dict, metadata, or defaults.
        
        Args:
            params: Dictionary of parameters for analysis
            metadata: Dictionary containing video metadata
            verbose: Whether to print details about parameter sources
        """
        # Define default parameters
        default_params = {
            # Motion detection parameters
            "motion_avg_weights": {"basic": 0.3, "bg_sub": 0.7},
            "moving_avg_window_duration": 0.2,  # in seconds
            "motion_threshold_method": "simple",
            "motion_start_threshold": 0.3,
            "motion_end_threshold": 0.2,
            
            # Pose estimation parameters
            "pose_static_image_mode": False,
            "pose_model_complexity": 1,
            "pose_smooth_landmarks": True,
            "pose_min_detection_confidence": 0.5,
            "pose_min_tracking_confidence": 0.5,
            
            # Processing parameters
            "reuse_results": True
        }
        
        # Apply any other parameters logic here
        # e.g. data source specific adjustments
        # if metadata.get("data_source") == "vl":
        #     default_params["motion_start_threshold"] = 0.5

        # Track parameter sources for verbose output
        from_params = {}
        from_metadata = {}
        from_defaults = {}

        # Set parameters, with priority: params > metadata > defaults
        for key, default_value in default_params.items():
            # Try to get from params dict
            if key in params:
                self.params[key] = params[key]
                from_params[key] = params[key]
            # Try to get from metadata
            elif key in metadata:
                self.params[key] = metadata[key]
                from_metadata[key] = metadata[key]
            # Use default
            else:
                self.params[key] = default_value
                from_defaults[key] = default_value
        
        # Save params for reference
        params_path = os.path.join(
            self.results_dir,
            f"{self.filename.split('.')[0]}_params.json"
            )
        with open(params_path, "w") as f:
            json.dump(self.params, f, indent=4)
        
        # Print parameter sources if verbose is True
        if verbose:
            if from_metadata:
                print("Using parameters from metadata:")
                for key, value in from_metadata.items():
                    print(f"\t{key} = {value}")
            
            if from_params:
                print("Using parameters from params dict:")
                for key, value in from_params.items():
                    print(f"\t{key} = {value}")
            
            if from_defaults:
                print("Using parameters from default values:")
                for key, value in from_defaults.items():
                    print(f"\t{key} = {value}")
    
    def get_raw_video_path(self) -> str:
        """Get the path to the raw video file based on metadata."""
        path_to_root = self.path_to_root
        filename = self.filename
        
        # Look in combined folder
        video_path = os.path.join(path_to_root, "data", "raw", "combined", "videos", filename)
        if os.path.exists(video_path):
            return video_path
        
        raise FileNotFoundError(f"Video file not found at {video_path}")
    
    def load_previous_results(self, result_type: str, timestamp: Optional[str] = None, verbose: bool = True) -> Dict:
        """
        Load previously saved results if they exist.
        
        Args:
            result_type: Type of results to load ('motion' or 'pose')
            timestamp: Specific timestamp to load (if None, loads the most recent)
            verbose: Whether to print status messages
            
        Returns:
            Dictionary containing loaded results or empty dict if none found
        """
        if result_type == "motion":
            base_path = os.path.join(self.motion_detection_dir, self.motion_detection_version)
        elif result_type == "pose":
            base_path = os.path.join(self.pose_estimation_dir, self.pose_detection_version)
        else:
            raise ValueError(f"Unknown result type: {result_type}")
        
        # Load results
        results = {}
        
        if result_type == "motion":
            # Load motion data
            basic_path = os.path.join(base_path, f"{self.filename.split('.')[0]}_basic.npy")
            bg_sub_path = os.path.join(base_path, f"{self.filename.split('.')[0]}_bg_sub.npy")
            
            if os.path.exists(basic_path) and os.path.exists(bg_sub_path):
                results["basic_raw"] = np.load(basic_path)
                results["bg_sub_raw"] = np.load(bg_sub_path)
                if verbose:
                    print(f"Loaded motion detection results from {self.motion_detection_version}")
            else:
                if verbose:
                    print(f"No motion detection results found for {self.filename} in {self.motion_detection_version}")
            
        elif result_type == "pose":
            # Load pose data
            params_path = os.path.join(base_path, "mediapipe_params.json")
            landmarks_path = os.path.join(base_path, f"{self.filename.split('.')[0]}.npy")
            
            if os.path.exists(params_path) and os.path.exists(landmarks_path):
                with open(params_path, "r") as f:
                    results["params"] = json.load(f)
                
                results["landmarks_raw"] = np.load(landmarks_path, allow_pickle=True)
                if verbose:
                    print(f"Loaded pose estimation results from {self.pose_detection_version}")
            else:
                if verbose:
                    print(f"No pose estimation results found for {self.filename} in {self.pose_detection_version}")
        
        return results
    
    def motion_detect(self, weights: Optional[Dict[str, float]] = None, 
                     window_duration: Optional[float] = None,
                     load: Optional[bool] = None) -> Dict:
        """
        Perform motion detection on the video and calculate derived metrics.
        
        Args:
            weights: Dictionary of weights for different motion detection methods
            window_duration: Duration in seconds for moving average window
            load: Whether to load previous results if available
            
        Returns:
            Dictionary containing motion detection results
        """
        # Use provided parameters or fall back to class parameters
        weights = weights or self.params.get("motion_avg_weights")
        window_duration = window_duration or self.params.get("moving_avg_window_duration")
        load = load if load is not None else self.params.get("reuse_results")
        verbose = self.params.get("verbose", True)
        
        # Try to load previous results if requested
        if load:
            previous_results = self.load_previous_results("motion", verbose=verbose)
            if previous_results:
                self.motion_data = previous_results
                
                # Calculate derived metrics
                self._process_motion_data(weights, window_duration)
                return self.motion_data
        
        # If no previous results or not loading, run motion detection
        video_path = self.get_raw_video_path()
        if verbose:
            print(f"Running motion detection on {video_path}")
        
        # Run motion detection algorithms
        basic_motion = md.measure_motion_basic(video_path)
        bg_sub_motion = md.measure_motion_background_subtraction(video_path)
        
        # Store raw results
        self.motion_data = {
            "basic_raw": np.array(basic_motion),
            "bg_sub_raw": np.array(bg_sub_motion),
        }
        
        # Calculate derived metrics
        self._process_motion_data(weights, window_duration, verbose)
        
        # Save results
        self._save_raw_motion_results()
        
        return self.motion_data
    
    def _process_motion_data(self, weights: Dict[str, float], window_duration: float, verbose: bool = True) -> None:
        """
        Process raw motion data to calculate normalized and moving-average smoothed metrics.
        
        Args:
            weights: Dictionary of weights for different motion detection methods
            window_duration: Duration in seconds for moving average window
        """
        # Normalize data
        motion_series = [self.motion_data["basic_raw"], self.motion_data["bg_sub_raw"]]
        normalized_series = md.normalize_lists_of_data(motion_series)
        
        # Store normalized data
        self.motion_data["basic_normalized"] = normalized_series[0]
        self.motion_data["bg_sub_normalized"] = normalized_series[1]
        
        # Calculate weighted average from normalized data
        weight_values = [weights["basic"], weights["bg_sub"]]
        weighted_avg = md.weighted_average_motion(normalized_series, weight_values)
        self.motion_data["weighted_avg"] = weighted_avg
        
        # Apply moving average
        smoothed_motion = md.moving_average(weighted_avg, self.fps, window_duration, verbose=verbose)
        self.motion_data["smoothed"] = smoothed_motion
    
    def _save_raw_motion_results(self, verbose: bool = True) -> None:
        """Save only the raw motion detection results to disk. They can be reused later with different processing parameters."""
        # Create directory for results
        result_dir = os.path.join(
            self.motion_detection_dir,
            self.motion_detection_version
        )
        os.makedirs(result_dir, exist_ok=True)
        
        # Save raw motion data
        np.save(os.path.join(result_dir, f"{self.filename.split('.')[0]}_basic.npy"), self.motion_data["basic_raw"])
        np.save(os.path.join(result_dir, f"{self.filename.split('.')[0]}_bg_sub.npy"), self.motion_data["bg_sub_raw"])
        
        if verbose:
            print(f"Saved motion detection results to {result_dir}")
    
    def motion_analyze(self, 
                      motion_data: Optional[np.ndarray] = None,
                      start_threshold: Optional[float] = None,
                      end_threshold: Optional[float] = None,
                      method: Optional[str] = None) -> Dict:
        """
        Analyze motion data to find start and end frames of meaningful motion.
        
        Args:
            motion_data: Motion data array to analyze (defaults to self.motion_data["smoothed"])
            start_threshold: Threshold for detecting start of motion
            end_threshold: Threshold for detecting end of motion
            method: Method for motion boundary detection ('simple' or 'complex')
            
        Returns:
            Dictionary containing start and end frames
        """
        # Use provided parameters or fall back to class parameters
        motion_data = motion_data if motion_data is not None else self.motion_data.get("smoothed")
        start_threshold = start_threshold or self.params.get("motion_start_threshold")
        end_threshold = end_threshold or self.params.get("motion_end_threshold")
        method = method or self.params.get("motion_threshold_method")
        
        if motion_data is None:
            raise ValueError("No motion data available. Run motion_detect first.")
        
        print(f"Analyzing motion using {method} method with thresholds {start_threshold}/{end_threshold}")
        
        # Detect motion boundaries
        if method == "simple":
            start_frame, end_frame = md.find_motion_boundaries_simple(
                motion_data, 
                start_threshold,
                end_threshold
            )
        elif method == "complex":
            start_frame, end_frame = md.find_motion_boundaries_complex(
                motion_data,
                self.fps,
                threshold=start_threshold,
                min_motion_duration=self.params.get("min_motion_duration", 0.3)
            )
        else:
            raise ValueError(f"Unknown motion analysis method: {method}")
        
        self.start_frame = start_frame
        self.end_frame = end_frame
        
        result = {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time": start_frame / self.fps if start_frame is not None else None,
            "end_time": end_frame / self.fps if end_frame is not None else None,
            "duration_frames": end_frame - start_frame if (start_frame is not None and end_frame is not None) else None,
            "duration_sec": (end_frame - start_frame) / self.fps if (start_frame is not None and end_frame is not None) else None
        }
        
        print(f"Motion detected from frame {start_frame} to {end_frame} ({result['duration_sec']:.2f} seconds)")
        
        # Store analysis results
        self.motion_data["analysis"] = result
        
        return result
    
    def pose_detect(self, load: Optional[bool] = None) -> Dict:
        """
        Perform pose estimation on the video using MediaPipe Holistic.
        
        Args:
            load: Whether to load previous results if available
            
        Returns:
            Dictionary containing pose estimation results
        """
        verbose = self.params.get("verbose", True)
        # Use provided parameter or fall back to class parameter
        load = load if load is not None else self.params.get("reuse_results")
        
        # Try to load previous results if requested
        if load:
            previous_results = self.load_previous_results("pose", verbose=verbose)
            if previous_results:
                self.pose_data = previous_results
                return self.pose_data
        
        # If no previous results or not loading, run pose estimation
        video_path = self.get_raw_video_path()
        if verbose:
            print(f"Running pose estimation on {video_path}")
        
        # Create MediaPipe instance with parameters
        mp_holistic = mph.MediaPipeHolistic(
            static_image_mode=self.params.get("pose_static_image_mode"),
            model_complexity=self.params.get("pose_model_complexity"),
            smooth_landmarks=self.params.get("pose_smooth_landmarks"),
            min_detection_confidence=self.params.get("pose_min_detection_confidence"),
            min_tracking_confidence=self.params.get("pose_min_tracking_confidence")
        )
        
        # Process video
        landmarks_list = mp_holistic.process_video(video_path)
        
        # Store results
        self.pose_data = {
            "landmarks_raw": landmarks_list,
            "params": {
                "pose_static_image_mode": self.params.get("pose_static_image_mode"),
                "pose_model_complexity": self.params.get("pose_model_complexity"),
                "pose_smooth_landmarks": self.params.get("pose_smooth_landmarks"),
                "pose_min_detection_confidence": self.params.get("pose_min_detection_confidence"),
                "pose_min_tracking_confidence": self.params.get("pose_min_tracking_confidence")
            },
        }
        
        # Save results
        self._save_pose_results(verbose)
        
        return self.pose_data
    
    def _save_pose_results(self, verbose: bool = True) -> None:
        """Save pose estimation results to disk."""
        # Create directory for results
        result_dir = os.path.join(
            self.pose_estimation_dir, 
            self.pose_detection_version
        )
        os.makedirs(result_dir, exist_ok=True)
        
        # Save landmarks data
        np.save(os.path.join(result_dir, f"{self.filename.split('.')[0]}.npy"), self.pose_data["landmarks_raw"])
        
        # Save MediaPipe parameters
        with open(os.path.join(result_dir, "mediapipe_params.json"), "w") as f:
            json.dump(self.pose_data["params"], f, indent=4)

        if verbose:
            print(f"Saved pose estimation results to {result_dir}")
        
    def pose_analyze(self) -> Dict:
        """
        Analyze pose data to get various measurements and statistics.
        
        Args:
            use_shoulders: Whether to use shoulder measurements
            use_face: Whether to use face measurements
            use_hips: Whether to use hip measurements
            
        Returns:
            Dictionary containing analysis results:
            - horizontal_offsets: Statistics for horizontal alignment
            - vertical_offsets: Statistics for vertical alignment
            - landmark_measurements: Statistics for various body measurements
            - scale_factors: Calculated scale factors if reference measurements provided
        """
        if not hasattr(self, "pose_data") or not self.pose_data:
            raise ValueError("No pose data available. Run pose_detect first.")
        
        verbose = self.params.get("verbose", True)
        if verbose:
            print("Analyzing pose data...")
        
        mp_holistic = mph.MediaPipeHolistic()
        # Get horizontal offsets
        horizontal_offsets = mp_holistic.get_video_horizontal_offsets(
            self.pose_data["landmarks_raw"]
        )
        
        # Get vertical offsets
        vertical_offsets = mp_holistic.get_video_vertical_offsets(
            self.pose_data["landmarks_raw"]
        )
        
        # Get landmark measurements
        landmark_measurements = mp_holistic.get_video_landmark_measurements(
            self.pose_data["landmarks_raw"]
        )
        
        # Store analysis results
        self.pose_data["analysis"] = {
            "horizontal_offsets": horizontal_offsets,
            "vertical_offsets": vertical_offsets,
            "landmark_measurements": landmark_measurements
        }
        
        if verbose:
            print("Pose analysis complete")
            print(f"Horizontal offsets: {horizontal_offsets}")
            print(f"Vertical offsets: {vertical_offsets}")
            print(f"Landmark measurements: {landmark_measurements}")
        
        return self.pose_data["analysis"]

    def save_analysis_info(self) -> Dict:
        """
        Save analysis information to a JSON file.
        Only saves parameters and results from analysis methods, not raw data arrays.
        
        Returns:
            Dictionary of analysis info that was saved
            
        Raises:
            ValueError: If any required analysis data is missing
        """
        def convert_numpy(obj):
            """Convert numpy types to Python native types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        # Check for required data
        if not hasattr(self, "motion_data") or not self.motion_data:
            raise ValueError("Motion data not available. Run motion_detect first.")
        if "analysis" not in self.motion_data:
            raise ValueError("Motion analysis not available. Run motion_analyze first.")
            
        if not hasattr(self, "pose_data") or not self.pose_data:
            raise ValueError("Pose data not available. Run pose_detect first.")
        if "analysis" not in self.pose_data:
            raise ValueError("Pose analysis not available. Run pose_analyze first.")
        
        # Convert metadata values
        metadata = {
            "filename": self.filename,
            "fps": int(self.fps),
            "width": int(self.width),
            "height": int(self.height),
            "duration_sec": float(self.duration_sec),
            "frame_count": int(self.frame_count),
            "data_source": getattr(self, "data_source", None),
            "label": getattr(self, "label", None),
            "motion_detection_version": self.motion_detection_version,
            "pose_detection_version": self.pose_detection_version,
            "timestamp": self.timestamp
        }
        
        # Convert motion analysis values
        motion_analysis = {
            "start_frame": int(self.motion_data["analysis"]["start_frame"]),
            "end_frame": int(self.motion_data["analysis"]["end_frame"]),
            "start_time": float(self.motion_data["analysis"]["start_time"]),
            "end_time": float(self.motion_data["analysis"]["end_time"]),
            "duration_frames": int(self.motion_data["analysis"]["duration_frames"]),
            "duration_sec": float(self.motion_data["analysis"]["duration_sec"])
        }
        
        # Collect all analysis info
        analysis_info = {
            "metadata": metadata,
            "params": self.params,
            "motion_analysis": motion_analysis,
        }
        
        # Add pose analysis results
        pose_analysis = self.pose_data["analysis"]
        analysis_info["pose_analysis"] = {
            "frames_with_landmarks": len(self.pose_data["landmarks_raw"]),
            "horizontal_offsets": convert_numpy(pose_analysis["horizontal_offsets"]),
            "vertical_offsets": convert_numpy(pose_analysis["vertical_offsets"]),
            "landmark_measurements": convert_numpy(pose_analysis["landmark_measurements"]),
        }
        
        # Save to file
        output_path = os.path.join(self.results_dir, f"{self.filename.split('.')[0]}_analysis_info.json")
        with open(output_path, "w") as f:
            json.dump(analysis_info, f, indent=4)
        
        print(f"Saved analysis info to {output_path}")
        return analysis_info