import os
import json
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import shutil
import copy

class Preprocessor:
    """
    Preprocessor class for applying consistent transformations to videos and landmarks.
    
    This class handles:
    - Trimming videos/landmarks by start and end frames
    - Horizontal and vertical alignment
    - Scaling content
    - Temporal padding to ensure consistent duration
    - Crop and resize
    - Saving results with appropriate metadata
    """
    
    def __init__(
        self,
        metadata_row: Dict,
        params_dict: Dict,
        path_to_root: str,
        preprocess_version: str = "v1",
        verbose: bool = True
    ):
        """
        Initialize the Preprocessor with metadata and parameters.
        
        Args:
            metadata_row: Dictionary containing video metadata (filename, fps, width, height, etc.)
            params_dict: Dictionary of preprocessing parameters
            path_to_root: Path to the root directory of the project
            preprocess_version: Version identifier for this preprocessing (default: "v1")
            verbose: Whether to print details about processing steps
        """
        # Store inputs
        self.metadata = metadata_row
        self.params = params_dict
        self.path_to_root = path_to_root
        self.preprocess_version = preprocess_version
        self.verbose = verbose
        
        # Extract key metadata
        self.filename = metadata_row["filename"]
        self.fps = metadata_row["fps"]
        self.width = metadata_row["width"]
        self.height = metadata_row["height"]
        self.frame_count = metadata_row["frame_count"]
        self.duration_sec = metadata_row["duration_sec"]
        
        # Set up directory paths
        self.raw_dir = os.path.join(path_to_root, "data", "raw", "combined")
        self.interim_dir = os.path.join(path_to_root, "data", "interim")
        self.preprocessed_dir = os.path.join(path_to_root, "data", "preprocessed")
        
        # Raw video path
        self.raw_videos_dir = os.path.join(self.raw_dir, "videos")
        
        # Interim paths
        self.interim_videos_dir = os.path.join(self.interim_dir, "Videos")
        self.interim_landmarks_dir = os.path.join(self.interim_dir, "RawPoseLandmarks")
        
        # Preprocessed paths
        self.preprocessed_videos_dir = os.path.join(self.preprocessed_dir, "Videos", preprocess_version)
        self.preprocessed_landmarks_dir = os.path.join(self.preprocessed_dir, "Landmarks", preprocess_version)
        
        # Create output directories if they don't exist
        for directory in [
            self.interim_videos_dir,
            os.path.join(self.preprocessed_videos_dir, "videos"),
            os.path.join(self.preprocessed_videos_dir, "individual_metadata"),
            os.path.join(self.preprocessed_landmarks_dir, "landmarks"),
            os.path.join(self.preprocessed_landmarks_dir, "individual_metadata")
        ]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                if self.verbose:
                    print(f"Created directory: {directory}")
                
        # Validate required parameters
        self._validate_params()
        
        if verbose:
            print(f"Initialized Preprocessor for {self.filename}")
            print(f"Naming this preprocessing version: {preprocess_version}")
    
    def _validate_params(self) -> None:
        """
        Validate that all required parameters are present and have valid values.
        
        Raises:
            ValueError: If any required parameter is missing or invalid
        """
        required_params = [
            "start_frame", 
            "end_frame", 
            "horizontal_offset", 
            "x_scale_factor", 
            "y_scale_factor",
            "vertical_offset", 
            "target_duration"
        ]
        
        missing_params = [param for param in required_params if param not in self.params]
        if missing_params:
            raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")
        
        # Validate parameter ranges
        if not (-1 <= self.params["horizontal_offset"] <= 1):
            raise ValueError(f"horizontal_offset must be between -1 and 1, got {self.params['horizontal_offset']}")
        
        if not (-1 <= self.params["vertical_offset"] <= 1):
            raise ValueError(f"vertical_offset must be between -1 and 1, got {self.params['vertical_offset']}")
        
        if self.params["x_scale_factor"] <= 0:
            raise ValueError(f"x_scale_factor must be positive, got {self.params['x_scale_factor']}")
            
        if self.params["y_scale_factor"] <= 0:
            raise ValueError(f"y_scale_factor must be positive, got {self.params['y_scale_factor']}")
        
        if self.params["target_duration"] <= 0:
            raise ValueError(f"target_duration must be positive, got {self.params['target_duration']}")
            
        if self.params["start_frame"] < 0 or self.params["start_frame"] >= self.frame_count:
            raise ValueError(f"start_frame must be between 0 and {self.frame_count-1}, got {self.params['start_frame']}")
            
        if self.params["end_frame"] <= self.params["start_frame"] or self.params["end_frame"] >= self.frame_count:
            raise ValueError(f"end_frame must be between {self.params['start_frame']+1} and {self.frame_count-1}, got {self.params['end_frame']}")
    
    def preprocess_video(self, save_intermediate: bool = False) -> str:
        """
        Apply preprocessing steps to video and save result.
        
        Args:
            save_intermediate: Whether to save after each major step
            
        Returns:
            Path to preprocessed video
        """
        video_path = os.path.join(self.raw_videos_dir, self.filename)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at {video_path}")
        
        if self.verbose:
            print(f"Preprocessing video: {video_path}")
            
        # Open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")
        
        # Read all framesManual control from frontend (e.g. "Stop" button)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        if self.verbose:
            print(f"Loaded {len(frames)} frames from video")
            
        # Step 1: Trim frames
        trimmed_frames = self._trim_frames(frames)
        if save_intermediate:
            self._save_intermediate_video(trimmed_frames, "trimmed")
            
        # Step 2: Horizontal alignment
        aligned_frames = self._horizontal_align_frames(trimmed_frames)
        if save_intermediate:
            self._save_intermediate_video(aligned_frames, "h_aligned")
            
        # Step 3: Scale content
        scaled_frames = self._scale_frames(aligned_frames)
        if save_intermediate:
            self._save_intermediate_video(scaled_frames, "scaled")
            
        # Step 4: Vertical alignment
        v_aligned_frames = self._vertical_align_frames(scaled_frames)
        
        # Save after vertical alignment to interim directory
        interim_video_path = self._save_interim_video(v_aligned_frames)
        
        # Step 5: Padding to target duration
        padded_frames = self._pad_frames_to_duration(v_aligned_frames)
        if save_intermediate:
            self._save_intermediate_video(padded_frames, "padded")
            
        # Step 6: Crop/resize if needed
        processed_frames = self._crop_resize_frames(padded_frames)
        
        # Step 7: Save final preprocessed video
        output_path = self._save_preprocessed_video(processed_frames)
        
        return output_path
    
    def preprocess_landmarks(self, landmark_version: str = "versionA") -> str:
        """
        Apply preprocessing steps to landmarks and save result.
        
        Args:
            landmark_version: Version of landmarks to process
            
        Returns:
            Path to preprocessed landmarks
        """
        # Get landmarks path
        landmarks_path = os.path.join(
            self.interim_landmarks_dir, 
            landmark_version, 
            f"{self.filename.split('.')[0]}.npy"
        )
        
        if not os.path.exists(landmarks_path):
            raise FileNotFoundError(f"Landmarks file not found at {landmarks_path}")
        
        if self.verbose:
            print(f"Preprocessing landmarks: {landmarks_path}")
            
        # Load landmarks
        landmarks = np.load(landmarks_path, allow_pickle=True)
        
        if self.verbose:
            print(f"Loaded landmarks with shape: {landmarks.shape}")
            
        # Step 1: Trim landmarks
        trimmed_landmarks = self._trim_landmarks(landmarks)

        output_filename = f"{self.filename.split('.')[0]}_1_trimmed.npy"
        output_path = os.path.join(self.interim_landmarks_dir, "versionA", output_filename)
        np.save(output_path, trimmed_landmarks)

        # Step 2: Horizontal alignment
        aligned_landmarks = self._horizontal_align_landmarks(trimmed_landmarks)
        
        # DEBUG: Compare trimmed and aligned landmarks
        self._debug_landmarks_format(trimmed_landmarks, aligned_landmarks, frame_idx=0, 
                                  label1="Trimmed", label2="Horizontally Aligned")

        output_filename = f"{self.filename.split('.')[0]}_2_h_aligned.npy"
        output_path = os.path.join(self.interim_landmarks_dir, "versionA", output_filename)
        np.save(output_path, aligned_landmarks)
        
        # Step 3: Scale landmarks
        scaled_landmarks = self._scale_landmarks(aligned_landmarks)

        output_filename = f"{self.filename.split('.')[0]}_3_scaled.npy"
        output_path = os.path.join(self.interim_landmarks_dir, "versionA", output_filename)
        np.save(output_path, scaled_landmarks)
        
        # Step 4: Vertical alignment
        v_aligned_landmarks = self._vertical_align_landmarks(scaled_landmarks)

        output_filename = f"{self.filename.split('.')[0]}_4_v_aligned.npy"
        output_path = os.path.join(self.interim_landmarks_dir, "versionA", output_filename)
        np.save(output_path, v_aligned_landmarks)
        
        # Step 5: Padding to target duration
        padded_landmarks = self._pad_landmarks_to_duration(v_aligned_landmarks)
        
        output_filename = f"{self.filename.split('.')[0]}_5_padded.npy"
        output_path = os.path.join(self.interim_landmarks_dir, "versionA", output_filename)
        np.save(output_path, padded_landmarks)
        
        # Step 6: Crop/resize if needed (equivalent for landmarks)
        processed_landmarks = self._crop_resize_landmarks(padded_landmarks)
        
        output_filename = f"{self.filename.split('.')[0]}_6_cropped.npy"
        output_path = os.path.join(self.interim_landmarks_dir, "versionA", output_filename)
        np.save(output_path, processed_landmarks)
        
        # Step 7: Save final preprocessed landmarks
        output_path = self._save_preprocessed_landmarks(processed_landmarks)
        
        return output_path
    
    def _trim_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Trim frames to only include the range between start_frame and end_frame.
        
        Args:
            frames: List of frames to trim
            
        Returns:
            List of trimmed frames
        """
        start_frame = self.params["start_frame"]
        end_frame = self.params["end_frame"]
        
        if start_frame >= len(frames) or end_frame >= len(frames):
            raise ValueError(f"Invalid frame range: {start_frame} to {end_frame}, video has {len(frames)} frames")
            
        trimmed = frames[start_frame:end_frame+1]
        
        if self.verbose:
            print(f"Trimmed video from {len(frames)} frames to {len(trimmed)} frames")
            print(f"Frame range: {start_frame} to {end_frame}")
            
        return trimmed
    
    def _horizontal_align_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Align frames horizontally based on horizontal_offset.
        
        Args:
            frames: List of frames to align
            
        Returns:
            List of horizontally aligned frames
            
        Note:
            horizontal_offset is between -1 and 1, where:
            - 0 means no change
            - Negative values shift the content left
            - Positive values shift the content right
        """
        horizontal_offset = self.params["horizontal_offset"]
        
        # No change needed if offset is 0
        if horizontal_offset == 0:
            if self.verbose:
                print("No horizontal alignment applied (horizontal_offset = 0)")
            return [frame.copy() for frame in frames]
        
        # Calculate shift amount directly from horizontal_offset
        # horizontal_offset range: -1 to 1, where negative shifts left, positive shifts right
        
        # Convert to pixels
        width = frames[0].shape[1]
        pixel_shift = int(round(width * horizontal_offset))
        
        aligned_frames = []
        for frame in frames:
            # Create a new frame filled with edge colors instead of zeros
            aligned_frame = np.zeros_like(frame)
            
            if pixel_shift > 0:
                # Shift right
                if pixel_shift < width:
                    # Copy the original frame data into the new position
                    aligned_frame[:, pixel_shift:] = frame[:, :-pixel_shift]
                    
                    # Fill the left area with the leftmost column of the frame
                    left_col = frame[:, 0:1]
                    for i in range(pixel_shift):
                        aligned_frame[:, i] = left_col[:, 0]
                else:
                    # Extreme case: shift exceeds frame width
                    # Fill the entire frame with the leftmost column color
                    left_col = frame[:, 0:1]
                    for i in range(width):
                        aligned_frame[:, i] = left_col[:, 0]
                    
            elif pixel_shift < 0:
                # Shift left
                pixel_shift_abs = abs(pixel_shift)
                if pixel_shift_abs < width:
                    # Copy the original frame data into the new position
                    aligned_frame[:, :-pixel_shift_abs] = frame[:, pixel_shift_abs:]
                    
                    # Fill the right area with the rightmost column of the frame
                    right_col = frame[:, -1:]
                    for i in range(pixel_shift_abs):
                        aligned_frame[:, width-i-1] = right_col[:, 0]
                else:
                    # Extreme case: shift exceeds frame width
                    # Fill the entire frame with the rightmost column color
                    right_col = frame[:, -1:]
                    for i in range(width):
                        aligned_frame[:, i] = right_col[:, 0]
            
            aligned_frames.append(aligned_frame)
        
        if self.verbose:
            shift_direction = "right" if pixel_shift > 0 else "left" if pixel_shift < 0 else "none"
            print(f"Horizontally aligned frames with offset {horizontal_offset}, shifted {abs(pixel_shift)} pixels {shift_direction}")
            print(f"Filled empty space with edge colors from the original frame")
            
        return aligned_frames
    
    def _scale_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Scale frames according to x_scale_factor and y_scale_factor.
        
        Args:
            frames: List of frames to scale
            
        Returns:
            List of scaled frames
        """
        x_scale = self.params["x_scale_factor"]
        y_scale = self.params["y_scale_factor"]
        
        height, width = frames[0].shape[:2]
        
        # Calculate new dimensions after scaling
        new_width = int(width * x_scale)
        new_height = int(height * y_scale)
        
        scaled_frames = []
        for frame in frames:
            # Create a resized frame
            resized = cv2.resize(frame, (new_width, new_height))
            
            if new_width <= width and new_height <= height:
                # Smaller or equal to original: create a canvas filled with edge colors
                
                # Create canvas filled with colors from the edges of the original frame
                # For the top and bottom, use the respective rows
                # For the left and right sides, use the respective columns
                canvas = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Fill with colors from the top edge (first row) of the original frame
                top_color = frame[0:1, :]
                for i in range(height):
                    canvas[i, :] = top_color
                
                # Calculate placement
                # For x: center the resized content
                x_offset = max(0, (width - new_width) // 2)
                # For y: place at the bottom with a small offset
                y_offset = max(0, (height - new_height))
                
                # Place the resized frame on the canvas
                canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
            else:
                # Larger than original: crop it to fit the canvas
                crop_x = max(0, (new_width - width) // 2)
                crop_y = max(0, (new_height - height))
                canvas = resized[crop_y:crop_y+height, crop_x:crop_x+width]
                
            scaled_frames.append(canvas)
            
        if self.verbose:
            print(f"Scaled frames with factors: x={x_scale}, y={y_scale}")
            print(f"Using edge colors to fill any empty space from scaling")
            
        return scaled_frames
    
    def _vertical_align_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Align frames vertically based on vertical_offset.
        
        Args:
            frames: List of frames to align
            
        Returns:
            List of vertically aligned frames
            
        Note:
            vertical_offset is between -1 and 1, where:
            - 0 means no change
            - Negative values shift the content up
            - Positive values shift the content down
        """
        vertical_offset = self.params["vertical_offset"]
        
        # No change needed if offset is 0
        if vertical_offset == 0:
            if self.verbose:
                print("No vertical alignment applied (vertical_offset = 0)")
            return [frame.copy() for frame in frames]
        
        # Calculate shift amount directly from vertical_offset
        # vertical_offset range: -1 to 1, where negative shifts up, positive shifts down
        
        # Convert to pixels
        height = frames[0].shape[0]
        pixel_shift = int(round(height * vertical_offset))
        
        aligned_frames = []
        for frame in frames:
            # Create a new frame filled with edge colors instead of zeros
            aligned_frame = np.zeros_like(frame)
            
            if pixel_shift > 0:
                # Shift down
                if pixel_shift < height:
                    # Copy the original frame data into the new position
                    aligned_frame[pixel_shift:, :] = frame[:-pixel_shift, :]
                    
                    # Fill the top area with the top row of the frame
                    top_row = frame[0:1, :]
                    for i in range(pixel_shift):
                        aligned_frame[i, :] = top_row
                else:
                    # Extreme case: shift exceeds frame height
                    # Fill the entire frame with the top row color
                    top_row = frame[0:1, :]
                    for i in range(height):
                        aligned_frame[i, :] = top_row
                    
            elif pixel_shift < 0:
                # Shift up
                pixel_shift_abs = abs(pixel_shift)
                if pixel_shift_abs < height:
                    # Copy the original frame data into the new position
                    aligned_frame[:-pixel_shift_abs, :] = frame[pixel_shift_abs:, :]
                    
                    # Fill the bottom area with the bottom row of the frame
                    bottom_row = frame[-1:, :]
                    for i in range(pixel_shift_abs):
                        aligned_frame[height-i-1, :] = bottom_row
                else:
                    # Extreme case: shift exceeds frame height
                    # Fill the entire frame with the bottom row color
                    bottom_row = frame[-1:, :]
                    for i in range(height):
                        aligned_frame[i, :] = bottom_row
            
            aligned_frames.append(aligned_frame)
        
        if self.verbose:
            shift_direction = "down" if pixel_shift > 0 else "up" if pixel_shift < 0 else "none"
            print(f"Vertically aligned frames with offset {vertical_offset}, shifted {abs(pixel_shift)} pixels {shift_direction}")
            print(f"Filled empty space with edge colors from the original frame")
            
        return aligned_frames
    
    def _pad_frames_to_duration(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Pad the frames to reach the target duration by repeating first and last frames.
        
        Args:
            frames: List of frames to pad
            
        Returns:
            List of padded frames
        """
        target_duration = self.params["target_duration"]
        target_frame_count = int(target_duration * self.fps)
        current_frame_count = len(frames)
        
        if current_frame_count >= target_frame_count:
            if self.verbose:
                print(f"No padding needed, current frame count {current_frame_count} >= target {target_frame_count}")
            return frames
        
        # Calculate how many frames to add
        frames_to_add = target_frame_count - current_frame_count
        
        # Split the padding between start and end
        pad_start = frames_to_add // 2
        pad_end = frames_to_add - pad_start
        
        # Create padded frames
        padded_frames = ([frames[0]] * pad_start) + frames + ([frames[-1]] * pad_end)
        
        if self.verbose:
            print(f"Padded frames from {current_frame_count} to {len(padded_frames)} frames")
            print(f"Added {pad_start} frames at the start and {pad_end} at the end")
            
        return padded_frames
    
    def _crop_resize_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply any final cropping or resizing to frames.
        
        Args:
            frames: List of frames to process
            
        Returns:
            List of processed frames
        """
        # This is a placeholder for any final resizing or cropping
        # For now, we'll just return the frames as-is
        return frames
    
    def _save_interim_video(self, frames: List[np.ndarray]) -> str:
        """
        Save the processed frames to the interim directory.
        
        Args:
            frames: List of frames to save
            
        Returns:
            Path to saved video
        """
        output_filename = f"{self.filename.split('.')[0]}_processed.mp4"
        output_path = os.path.join(self.interim_videos_dir, output_filename)
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = frames[0].shape[:2]
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
        
        # Write frames
        for frame in frames:
            out.write(frame)
        
        out.release()
        
        if self.verbose:
            print(f"Saved interim processed video to {output_path}")
            
        return output_path
    
    def _save_preprocessed_video(self, frames: List[np.ndarray]) -> str:
        """
        Save the fully preprocessed frames to the preprocessed directory.
        
        Args:
            frames: List of frames to save
            
        Returns:
            Path to saved video
        """
        output_filename = self.filename
        output_path = os.path.join(self.preprocessed_videos_dir, "videos", output_filename)
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = frames[0].shape[:2]
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
        
        # Write frames
        for frame in frames:
            out.write(frame)
        
        out.release()
        
        # Save metadata
        self._save_video_metadata(output_path)
        
        if self.verbose:
            print(f"Saved preprocessed video to {output_path}")
            
        # Update the master metadata CSV
        self._update_video_metadata_csv()
            
        return output_path
    
    def _save_intermediate_video(self, frames: List[np.ndarray], step_name: str) -> str:
        """
        Save an intermediate video for debugging.
        
        Args:
            frames: List of frames to save
            step_name: Name of the preprocessing step
            
        Returns:
            Path to saved video
        """
        debug_dir = os.path.join(self.interim_videos_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        
        output_filename = f"{self.filename.split('.')[0]}_{step_name}.mp4"
        output_path = os.path.join(debug_dir, output_filename)
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = frames[0].shape[:2]
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
        
        # Write frames
        for frame in frames:
            out.write(frame)
        
        out.release()
        
        if self.verbose:
            print(f"Saved intermediate {step_name} video to {output_path}")
            
        return output_path
    
    def _save_video_metadata(self, video_path: str) -> None:
        """
        Save metadata for a preprocessed video.
        
        Args:
            video_path: Path to the video file
        """
        def _convert_for_json(obj):
            """Recursively convert data types to JSON serializable types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return _convert_for_json(obj.tolist())
            elif isinstance(obj, dict):
                return {key: _convert_for_json(value) for key, value in obj.items()}
            elif isinstance(obj, list) or isinstance(obj, tuple):
                return [_convert_for_json(item) for item in obj]
            elif hasattr(obj, 'to_dict'):
                return _convert_for_json(obj.to_dict())
            elif hasattr(obj, 'item'):  # For numpy scalars
                return obj.item()
            else:
                return obj
        
        # Create metadata
        original_metadata = self.metadata
        preprocessing_params = self.params
            
        metadata = {
            "original_metadata": original_metadata,
            "preprocessing_params": preprocessing_params,
            "preprocessing_version": self.preprocess_version,
            "processed_filename": os.path.basename(video_path),
            "processed_path": video_path,
            "processed_frame_count": int(self.params["target_duration"] * self.fps),
            "processed_duration_sec": self.params["target_duration"],
            "processed_width": self.width,
            "processed_height": self.height,
            "fps": self.fps
        }
        
        # Convert to JSON serializable types
        metadata = _convert_for_json(metadata)
        
        # Save to individual metadata file
        metadata_filename = f"{self.filename.split('.')[0]}.json"
        metadata_path = os.path.join(self.preprocessed_videos_dir, "individual_metadata", metadata_filename)
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
            
        if self.verbose:
            print(f"Saved video metadata to {metadata_path}")
    
    def _update_video_metadata_csv(self) -> None:
        """
        Update the master CSV file with this video's metadata.
        """
        csv_path = os.path.join(self.preprocessed_videos_dir, "video_metadata.csv")
        
        # Create a new row for this video
        row = {
            "filename": self.filename,
            "original_fps": self.fps,
            "original_width": self.width,
            "original_height": self.height,
            "original_frame_count": self.frame_count,
            "original_duration_sec": self.duration_sec,
            "preprocessed_frame_count": int(self.params["target_duration"] * self.fps),
            "preprocessed_duration_sec": self.params["target_duration"],
            "start_frame": self.params["start_frame"],
            "end_frame": self.params["end_frame"],
            "horizontal_offset": self.params["horizontal_offset"],
            "vertical_offset": self.params["vertical_offset"],
            "x_scale_factor": self.params["x_scale_factor"],
            "y_scale_factor": self.params["y_scale_factor"],
            "preprocess_version": self.preprocess_version
        }
        
        # Check if the CSV exists
        if os.path.exists(csv_path):
            # Load existing data
            df = pd.read_csv(csv_path)
            
            # Check if this file is already in the CSV
            if self.filename in df["filename"].values:
                # Update the row - use DataFrame.update or replace the row completely
                idx = df.index[df["filename"] == self.filename].tolist()[0]
                df.loc[idx] = pd.Series(row)
            else:
                # Append the row
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            # Create a new DataFrame
            df = pd.DataFrame([row])
            
        # Save the CSV
        df.to_csv(csv_path, index=False)
        
        if self.verbose:
            print(f"Updated video metadata CSV at {csv_path}")
    
    # Landmark processing methods
    
    def _trim_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Trim landmarks to only include the range between start_frame and end_frame.
        
        Args:
            landmarks: Array of landmarks to trim
            
        Returns:
            Array of trimmed landmarks
        """
        start_frame = self.params["start_frame"]
        end_frame = self.params["end_frame"]
        
        if start_frame >= len(landmarks) or end_frame >= len(landmarks):
            raise ValueError(f"Invalid frame range: {start_frame} to {end_frame}, landmarks have {len(landmarks)} frames")
            
        # Make a slice of the landmarks array for the specified frame range
        trimmed = landmarks[start_frame:end_frame+1]
        
        if self.verbose:
            print(f"Trimmed landmarks from {len(landmarks)} frames to {len(trimmed)} frames")
            print(f"Frame range: {start_frame} to {end_frame}")
            
        return trimmed
    
    def _deep_copy_landmarks_frame(self, frame_landmarks):
        """
        Create a deep copy of a landmarks frame while preserving MediaPipe object structure.
        
        Args:
            frame_landmarks: Dictionary containing landmark data for a single frame
            
        Returns:
            Deep copy of the landmarks frame with preserved structure
        """
        if frame_landmarks is None:
            return None
            
        # Create a copy of the frame's landmarks
        new_frame = {}
        for key in frame_landmarks.keys():
            if key not in ['pose_landmarks', 'face_landmarks', 'left_hand_landmarks', 'right_hand_landmarks']:
                # Copy non-landmark keys directly
                new_frame[key] = frame_landmarks[key]
                continue
            
            # Skip None entries
            if frame_landmarks[key] is None:
                new_frame[key] = None
                continue
            
            # Get the landmarks for this type and create a deep copy
            landmarks_obj = frame_landmarks[key]
            new_frame[key] = copy.deepcopy(landmarks_obj)
        
        return new_frame

    def _horizontal_align_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Align landmarks horizontally based on horizontal_offset.
        
        Args:
            landmarks: Array of landmarks to align
            
        Returns:
            Array of horizontally aligned landmarks
        """
        horizontal_offset = self.params["horizontal_offset"]
        
        # No shift needed if offset is 0
        if horizontal_offset == 0:
            if self.verbose:
                print("No horizontal alignment applied to landmarks (horizontal_offset = 0)")
            return landmarks.copy()
        
        # Apply shift directly based on horizontal_offset
        aligned_landmarks = []
        
        # Process each frame's landmarks
        for frame_idx in range(len(landmarks)):
            frame_landmarks = landmarks[frame_idx]
            
            # Make a deep copy of the frame landmarks
            new_frame = self._deep_copy_landmarks_frame(frame_landmarks)
            if new_frame is None:
                aligned_landmarks.append(None)
                continue
            
            # Update the x coordinates with the offset
            for key in ['pose_landmarks', 'face_landmarks', 'left_hand_landmarks', 'right_hand_landmarks']:
                if key in new_frame and new_frame[key] is not None:
                    for i, lm in enumerate(new_frame[key].landmark):
                        # Add shift directly (positive value shifts right, negative shifts left)
                        new_x = lm.x + horizontal_offset
                        
                        # If shifted outside boundaries, push to edge
                        new_x = max(0, min(1, new_x))
                        
                        # Update the x coordinate in place
                        new_frame[key].landmark[i].x = new_x
            
            aligned_landmarks.append(new_frame)
        
        # Convert to numpy array
        aligned_landmarks = np.array(aligned_landmarks, dtype=object)
        
        if self.verbose:
            shift_direction = "right" if horizontal_offset > 0 else "left" if horizontal_offset < 0 else "none"
            print(f"Horizontally aligned landmarks with offset {horizontal_offset}, shifted {abs(horizontal_offset):.3f} {shift_direction}")
            
        return aligned_landmarks
    def _scale_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Scale landmarks according to x_scale_factor and y_scale_factor.
        
        Args:
            landmarks: Array of landmarks to scale
            
        Returns:
            Array of scaled landmarks
            
        Note:
            Scale factors > 1 make the content larger
            Scale factors < 1 make the content smaller
            The scaling is applied relative to the center point (0.5, 0.5)
        """
        x_scale = self.params["x_scale_factor"]
        y_scale = self.params["y_scale_factor"]
        
        # Center point for scaling (0.5, 0.5)
        center_x, center_y = 0.5, 0.5
        
        # Process each frame's landmarks
        scaled_landmarks = []
        for frame_idx in range(len(landmarks)):
            frame_landmarks = landmarks[frame_idx]
            
            # Make a deep copy of the frame landmarks
            new_frame = self._deep_copy_landmarks_frame(frame_landmarks)
            if new_frame is None:
                scaled_landmarks.append(None)
                continue
            
            # Update the coordinates with scaling
            for key in ['pose_landmarks', 'face_landmarks', 'left_hand_landmarks', 'right_hand_landmarks']:
                if key in new_frame and new_frame[key] is not None:
                    for i, lm in enumerate(new_frame[key].landmark):
                        # Calculate distance from center
                        dx = lm.x - center_x
                        dy = lm.y - center_y
                        
                        # Apply scale relative to center
                        # For scale < 1, the distance from center should decrease
                        # For scale > 1, the distance from center should increase
                        new_x = center_x + (dx * x_scale)
                        if lm.y <= center_y:
                            new_y = center_y + (dy * y_scale)
                        else:
                            new_y = lm.y
                        
                        # Clamp to [0,1]
                        new_x = max(0, min(1, new_x))
                        new_y = max(0, min(1, new_y))
                        
                        # Update coordinates in place
                        new_frame[key].landmark[i].x = new_x
                        new_frame[key].landmark[i].y = new_y
            
            scaled_landmarks.append(new_frame)
        
        # Convert to numpy array
        scaled_landmarks = np.array(scaled_landmarks, dtype=object)
        
        if self.verbose:
            print(f"Scaled landmarks with factors: x={x_scale}, y={y_scale}")
            
        return scaled_landmarks
    def _vertical_align_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Align landmarks vertically based on vertical_offset.
        
        Args:
            landmarks: Array of landmarks to align
            
        Returns:
            Array of vertically aligned landmarks
        """
        vertical_offset = self.params["vertical_offset"]
        
        # No shift needed if offset is 0
        if vertical_offset == 0:
            if self.verbose:
                print("No vertical alignment applied to landmarks (vertical_offset = 0)")
            return landmarks.copy()
        
        # Apply shift directly based on vertical_offset
        aligned_landmarks = []
        
        # Process each frame's landmarks
        for frame_idx in range(len(landmarks)):
            frame_landmarks = landmarks[frame_idx]
            
            # Make a deep copy of the frame landmarks
            new_frame = self._deep_copy_landmarks_frame(frame_landmarks)
            if new_frame is None:
                aligned_landmarks.append(None)
                continue
            
            # Update the y coordinates with the offset
            for key in ['pose_landmarks', 'face_landmarks', 'left_hand_landmarks', 'right_hand_landmarks']:
                if key in new_frame and new_frame[key] is not None:
                    for i, lm in enumerate(new_frame[key].landmark):
                        # Add shift directly (positive value shifts down, negative shifts up)
                        new_y = lm.y + vertical_offset
                        
                        # If shifted outside boundaries, push to edge
                        new_y = max(0, min(1, new_y))
                        
                        # Update y coordinate in place
                        new_frame[key].landmark[i].y = new_y
            
            aligned_landmarks.append(new_frame)
        
        # Convert to numpy array
        aligned_landmarks = np.array(aligned_landmarks, dtype=object)
        
        if self.verbose:
            shift_direction = "down" if vertical_offset > 0 else "up" if vertical_offset < 0 else "none"
            print(f"Vertically aligned landmarks with offset {vertical_offset}, shifted {abs(vertical_offset):.3f} {shift_direction}")
            
        return aligned_landmarks
    
    def _pad_landmarks_to_duration(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Pad the landmarks to reach the target duration by repeating first and last frames.
        
        Args:
            landmarks: Array of landmarks to pad
            
        Returns:
            Array of padded landmarks
        """
        target_duration = self.params["target_duration"]
        target_frame_count = int(target_duration * self.fps)
        current_frame_count = len(landmarks)
        
        if current_frame_count >= target_frame_count:
            if self.verbose:
                print(f"No padding needed, current frame count {current_frame_count} >= target {target_frame_count}")
            return landmarks
        
        # Calculate how many frames to add
        frames_to_add = target_frame_count - current_frame_count
        
        # Split the padding between start and end
        pad_start = frames_to_add // 2
        pad_end = frames_to_add - pad_start
        
        if self.verbose:
            print(f"Padding landmarks from {current_frame_count} to {target_frame_count} frames")
            print(f"Adding {pad_start} frames at the start and {pad_end} at the end")
        
        # Create padding arrays by repeating first and last frame
        if len(landmarks) > 0:
            start_padding = [landmarks[0]] * pad_start
            end_padding = [landmarks[-1]] * pad_end
            
            # Concatenate with original landmarks
            padded_landmarks = start_padding + list(landmarks) + end_padding
            padded_landmarks = np.array(padded_landmarks, dtype=object)
        else:
            # Handle case with no landmarks
            padded_landmarks = landmarks
            
        return padded_landmarks
    
    def _crop_resize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Apply any final adjustments to landmarks coordinates.
        
        Args:
            landmarks: Array of landmarks to process
            
        Returns:
            Array of processed landmarks
        """
        # This is a placeholder for any final adjustments
        # For now, we'll just return the landmarks as-is
        return landmarks
    
    def _save_preprocessed_landmarks(self, landmarks: np.ndarray) -> str:
        """
        Save the fully preprocessed landmarks to the preprocessed directory.
        
        Args:
            landmarks: Array of landmarks to save
            
        Returns:
            Path to saved landmarks
        """
        output_filename = f"{self.filename.split('.')[0]}.npy"
        output_path = os.path.join(self.preprocessed_landmarks_dir, "landmarks", output_filename)
        
        # Save landmarks
        np.save(output_path, landmarks)
        
        # Save metadata
        self._save_landmarks_metadata(output_path)
        
        if self.verbose:
            print(f"Saved preprocessed landmarks to {output_path}")
            
        # Update the master metadata CSV
        self._update_landmarks_metadata_csv()
            
        return output_path
    
    def _save_landmarks_metadata(self, landmarks_path: str) -> None:
        """
        Save metadata for preprocessed landmarks.
        
        Args:
            landmarks_path: Path to the landmarks file
        """
        def _convert_for_json(obj):
            """Recursively convert data types to JSON serializable types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return _convert_for_json(obj.tolist())
            elif isinstance(obj, dict):
                return {key: _convert_for_json(value) for key, value in obj.items()}
            elif isinstance(obj, list) or isinstance(obj, tuple):
                return [_convert_for_json(item) for item in obj]
            elif hasattr(obj, 'to_dict'):
                return _convert_for_json(obj.to_dict())
            elif hasattr(obj, 'item'):  # For numpy scalars
                return obj.item()
            else:
                return obj
                
        # Create metadata
        original_metadata = self.metadata
        preprocessing_params = self.params
        
        metadata = {
            "original_metadata": original_metadata,
            "preprocessing_params": preprocessing_params,
            "preprocessing_version": self.preprocess_version,
            "processed_filename": os.path.basename(landmarks_path),
            "processed_path": landmarks_path,
            "processed_frame_count": int(self.params["target_duration"] * self.fps),
            "processed_duration_sec": self.params["target_duration"],
            "fps": self.fps
        }
        
        # Convert to JSON serializable types
        metadata = _convert_for_json(metadata)
        
        # Save to individual metadata file
        metadata_filename = f"{self.filename.split('.')[0]}.json"
        metadata_path = os.path.join(self.preprocessed_landmarks_dir, "individual_metadata", metadata_filename)
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
            
        if self.verbose:
            print(f"Saved landmarks metadata to {metadata_path}")
    
    def _update_landmarks_metadata_csv(self) -> None:
        """
        Update the master CSV file with this landmarks' metadata.
        """
        csv_path = os.path.join(self.preprocessed_landmarks_dir, "landmarks_metadata.csv")
        
        # Create a new row for this landmarks file
        row = {
            "filename": self.filename,
            "original_fps": self.fps,
            "original_frame_count": self.frame_count,
            "original_duration_sec": self.duration_sec,
            "preprocessed_frame_count": int(self.params["target_duration"] * self.fps),
            "preprocessed_duration_sec": self.params["target_duration"],
            "start_frame": self.params["start_frame"],
            "end_frame": self.params["end_frame"],
            "horizontal_offset": self.params["horizontal_offset"],
            "vertical_offset": self.params["vertical_offset"],
            "x_scale_factor": self.params["x_scale_factor"],
            "y_scale_factor": self.params["y_scale_factor"],
            "preprocess_version": self.preprocess_version
        }
        
        # Check if the CSV exists
        if os.path.exists(csv_path):
            # Load existing data
            df = pd.read_csv(csv_path)
            
            # Check if this file is already in the CSV
            if self.filename in df["filename"].values:
                # Update the row - use DataFrame.update or replace the row completely
                idx = df.index[df["filename"] == self.filename].tolist()[0]
                df.loc[idx] = pd.Series(row)
            else:
                # Append the row
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            # Create a new DataFrame
            df = pd.DataFrame([row])
            
        # Save the CSV
        df.to_csv(csv_path, index=False)
        
        if self.verbose:
            print(f"Updated landmarks metadata CSV at {csv_path}")

    def _debug_landmarks_format(self, landmarks1, landmarks2, frame_idx=0, label1="Original", label2="Modified"):
        """
        Debug function to compare landmark formats and check for data loss between two landmark arrays.
        
        Args:
            landmarks1: First landmarks array
            landmarks2: Second landmarks array
            frame_idx: Frame index to compare (default: 0)
            label1: Label for first landmarks (default: "Original")
            label2: Label for second landmarks (default: "Modified")
        """
        if frame_idx >= len(landmarks1) or frame_idx >= len(landmarks2):
            print(f"Invalid frame index: {frame_idx}, arrays have lengths {len(landmarks1)} and {len(landmarks2)}")
            return
        
        frame1 = landmarks1[frame_idx]
        frame2 = landmarks2[frame_idx]
        
        print(f"Comparing {label1} vs {label2} landmarks for frame {frame_idx}")
        print(f"{label1} frame type: {type(frame1)}")
        print(f"{label2} frame type: {type(frame2)}")
        
        # Check keys in dictionary
        if isinstance(frame1, dict) and isinstance(frame2, dict):
            print(f"\nKeys in {label1}: {sorted(frame1.keys())}")
            print(f"Keys in {label2}: {sorted(frame2.keys())}")
            
            # Compare landmark types
            for landmark_type in ['pose_landmarks', 'face_landmarks', 'left_hand_landmarks', 'right_hand_landmarks']:
                if landmark_type in frame1 and landmark_type in frame2:
                    lm1 = frame1[landmark_type]
                    lm2 = frame2[landmark_type]
                    
                    print(f"\n{landmark_type}:")
                    print(f"  {label1} type: {type(lm1)}")
                    print(f"  {label2} type: {type(lm2)}")
                    
                    if lm1 is None or lm2 is None:
                        print(f"  One of the {landmark_type} is None")
                        continue
                    
                    # Check attributes
                    lm1_attrs = dir(lm1)
                    lm2_attrs = dir(lm2)
                    
                    print(f"  {label1} attributes count: {len(lm1_attrs)}")
                    print(f"  {label2} attributes count: {len(lm2_attrs)}")
                    
                    # Check for missing attributes
                    missing_attrs = [attr for attr in lm1_attrs if attr not in lm2_attrs]
                    if missing_attrs:
                        print(f"  Attributes in {label1} but not in {label2}: {missing_attrs}")
                    
                    # Compare landmark count
                    if hasattr(lm1, 'landmark') and hasattr(lm2, 'landmark'):
                        print(f"  {label1} landmark count: {len(lm1.landmark)}")
                        print(f"  {label2} landmark count: {len(lm2.landmark)}")
                        
                        # Check first landmark
                        if len(lm1.landmark) > 0 and len(lm2.landmark) > 0:
                            l1 = lm1.landmark[0]
                            l2 = lm2.landmark[0]
                            
                            print(f"  First landmark in {label1}: (x={l1.x:.4f}, y={l1.y:.4f}, z={l1.z:.4f})")
                            print(f"  First landmark in {label2}: (x={l2.x:.4f}, y={l2.y:.4f}, z={l2.z:.4f})")
                            
                            # Check attributes of individual landmarks
                            l1_attrs = dir(l1)
                            l2_attrs = dir(l2)
                            missing_landmark_attrs = [attr for attr in l1_attrs if attr not in l2_attrs]
                            if missing_landmark_attrs:
                                print(f"  Landmark attributes in {label1} but not in {label2}: {missing_landmark_attrs}")
