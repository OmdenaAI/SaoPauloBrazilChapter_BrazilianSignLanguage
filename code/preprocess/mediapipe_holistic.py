import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt

class MediaPipeHolistic:
    def __init__(self,
                 static_image_mode: bool = False,
                 model_complexity: int = 1,
                 smooth_landmarks: bool = True,
                 enable_segmentation: bool = False,
                 smooth_segmentation: bool = True,
                 refine_face_landmarks: bool = True,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe Holistic detector for combined face, pose, and hand detection.
        
        Args:
            static_image_mode: Whether to treat input images as a video stream (False) or independent images (True).
                             For video processing, set to False to enable tracking instead of detection per image.
            model_complexity: Complexity of the pose landmark model: 0, 1 or 2. 
                            Landmark accuracy as well as inference latency generally go up with the model complexity.
                            0: Fastest but least accurate
                            1: Good balance of speed and accuracy
                            2: Slowest but most accurate
            smooth_landmarks: Whether to filter landmarks across different input images to reduce jitter.
                            Should be disabled for static images.
            enable_segmentation: Whether to generate segmentation mask.
            smooth_segmentation: Whether to filter segmentation across different input images to reduce jitter.
                               Should be disabled for static images.
            refine_face_landmarks: Whether to further refine the landmark coordinates around the eyes and lips, and output additional landmarks around the irises.
            min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for person detection to be considered successful.
            min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) for the pose landmarks to be considered tracked 
                                  successfully. Only used when static_image_mode is False.
        """
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            smooth_segmentation=smooth_segmentation,
            refine_face_landmarks=refine_face_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )    

    def process_frame(self, frame: np.ndarray, timestamp_ms: int) -> Dict:
        """
        Process a single frame and detect holistic landmarks.
        
        Args:
            frame: Input frame in BGR format
            timestamp_ms: Timestamp of the frame in milliseconds
        
        Returns:
            Dictionary containing detection results with the following keys:
            - face_landmarks: 468 face landmarks if detected
            - pose_landmarks: 33 pose landmarks if detected
            - left_hand_landmarks: 21 left hand landmarks if detected
            - right_hand_landmarks: 21 right hand landmarks if detected
            - timestamp_ms: Frame timestamp
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with holistic model
        results = self.holistic.process(frame_rgb)  # Pass numpy array directly
        
        # Package results
        return {
            'face_landmarks': results.face_landmarks,  # 468 landmarks
            'pose_landmarks': results.pose_landmarks,  # 33 landmarks
            'left_hand_landmarks': results.left_hand_landmarks,  # 21 landmarks
            'right_hand_landmarks': results.right_hand_landmarks,  # 21 landmarks
            'pose_world_landmarks': results.pose_world_landmarks,  # 3D pose landmarks
            'timestamp_ms': timestamp_ms
        }

    def process_video(self, video_path: str, output_path: Optional[str] = None) -> List[Dict]:
        """
        Process a video file and detect holistic landmarks in each frame.
        No post-processing (filtering, centering, scaling) is applied.
        
        Args:
            video_path: Path to input video file
            output_path: Optional path to save the processed video with visualizations
        
        Returns:
            List of dictionaries containing detection results for each frame
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_time_ms = int(1000 / fps)
        all_results = []
        current_time_ms = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                results = self.process_frame(frame, current_time_ms)
                all_results.append(results)
                
                # Visualize if output path is provided
                if output_path:
                    annotated_frame = self.draw_landmarks_on_frame(frame, results)
                    out.write(annotated_frame)
                
                current_time_ms += frame_time_ms
        
        finally:
            cap.release()
            if output_path:
                out.release()
        
        return all_results
    
    def __del__(self):
        """Clean up resources"""
        self.holistic.close()

    ### Visualization
    
    def draw_landmarks_on_frame(self, image: np.ndarray, results: Dict) -> np.ndarray:
        """
        Draw landmarks and connections on the image.
        
        Args:
            image: Input image in BGR format
            results: Dictionary containing detection results
        
        Returns:
            Image with landmarks drawn
        """
        # Create a copy of the image to draw on
        annotated_image = image.copy()
        
        # Draw face landmarks
        if results['face_landmarks']:
            self.mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=results['face_landmarks'],
                connections=self.mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            # Draw face contours
            self.mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=results['face_landmarks'],
                connections=self.mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
            )
        
        # Draw pose landmarks
        if results['pose_landmarks']:
            self.mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=results['pose_landmarks'],
                connections=self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Draw left hand landmarks
        if results['left_hand_landmarks']:
            self.mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=results['left_hand_landmarks'],
                connections=self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        # Draw right hand landmarks
        if results['right_hand_landmarks']:
            self.mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=results['right_hand_landmarks'],
                connections=self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        return annotated_image

    def draw_landmarks_on_video(self, 
                              video_path: str,
                              results_list: List[Dict],
                              output_path: str) -> None:
        """
        Process a video and draw landmarks on each frame.
        
        Args:
            video_path: Path to input video file
            results_list: List of dictionaries containing detection results for each frame
            output_path: Path to save the processed video with landmarks drawn
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        try:
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx >= len(results_list):
                    break
                
                # Get results for this frame
                results = results_list[frame_idx]
                
                # Draw landmarks on frame
                annotated_frame = self.draw_landmarks_on_frame(frame, results)
                
                # Write to output video
                out.write(annotated_frame)
                
                frame_idx += 1
        
        finally:
            cap.release()
            out.release()
        print(f"Landmarks drawn video saved to {output_path}")

    def draw_landmarks_3d(self, results: Dict, figsize: Tuple[int, int] = (10, 10),
                         elev: float = 0, azim: float = 0,
                         xlim: Optional[Tuple[float, float]] = None,
                         ylim: Optional[Tuple[float, float]] = None,
                         zlim: Optional[Tuple[float, float]] = None) -> plt.Figure:
        """
        Draw landmarks in 3D using matplotlib.
        
        Args:
            results: Dictionary containing detection results
            figsize: Figure size in inches (width, height)
            elev: Elevation angle in degrees (vertical rotation)
                 0 = front view, positive = looking down, negative = looking up
            azim: Azimuth angle in degrees (horizontal rotation)
                 0 = front view, 90 = right side, -90 = left side, 180 = back
            xlim: Optional tuple of (min, max) for x-axis limits
            ylim: Optional tuple of (min, max) for y-axis limits
            zlim: Optional tuple of (min, max) for z-axis limits
        
        Returns:
            Matplotlib figure with 3D visualization
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Set viewing angle
        ax.view_init(elev=elev, azim=azim)
        
        # Draw pose landmarks if available
        if results['pose_world_landmarks']:
            landmarks = results['pose_world_landmarks'].landmark
            
            # Transform coordinates to make front view the default
            # MediaPipe: x = right/left, y = up/down, z = forward/backward
            # Desired: x = right/left, y = forward/backward, z = up/down
            x = [lm.x for lm in landmarks]  # x stays the same (right/left)
            y = [-lm.z for lm in landmarks]  # negative z becomes y (forward/backward)
            z = [-lm.y for lm in landmarks]  # negative y becomes z (up/down)
            
            # Draw hands first if available, using pose landmarks as reference points
            hand_landmarks = {
                'left_hand_landmarks': (
                    '#FF3366', 'Left Hand',  # Bright pink
                    landmarks[15],  # Left wrist landmark
                    landmarks[13]   # Left elbow landmark for orientation
                ),
                'right_hand_landmarks': (
                    '#33FF66', 'Right Hand',  # Bright green
                    landmarks[16],  # Right wrist landmark
                    landmarks[14]   # Right elbow landmark for orientation
                )
            }
            
            for hand_key, (color, label, wrist_ref, elbow_ref) in hand_landmarks.items():
                if results[hand_key]:
                    hand_landmarks = results[hand_key].landmark
                    
                    # Transform hand coordinates
                    hand_x = []
                    hand_y = []
                    hand_z = []
                    scale = 1  # Increased scale factor for hand size
                    
                    # Transform wrist reference point
                    wrist_pos = np.array([wrist_ref.x, -wrist_ref.z, -wrist_ref.y])
                    
                    for lm in hand_landmarks:
                        # Center hand at wrist and scale
                        rel_x = (lm.x - hand_landmarks[0].x) * scale
                        rel_y = -(lm.z - hand_landmarks[0].z) * scale  # Transform z to y
                        rel_z = -(lm.y - hand_landmarks[0].y) * scale  # Transform y to z
                        
                        # Add to wrist position
                        hand_x.append(wrist_pos[0] + rel_x)
                        hand_y.append(wrist_pos[1] + rel_y)
                        hand_z.append(wrist_pos[2] + rel_z)
                    
                    # Plot hand points with larger markers
                    ax.scatter(hand_x, hand_y, hand_z, 
                             c=color, marker='o', s=100,  
                             label=label, zorder=3)  # Higher zorder to stay on top
                    
                    # Draw hand connections with thicker lines
                    for connection in self.mp_holistic.HAND_CONNECTIONS:
                        start_idx = connection[0]
                        end_idx = connection[1]
                        ax.plot([hand_x[start_idx], hand_x[end_idx]],
                               [hand_y[start_idx], hand_y[end_idx]],
                               [hand_z[start_idx], hand_z[end_idx]], 
                               color, linewidth=2, zorder=2)  # Thicker lines
            
            # Then draw pose skeleton
            # Plot pose points with smaller markers
            ax.scatter(x, y, z, c='#3366FF', marker='o', s=50,  # Bright blue, smaller circles
                      label='Pose', zorder=1)
            
            # Draw pose connections
            for connection in self.mp_holistic.POSE_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                ax.plot([x[start_idx], x[end_idx]],
                       [y[start_idx], y[end_idx]],
                       [z[start_idx], z[end_idx]], 
                       color='#CCCCCC', linewidth=1, zorder=0)  # Light gray, thinner lines
        
        # Set axis labels with correct orientation description
        ax.set_xlabel('X (Right/Left)')
        ax.set_ylabel('Y (Forward/Backward)')
        ax.set_zlabel('Z (Up/Down)')
        
        # Set axis limits if provided
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        if zlim:
            ax.set_zlim(zlim)
        
        # Add legend
        ax.legend()
        
        # Equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        plt.tight_layout()
        return fig
    
    ### Post Processing

    def get_frame_horizontal_offset(self, results: Dict, 
                           use_shoulders: bool = True,
                           use_face: bool = True,
                           use_hips: bool = False) -> Dict[str, float]:
        """
        Calculate the horizontal offset for each specified reference point.
        
        Args:
            results: Dictionary containing detection results
            use_shoulders: Whether to calculate shoulder midpoint offset
            use_face: Whether to calculate face center offset
            use_hips: Whether to calculate hip midpoint offset
        
        Returns:
            Dictionary containing horizontal offsets (0.0 to 1.0, where 0.5 is center) for each enabled reference point:
            {
                'shoulders': float,  # Only if use_shoulders is True
                'face': float,       # Only if use_face is True and face landmarks are detected
                'hips': float        # Only if use_hips is True
            }
        """
        offsets = {}
        
        if not results['pose_landmarks']:
            return offsets
            
        pose_landmarks = results['pose_landmarks'].landmark
        
        if use_shoulders:
            left_shoulder = pose_landmarks[11]
            right_shoulder = pose_landmarks[12]
            offsets['shoulders'] = (left_shoulder.x + right_shoulder.x) / 2
        
        if use_face and results['face_landmarks']:
            face_landmarks = results['face_landmarks'].landmark
            nose_tip = face_landmarks[5]  # Nose tip
            offsets['face'] = nose_tip.x
        
        if use_hips:
            left_hip = pose_landmarks[23]
            right_hip = pose_landmarks[24]
            offsets['hips'] = (left_hip.x + right_hip.x) / 2
        
        return offsets

    def get_video_horizontal_offsets(self, results_list: List[Dict],
                                  use_shoulders: bool = True,
                                  use_face: bool = True,
                                  use_hips: bool = False) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for horizontal offsets across multiple frames.
        
        Args:
            results_list: List of dictionaries containing detection results for each frame
            use_shoulders: Whether to calculate shoulder midpoint offset
            use_face: Whether to calculate face center offset
            use_hips: Whether to calculate hip midpoint offset
        
        Returns:
            Dictionary containing statistics for each reference point's horizontal offset:
            {
                'reference_point': {
                    'mean': float,
                    'median': float,
                    'max': float,
                    'min': float
                }
            }
        """
        # Initialize storage for each reference point
        offsets_by_point = {}
        
        # Get offsets for each frame
        for results in results_list:
            frame_offsets = self.get_frame_horizontal_offset(
                results,
                use_shoulders=use_shoulders,
                use_face=use_face,
                use_hips=use_hips
            )
            
            # Store offsets for each reference point
            for point, offset in frame_offsets.items():
                if point not in offsets_by_point:
                    offsets_by_point[point] = []
                offsets_by_point[point].append(offset)
        
        # Calculate statistics for each reference point
        stats = {}
        for point, offsets in offsets_by_point.items():
            if offsets:  # Only calculate if we have measurements
                stats[point] = {
                    'mean': np.mean(offsets),
                    'median': np.median(offsets),
                    'max': np.max(offsets),
                    'min': np.min(offsets)
                }
        
        return stats

    def get_frame_vertical_offset(self, results: Dict, 
                           use_shoulders: bool = True,
                           use_face: bool = True,
                           use_hips: bool = False) -> Dict[str, float]:
        """
        Calculate the vertical offset for each specified reference point.
        
        Args:
            results: Dictionary containing detection results
            use_shoulders: Whether to calculate shoulder midpoint offset
            use_face: Whether to calculate face center offset
            use_hips: Whether to calculate hip midpoint offset
        
        Returns:
            Dictionary containing vertical offsets (0.0 to 1.0, where 0.5 is center) for each enabled reference point:
            {
                'shoulders': float,  # Only if use_shoulders is True
                'face': float,       # Only if use_face is True and face landmarks are detected
                'hips': float        # Only if use_hips is True
            }
        """
        offsets = {}
        
        if not results['pose_landmarks']:
            return offsets
            
        pose_landmarks = results['pose_landmarks'].landmark
        
        if use_shoulders:
            left_shoulder = pose_landmarks[11]
            right_shoulder = pose_landmarks[12]
            offsets['shoulders'] = (left_shoulder.y + right_shoulder.y) / 2
        
        if use_face and results['face_landmarks']:
            face_landmarks = results['face_landmarks'].landmark
            nose_tip = face_landmarks[5]  # Nose tip
            offsets['face'] = nose_tip.y
        
        if use_hips:
            left_hip = pose_landmarks[23]
            right_hip = pose_landmarks[24]
            offsets['hips'] = (left_hip.y + right_hip.y) / 2
        
        return offsets

    def shift_landmarks_horizontally(self, results: Dict, offset: float) -> Dict:
        """
        Shift all landmarks horizontally by a given offset.
        
        Args:
            results: Dictionary containing detection results
            offset: Horizontal offset to apply (in normalized coordinates, 0.0 to 1.0)
                   Positive values shift right, negative values shift left
        
        Returns:
            Dictionary containing shifted landmark coordinates
        """
        shifted_results = results.copy()
        
        # Helper function to shift a single landmark
        def shift_landmark(lm):
            # Add offset to x coordinate, clamping between 0 and 1
            new_x = max(0.0, min(1.0, lm.x + offset))
            return type(lm)(x=new_x, y=lm.y, z=lm.z)
        
        # Shift pose landmarks
        if results['pose_landmarks']:
            shifted_landmarks = []
            for lm in results['pose_landmarks'].landmark:
                shifted_landmarks.append(shift_landmark(lm))
            shifted_results['pose_landmarks'] = type(results['pose_landmarks'])(landmark=shifted_landmarks)
        
        # Shift face landmarks
        if results['face_landmarks']:
            shifted_landmarks = []
            for lm in results['face_landmarks'].landmark:
                shifted_landmarks.append(shift_landmark(lm))
            shifted_results['face_landmarks'] = type(results['face_landmarks'])(landmark=shifted_landmarks)
        
        # Shift hand landmarks
        if results['left_hand_landmarks']:
            shifted_landmarks = []
            for lm in results['left_hand_landmarks'].landmark:
                shifted_landmarks.append(shift_landmark(lm))
            shifted_results['left_hand_landmarks'] = type(results['left_hand_landmarks'])(landmark=shifted_landmarks)
        
        if results['right_hand_landmarks']:
            shifted_landmarks = []
            for lm in results['right_hand_landmarks'].landmark:
                shifted_landmarks.append(shift_landmark(lm))
            shifted_results['right_hand_landmarks'] = type(results['right_hand_landmarks'])(landmark=shifted_landmarks)
        
        return shifted_results

    def shift_landmarks_series_horizontally(self, results_list: List[Dict], offset: float) -> List[Dict]:
        """
        Shift all landmarks in a series of results horizontally by a given offset.
        
        Args:
            results_list: List of dictionaries containing detection results for each frame
            offset: Horizontal offset to apply (in normalized coordinates, 0.0 to 1.0)
                   Positive values shift right, negative values shift left
        
        Returns:
            List of dictionaries containing shifted landmark coordinates
        """
        return [self.shift_landmarks_horizontally(results, offset) for results in results_list]

    def get_video_vertical_offsets(self, results_list: List[Dict],
                                use_shoulders: bool = True,
                                use_face: bool = True,
                                use_hips: bool = False) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for vertical offsets across multiple frames.
        
        Args:
            results_list: List of dictionaries containing detection results for each frame
            use_shoulders: Whether to calculate shoulder midpoint offset
            use_face: Whether to calculate face center offset
            use_hips: Whether to calculate hip midpoint offset
        
        Returns:
            Dictionary containing statistics for each reference point's vertical offset:
            {
                'reference_point': {
                    'mean': float,
                    'median': float,
                    'max': float,
                    'min': float
                }
            }
        """
        # Initialize storage for each reference point
        offsets_by_point = {}
        
        # Get offsets for each frame
        for results in results_list:
            frame_offsets = self.get_frame_vertical_offset(
                results,
                use_shoulders=use_shoulders,
                use_face=use_face,
                use_hips=use_hips
            )
            
            # Store offsets for each reference point
            for point, offset in frame_offsets.items():
                if point not in offsets_by_point:
                    offsets_by_point[point] = []
                offsets_by_point[point].append(offset)
        
        # Calculate statistics for each reference point
        stats = {}
        for point, offsets in offsets_by_point.items():
            if offsets:  # Only calculate if we have measurements
                stats[point] = {
                    'mean': np.mean(offsets),
                    'median': np.median(offsets),
                    'max': np.max(offsets),
                    'min': np.min(offsets)
                }
        
        return stats
    
    def shift_landmarks_vertically(self, results: Dict, offset: float) -> Dict:
        """
        Shift all landmarks vertically by a given offset.
        
        Args:
            results: Dictionary containing detection results
            offset: Vertical offset to apply (in normalized coordinates, 0.0 to 1.0)
                   Positive values shift down, negative values shift up
        
        Returns:
            Dictionary containing shifted landmark coordinates
        """
        shifted_results = results.copy()
        
        # Helper function to shift a single landmark
        def shift_landmark(lm):
            # Add offset to y coordinate, clamping between 0 and 1
            new_y = max(0.0, min(1.0, lm.y + offset))
            return type(lm)(x=lm.x, y=new_y, z=lm.z)
        
        # Shift pose landmarks
        if results['pose_landmarks']:
            shifted_landmarks = []
            for lm in results['pose_landmarks'].landmark:
                shifted_landmarks.append(shift_landmark(lm))
            shifted_results['pose_landmarks'] = type(results['pose_landmarks'])(landmark=shifted_landmarks)
        
        # Shift face landmarks
        if results['face_landmarks']:
            shifted_landmarks = []
            for lm in results['face_landmarks'].landmark:
                shifted_landmarks.append(shift_landmark(lm))
            shifted_results['face_landmarks'] = type(results['face_landmarks'])(landmark=shifted_landmarks)
        
        # Shift hand landmarks
        if results['left_hand_landmarks']:
            shifted_landmarks = []
            for lm in results['left_hand_landmarks'].landmark:
                shifted_landmarks.append(shift_landmark(lm))
            shifted_results['left_hand_landmarks'] = type(results['left_hand_landmarks'])(landmark=shifted_landmarks)
        
        if results['right_hand_landmarks']:
            shifted_landmarks = []
            for lm in results['right_hand_landmarks'].landmark:
                shifted_landmarks.append(shift_landmark(lm))
            shifted_results['right_hand_landmarks'] = type(results['right_hand_landmarks'])(landmark=shifted_landmarks)
        
        return shifted_results

    def shift_landmarks_series_vertically(self, results_list: List[Dict], offset: float) -> List[Dict]:
        """
        Shift all landmarks in a series of results vertically by a given offset.
        
        Args:
            results_list: List of dictionaries containing detection results for each frame
            offset: Vertical offset to apply (in normalized coordinates, 0.0 to 1.0)
                   Positive values shift down, negative values shift up
        
        Returns:
            List of dictionaries containing shifted landmark coordinates
        """
        return [self.shift_landmarks_vertically(results, offset) for results in results_list]


    def get_frame_landmark_measurements(self, results: Dict) -> Dict[str, float]:
        """
        Calculate various measurements between landmarks.
        
        Args:
            results: Dictionary containing detection results
        
        Returns:
            Dictionary of measurements:
            - shoulder_width: Distance between left and right shoulders
            - face_width: Approximate face width (distance between left and right face edges)
            - face_height: Approximate face height (distance between top of head and chin)
            - left_arm_length: Distance from left shoulder to left wrist
            - right_arm_length: Distance from right shoulder to right wrist
            - hip_width: Distance between left and right hips
            - shoulder_to_hip: Distance from shoulder midpoint to hip midpoint
            - top_head_to_shoulders: Vertical distance from top of head to shoulder midpoint
            - nose_to_shoulders: Vertical distance from nose tip to shoulder midpoint
            - chin_to_shoulders: Vertical distance from chin to shoulder midpoint
        """
        measurements = {}
        
        if not results['pose_landmarks']:
            return measurements
            
        pose_landmarks = results['pose_landmarks'].landmark
        
        # Get shoulder width and midpoint
        left_shoulder = pose_landmarks[11]
        right_shoulder = pose_landmarks[12]
        shoulder_mid = np.array([
            (left_shoulder.x + right_shoulder.x) / 2,
            (left_shoulder.y + right_shoulder.y) / 2
        ])
        measurements['shoulder_width'] = np.sqrt(
            (left_shoulder.x - right_shoulder.x)**2 + 
            (left_shoulder.y - right_shoulder.y)**2
        )
        
        # Get hip width
        left_hip = pose_landmarks[23]
        right_hip = pose_landmarks[24]
        measurements['hip_width'] = np.sqrt(
            (left_hip.x - right_hip.x)**2 + 
            (left_hip.y - right_hip.y)**2
        )
        
        # Get shoulder to hip distance
        hip_mid = np.array([
            (left_hip.x + right_hip.x) / 2,
            (left_hip.y + right_hip.y) / 2
        ])
        measurements['shoulder_to_hip'] = np.linalg.norm(shoulder_mid - hip_mid)
        
        # Get arm lengths
        left_wrist = pose_landmarks[15]
        right_wrist = pose_landmarks[16]
        measurements['left_arm_length'] = np.sqrt(
            (left_shoulder.x - left_wrist.x)**2 + 
            (left_shoulder.y - left_wrist.y)**2
        )
        measurements['right_arm_length'] = np.sqrt(
            (right_shoulder.x - right_wrist.x)**2 + 
            (right_shoulder.y - right_wrist.y)**2
        )
        
        # Get face measurements if available
        if results['face_landmarks']:
            face_landmarks = results['face_landmarks'].landmark
            
            # Face width (distance between left and right face edges)
            left_face = face_landmarks[234]  # Left face edge
            right_face = face_landmarks[454]  # Right face edge
            measurements['face_width'] = np.sqrt(
                (left_face.x - right_face.x)**2 + 
                (left_face.y - right_face.y)**2
            )
            
            # Face height (distance between top of head and chin)
            top_head = face_landmarks[10]  # Top of head
            chin = face_landmarks[152]  # Chin
            measurements['face_height'] = np.sqrt(
                (top_head.x - chin.x)**2 + 
                (top_head.y - chin.y)**2
            )
            
            # Vertical measurements from face to shoulder midpoint
            nose_tip = face_landmarks[5]  # Nose tip
            
            # Calculate vertical distances (only y-coordinate difference)
            measurements['top_head_to_shoulders'] = abs(top_head.y - shoulder_mid[1])
            measurements['nose_to_shoulders'] = abs(nose_tip.y - shoulder_mid[1])
            measurements['chin_to_shoulders'] = abs(chin.y - shoulder_mid[1])
        
        return measurements

    def get_video_landmark_measurements(self, results_list: List[Dict]) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for all measurements across multiple frames.
        
        Args:
            results_list: List of dictionaries containing detection results for each frame
        
        Returns:
            Dictionary containing statistics for each measurement:
            {
                'measurement_name': {
                    'mean': float,
                    'median': float,
                    'max': float
                }
            }
        """
        # Initialize measurement storage
        all_measurements = {}
        
        # Process each frame
        for results in results_list:
            measurements = self.get_frame_landmark_measurements(results)
            
            # Store measurements
            for name, value in measurements.items():
                if name not in all_measurements:
                    all_measurements[name] = []
                all_measurements[name].append(value)
        
        # Calculate statistics
        stats = {}
        for name, values in all_measurements.items():
            if values:  # Only calculate if we have measurements
                stats[name] = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'max': np.max(values)
                }
        
        return stats

    def calculate_scale_factors(self, 
                              measurements: Dict[str, Dict[str, float]],
                              reference_measurements: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate x and y scale factors based on reference measurements.
        
        Args:
            measurements: Dictionary containing statistics for each measurement
            reference_measurements: Dictionary containing target values for each measurement
        
        Returns:
            Dictionary containing scale factors for x and y dimensions:
            {
                'x': float,  # Scale factor for horizontal dimension
                'y': float   # Scale factor for vertical dimension
            }
        """
        # Define which measurements are primarily horizontal vs vertical
        horizontal_measurements = {
            'shoulder_width': 1.0,
            'hip_width': 1.0,
            'face_width': 1.0
        }
        
        vertical_measurements = {
            'shoulder_to_hip': 1.0,
            'face_height': 1.0,
            'left_arm_length': 1.0,
            'right_arm_length': 1.0,
            'top_head_to_shoulders': 1.0,
            'nose_to_shoulders': 1.0,
            'chin_to_shoulders': 1.0
        }
        
        # Calculate scale factors for each measurement
        x_scales = []
        y_scales = []
        
        # Use all keys from reference_measurements
        for measurement in reference_measurements.keys():
            if measurement not in measurements:
                continue
                
            # Get the mean value from our measurements
            measured_value = measurements[measurement]['mean']
            if measured_value == 0:  # Avoid division by zero
                continue
                
            # Calculate scale factor
            scale = reference_measurements[measurement] / measured_value
            
            # Add to appropriate list based on measurement type
            if measurement in horizontal_measurements:
                x_scales.append(scale)
            elif measurement in vertical_measurements:
                y_scales.append(scale)
        
        # Calculate final scale factors (use median to be robust to outliers)
        return {
            'x': np.median(x_scales) if x_scales else 1.0,
            'y': np.median(y_scales) if y_scales else 1.0
        }

    def scale_frame(self, 
                   frame: np.ndarray,
                   scale_factors: Dict[str, float],
                   center_point: Tuple[float, float] = (0.5, 0.5)) -> np.ndarray:
        """
        Scale a frame around a center point using different factors for x and y dimensions,
        while maintaining the original frame size.
        For y-dimension, only scales upward (never crops/pads bottom edge).
        
        Args:
            frame: Input frame in BGR format
            scale_factors: Dictionary containing scale factors for x and y dimensions
            center_point: Tuple of (x, y) coordinates (0.0 to 1.0) around which to scale
        
        Returns:
            Scaled frame with same dimensions as input frame
        """
        height, width = frame.shape[:2]
        
        # Convert center point to pixel coordinates
        center_x = int(center_point[0] * width)
        center_y = int(center_point[1] * height)
        
        # Calculate new dimensions
        new_width = int(width * scale_factors['x'])
        # For y, only scale up if needed
        new_height = max(height, int(height * scale_factors['y']))
        
        # Create temporary frame for scaling
        temp_frame = np.zeros((new_height, new_width, 3), dtype=frame.dtype)
        
        # Calculate source and destination regions for scaling
        src_x1 = max(0, center_x - width // 2)
        src_x2 = min(width, center_x + width // 2)
        src_y1 = max(0, center_y - height // 2)
        src_y2 = min(height, center_y + height // 2)
        
        dst_x1 = max(0, center_x - new_width // 2)
        dst_x2 = min(new_width, center_x + new_width // 2)
        dst_y1 = max(0, center_y - new_height // 2)
        dst_y2 = min(new_height, center_y + new_height // 2)
        
        # Copy and scale the region
        temp_frame[dst_y1:dst_y2, dst_x1:dst_x2] = cv2.resize(
            frame[src_y1:src_y2, src_x1:src_x2],
            (dst_x2 - dst_x1, dst_y2 - dst_y1),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Create output frame with original dimensions
        output_frame = np.zeros((height, width, 3), dtype=frame.dtype)
        
        # Calculate regions for copying back to original size
        if scale_factors['x'] > 1.0:  # Scaling up - crop
            src_x1 = max(0, center_x - width // 2)
            src_x2 = min(new_width, center_x + width // 2)
            dst_x1 = 0
            dst_x2 = width
        else:  # Scaling down - pad with edge values
            src_x1 = max(0, center_x - new_width // 2)
            src_x2 = min(new_width, center_x + new_width // 2)
            dst_x1 = max(0, center_x - width // 2)
            dst_x2 = min(width, center_x + width // 2)
            
        # For y, always keep the bottom edge and only crop/pad the top
        if scale_factors['y'] > 1.0:  # Scaling up - crop from top
            src_y1 = max(0, center_y - height // 2)
            src_y2 = min(new_height, center_y + height // 2)
            dst_y1 = 0
            dst_y2 = height
        else:  # Scaling down - pad top with edge values
            src_y1 = max(0, center_y - new_height // 2)
            src_y2 = min(new_height, center_y + new_height // 2)
            dst_y1 = max(0, center_y - height // 2)
            dst_y2 = height  # Always keep the bottom edge
        
        # Copy scaled region back to original size
        output_frame[dst_y1:dst_y2, dst_x1:dst_x2] = temp_frame[src_y1:src_y2, src_x1:src_x2]
        
        # If scaling down, fill empty regions with edge values
        if scale_factors['x'] < 1.0:
            if dst_x1 > 0:
                output_frame[:, :dst_x1] = output_frame[:, dst_x1]
            if dst_x2 < width:
                output_frame[:, dst_x2:] = output_frame[:, dst_x2-1]
                
        if scale_factors['y'] < 1.0:
            if dst_y1 > 0:
                output_frame[:dst_y1, :] = output_frame[dst_y1, :]
            # No need to pad bottom edge
        
        return output_frame

    def scale_landmarks(self, 
                       results: Dict,
                       scale_factors: Dict[str, float],
                       center_point: Tuple[float, float] = (0.5, 0.5)) -> Dict:
        """
        Scale landmark coordinates using the same scale factors as the frame.
        For y-dimension, only scales upward (never scales down).
        
        Args:
            results: Dictionary containing detection results
            scale_factors: Dictionary containing scale factors for x and y dimensions
            center_point: Tuple of (x, y) coordinates (0.0 to 1.0) around which to scale
        
        Returns:
            Dictionary containing scaled landmark coordinates
        """
        scaled_results = results.copy()
        
        # Helper function to scale a single landmark
        def scale_landmark(lm, frame_width: int, frame_height: int):
            # Convert to pixel coordinates
            x = lm.x * frame_width
            y = lm.y * frame_height
            
            # Convert center point to pixel coordinates
            center_x = center_point[0] * frame_width
            center_y = center_point[1] * frame_height
            
            # Scale relative to center point
            x = center_x + (x - center_x) * scale_factors['x']
            # For y, only scale up if needed
            if scale_factors['y'] > 1.0:
                y = center_y + (y - center_y) * scale_factors['y']
            
            # Convert back to normalized coordinates
            return type(lm)(x / frame_width, y / frame_height, lm.z)
        
        # Scale pose landmarks
        if results['pose_landmarks']:
            scaled_landmarks = []
            for lm in results['pose_landmarks'].landmark:
                scaled_landmarks.append(scale_landmark(lm, 1, 1))  # MediaPipe landmarks are already normalized
            scaled_results['pose_landmarks'] = type(results['pose_landmarks'])(landmark=scaled_landmarks)
        
        # Scale face landmarks
        if results['face_landmarks']:
            scaled_landmarks = []
            for lm in results['face_landmarks'].landmark:
                scaled_landmarks.append(scale_landmark(lm, 1, 1))
            scaled_results['face_landmarks'] = type(results['face_landmarks'])(landmark=scaled_landmarks)
        
        # Scale hand landmarks
        if results['left_hand_landmarks']:
            scaled_landmarks = []
            for lm in results['left_hand_landmarks'].landmark:
                scaled_landmarks.append(scale_landmark(lm, 1, 1))
            scaled_results['left_hand_landmarks'] = type(results['left_hand_landmarks'])(landmark=scaled_landmarks)
        
        if results['right_hand_landmarks']:
            scaled_landmarks = []
            for lm in results['right_hand_landmarks'].landmark:
                scaled_landmarks.append(scale_landmark(lm, 1, 1))
            scaled_results['right_hand_landmarks'] = type(results['right_hand_landmarks'])(landmark=scaled_landmarks)
        
        return scaled_results

    def process_video_with_scaling(self, 
                                 video_path: str,
                                 results_list: List[Dict],
                                 scale_factors: Dict[str, float],
                                 center_point: Tuple[float, float] = (0.5, 0.5),
                                 output_path: Optional[str] = None) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Process a video and its landmarks with scaling.
        
        Args:
            video_path: Path to input video file
            results_list: List of dictionaries containing detection results for each frame
            scale_factors: Dictionary containing scale factors for x and y dimensions
            center_point: Tuple of (x, y) coordinates (0.0 to 1.0) around which to scale
            output_path: Optional path to save the processed video
        
        Returns:
            Tuple containing:
            - List of scaled frames
            - List of scaled landmark results
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        scaled_frames = []
        scaled_results = []
        
        try:
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx >= len(results_list):
                    break
                
                # Get results for this frame
                results = results_list[frame_idx]
                
                # Scale frame
                scaled_frame = self.scale_frame(frame, scale_factors, center_point)
                scaled_frames.append(scaled_frame)
                
                # Scale landmarks
                scaled_result = self.scale_landmarks(results, scale_factors, center_point)
                scaled_results.append(scaled_result)
                
                # Write to output video if path is provided
                if output_path:
                    annotated_frame = self.draw_landmarks_on_frame(scaled_frame, scaled_result)
                    out.write(annotated_frame)
                
                frame_idx += 1
        
        finally:
            cap.release()
            if output_path:
                out.release()
        
        return scaled_frames, scaled_results