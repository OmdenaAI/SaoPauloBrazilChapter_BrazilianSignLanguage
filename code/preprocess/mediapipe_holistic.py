import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MediaPipeHolisticDetector:
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
            refine_face_landmarks: Whether to further refine the landmark coordinates around the eyes and lips, 
                                 and output additional landmarks around the irises.
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
    
    def draw_landmarks(self, image: np.ndarray, results: Dict) -> np.ndarray:
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
    
    def draw_landmarks_on_blank(self, results: Dict, canvas_size: int = 800, background_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """
        Draw landmarks on a blank square canvas.
        
        Args:
            results: Dictionary containing detection results
            canvas_size: Size of the square canvas in pixels
            background_color: Background color in BGR format
        
        Returns:
            Square canvas with landmarks drawn
        """
        # Create a blank square canvas
        canvas = np.full((canvas_size, canvas_size, 3), background_color, dtype=np.uint8)
        
        # Define custom drawing specs for better visibility on blank background
        landmark_drawing_spec = self.mp_drawing.DrawingSpec(
            color=(0, 0, 255),  # Red in BGR
            thickness=2,
            circle_radius=2
        )
        connection_drawing_spec = self.mp_drawing.DrawingSpec(
            color=(0, 255, 0),  # Green in BGR
            thickness=2
        )
        
        # Draw face landmarks with custom specs
        if results['face_landmarks']:
            # Draw face mesh
            self.mp_drawing.draw_landmarks(
                image=canvas,
                landmark_list=results['face_landmarks'],
                connections=self.mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,  # No dots for face mesh
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(200, 200, 200),  # Light gray
                    thickness=1
                )
            )
            # Draw face contours
            self.mp_drawing.draw_landmarks(
                image=canvas,
                landmark_list=results['face_landmarks'],
                connections=self.mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 0, 0),  # Black
                    thickness=2
                )
            )
        
        # Draw pose landmarks with custom specs
        if results['pose_landmarks']:
            self.mp_drawing.draw_landmarks(
                image=canvas,
                landmark_list=results['pose_landmarks'],
                connections=self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(255, 0, 0),  # Blue in BGR
                    thickness=4,
                    circle_radius=4
                ),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 255, 0),  # Green in BGR
                    thickness=2
                )
            )
        
        # Draw hand landmarks with custom specs
        hand_landmark_spec = self.mp_drawing.DrawingSpec(
            color=(0, 0, 255),  # Red in BGR
            thickness=4,
            circle_radius=3
        )
        hand_connection_spec = self.mp_drawing.DrawingSpec(
            color=(255, 0, 0),  # Blue in BGR
            thickness=2
        )
        
        # Draw left hand
        if results['left_hand_landmarks']:
            self.mp_drawing.draw_landmarks(
                image=canvas,
                landmark_list=results['left_hand_landmarks'],
                connections=self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=hand_landmark_spec,
                connection_drawing_spec=hand_connection_spec
            )
        
        # Draw right hand
        if results['right_hand_landmarks']:
            self.mp_drawing.draw_landmarks(
                image=canvas,
                landmark_list=results['right_hand_landmarks'],
                connections=self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=hand_landmark_spec,
                connection_drawing_spec=hand_connection_spec
            )
        
        return canvas

    def visualize_frame_with_blank(self, frame: np.ndarray, results: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create both regular visualization and blank canvas visualization.
        
        Args:
            frame: Input frame in BGR format
            results: Dictionary containing detection results
        
        Returns:
            Tuple of (frame with landmarks, blank canvas with landmarks)
        """
        return self.draw_landmarks(frame, results), self.draw_landmarks_on_blank(results)

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
    
    def filter_landmarks_in_frame(self, results: Dict, frame_width: int, frame_height: int, margin: float = 0.1) -> Dict:
        """
        Filter out landmarks that are outside the frame (with a margin).
        
        Args:
            results: Dictionary containing detection results
            frame_width: Width of the video frame
            frame_height: Height of the video frame
            margin: Margin percentage (0.1 = 10% outside frame allowed)
        
        Returns:
            Dictionary with filtered landmarks
        """
        def is_point_in_frame(point, width, height, margin):
            # Allow points to be slightly outside frame with margin
            x, y = point.x * width, point.y * height
            margin_w, margin_h = width * margin, height * margin
            return (-margin_w <= x <= width + margin_w) and (-margin_h <= y <= height + margin_h)
        
        filtered_results = results.copy()
        
        # Filter face landmarks
        if results['face_landmarks']:
            points = [point for point in results['face_landmarks'].landmark 
                     if is_point_in_frame(point, frame_width, frame_height, margin)]
            if len(points) < len(results['face_landmarks'].landmark) * 0.5:  # If more than 50% points are outside
                filtered_results['face_landmarks'] = None
        
        # Filter pose landmarks
        if results['pose_landmarks']:
            points = [point for point in results['pose_landmarks'].landmark 
                     if is_point_in_frame(point, frame_width, frame_height, margin)]
            if len(points) < len(results['pose_landmarks'].landmark) * 0.5:
                filtered_results['pose_landmarks'] = None
        
        # Filter hand landmarks
        for hand_key in ['left_hand_landmarks', 'right_hand_landmarks']:
            if results[hand_key]:
                points = [point for point in results[hand_key].landmark 
                         if is_point_in_frame(point, frame_width, frame_height, margin)]
                if len(points) < len(results[hand_key].landmark) * 0.5:
                    filtered_results[hand_key] = None
        
        return filtered_results

    def center_landmarks(self, results: Dict, target_center: Tuple[float, float] = (0.5, 0.5)) -> Dict:
        """
        Center landmarks around a target point.
        Uses torso center (between shoulders and hips) as reference when available.
        
        Args:
            results: Dictionary containing detection results
            target_center: Target center point in normalized coordinates (default: center of frame)
        
        Returns:
            Dictionary with centered landmarks
        """
        centered_results = results.copy()
        
        # Find reference center
        if results['pose_landmarks']:
            # Use torso center (between shoulders and hips)
            shoulders = np.array([
                [results['pose_landmarks'].landmark[11].x, results['pose_landmarks'].landmark[11].y],  # left shoulder
                [results['pose_landmarks'].landmark[12].x, results['pose_landmarks'].landmark[12].y]   # right shoulder
            ])
            hips = np.array([
                [results['pose_landmarks'].landmark[23].x, results['pose_landmarks'].landmark[23].y],  # left hip
                [results['pose_landmarks'].landmark[24].x, results['pose_landmarks'].landmark[24].y]   # right hip
            ])
            current_center = (shoulders.mean(axis=0) + hips.mean(axis=0)) / 2
        else:
            # Fallback: use average of all landmarks
            all_points = []
            for key in ['pose_landmarks', 'face_landmarks', 'left_hand_landmarks', 'right_hand_landmarks']:
                if results[key]:
                    points = [[l.x, l.y] for l in results[key].landmark]
                    all_points.extend(points)
            
            if not all_points:
                return centered_results  # No landmarks to center
            
            current_center = np.mean(all_points, axis=0)
        
        # Calculate shift needed
        shift_x = target_center[0] - current_center[0]
        shift_y = target_center[1] - current_center[1]
        
        # Apply shift to all landmarks
        for key in ['pose_landmarks', 'face_landmarks', 'left_hand_landmarks', 'right_hand_landmarks']:
            if centered_results[key]:
                for landmark in centered_results[key].landmark:
                    landmark.x += shift_x
                    landmark.y += shift_y
        
        return centered_results

    def get_landmark_distances(self, pose_landmarks) -> Dict[str, float]:
        """
        Calculate distances between key landmark pairs.
        
        Args:
            pose_landmarks: MediaPipe pose landmarks object
        
        Returns:
            Dictionary containing distances between key points:
            - 'shoulder_width': Distance between shoulders
            - 'hip_width': Distance between hips
            - 'torso_height': Distance between shoulder and hip center
            - 'arm_length': Average distance from shoulder to wrist
            - 'leg_length': Average distance from hip to ankle
            Returns None if pose_landmarks is None
        """
        if not pose_landmarks:
            return None
            
        def get_distance(p1, p2):
            return np.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
        
        pose = pose_landmarks.landmark
        
        # Calculate midpoints for torso
        shoulder_center = type('Point', (), {
            'x': (pose[11].x + pose[12].x)/2, 
            'y': (pose[11].y + pose[12].y)/2
        })()
        hip_center = type('Point', (), {
            'x': (pose[23].x + pose[24].x)/2, 
            'y': (pose[23].y + pose[24].y)/2
        })()
        
        distances = {
            'shoulder_width': get_distance(pose[11], pose[12]),  # left to right shoulder
            'hip_width': get_distance(pose[23], pose[24]),      # left to right hip
            'torso_height': get_distance(shoulder_center, hip_center),  # shoulder center to hip center
            'arm_length': (get_distance(pose[11], pose[15]) +   # average of left and right arm
                         get_distance(pose[12], pose[16])) / 2,
            'leg_length': (get_distance(pose[23], pose[27]) +   # average of left and right leg
                         get_distance(pose[24], pose[28])) / 2
        }
        
        return distances

    def scale_landmarks(self, results: Dict, target_distances: Dict[str, float]) -> Dict:
        """
        Scale landmarks to match target distances between key points.
        
        Args:
            results: Dictionary containing detection results
            target_distances: Dictionary of target distances for key landmark pairs:
                - 'shoulder_width': Target distance between shoulders (default: 0.3)
                - 'hip_width': Target distance between hips (default: 0.25)
                - 'torso_height': Target distance between shoulder and hip center (default: 0.3)
                - 'arm_length': Target distance between shoulder and wrist (default: 0.3)
                - 'leg_length': Target distance between hip and ankle (default: 0.4)
        
        Returns:
            Dictionary with scaled landmarks
        """
        scaled_results = results.copy()
        
        if not results['pose_landmarks']:
            return scaled_results  # Cannot scale without pose landmarks
        
        # Default target distances if not provided
        default_targets = {
            'shoulder_width': 0.3,
            'hip_width': 0.25,
            'torso_height': 0.3,
            'arm_length': 0.3,
            'leg_length': 0.4
        }
        target_distances = {**default_targets, **target_distances}
        
        # Get current distances
        current_distances = self.get_landmark_distances(results['pose_landmarks'])
        if not current_distances:
            return scaled_results
        
        # Calculate scale factors for each distance
        scale_factors = []
        for key, target in target_distances.items():
            if current_distances[key] > 0:  # Avoid division by zero
                scale_factors.append(target / current_distances[key])
        
        if not scale_factors:
            return scaled_results  # No valid scale factors
        
        # Use median scale factor to avoid outliers
        scale_factor = np.median(scale_factors)
        
        # Apply scaling to all landmarks around their respective centers
        for key in ['pose_landmarks', 'face_landmarks', 'left_hand_landmarks', 'right_hand_landmarks']:
            if scaled_results[key]:
                landmarks = scaled_results[key].landmark
                center_x = np.mean([l.x for l in landmarks])
                center_y = np.mean([l.y for l in landmarks])
                
                for landmark in landmarks:
                    # Scale around center
                    landmark.x = center_x + (landmark.x - center_x) * scale_factor
                    landmark.y = center_y + (landmark.y - center_y) * scale_factor
        
        return scaled_results

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
                    annotated_frame = self.draw_landmarks(frame, results)
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

# Example usage
if __name__ == "__main__":
    detector = MediaPipeHolisticDetector(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
        refine_face_landmarks=True
    )
    
    # Process video to get raw landmarks
    results = detector.process_video(
        video_path="input_video.mp4",
        output_path="output_raw.mp4"
    )
    
    # Post-processing can be applied separately as needed:
    # Example for a single frame:
    frame_results = results[0]
    
    # Filter landmarks that are outside the frame
    filtered = detector.filter_landmarks_in_frame(frame_results, width=1920, height=1080)
    
    # Center the landmarks
    centered = detector.center_landmarks(filtered, target_center=(0.5, 0.5))
    
    # Scale the landmarks
    target_distances = {
        'shoulder_width': 0.3,
        'torso_height': 0.4
    }
    scaled = detector.scale_landmarks(centered, target_distances) 