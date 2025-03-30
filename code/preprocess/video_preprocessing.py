import cv2
from pathlib import Path
from typing import Union

def resize_video_with_crop(input_path: Union[str, Path], 
                         output_path: Union[str, Path],
                         target_width: int,
                         target_height: int) -> None:
    """
    Resize a video to target dimensions while maintaining aspect ratio by cropping.
    
    Args:
        input_path: Path to input video
        output_path: Path to save resized video
        target_width: Target width in pixels
        target_height: Target height in pixels
    """
    # Open input video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate scaling factors
    width_scale = target_width / orig_width
    height_scale = target_height / orig_height
    scale = max(width_scale, height_scale)
    
    # Calculate new dimensions before cropping
    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)
    
    # Calculate crop coordinates to center the frame
    crop_x = (new_width - target_width) // 2
    crop_y = (new_height - target_height) // 2
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (target_width, target_height))
    
    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize frame
        resized = cv2.resize(frame, (new_width, new_height))
        
        # Crop to target dimensions
        cropped = resized[crop_y:crop_y + target_height, crop_x:crop_x + target_width]
        
        # Write frame
        out.write(cropped)
    
    # Release resources
    cap.release()
    out.release()

def pad_video_to_duration(input_path: Union[str, Path],
                         output_path: Union[str, Path],
                         target_duration: int) -> None:
    """
    Pad or trim a video to target duration by equally padding/trimming from start and end.
    
    Args:
        input_path: Path to input video
        output_path: Path to save padded/trimmed video
        target_duration: Target number of frames
    """
    # Open input video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if frame_count > target_duration:
        # Need to trim
        trim_amount = frame_count - target_duration
        trim_start = trim_amount // 2
        trim_end = frame_count - (trim_amount - trim_start)
        
        print(f"Trimming video from {frame_count} to {target_duration} frames")
        print(f"Removing {trim_start} frames from start and {frame_count - trim_end} frames from end")
        
        # Skip frames until trim_start
        for _ in range(trim_start):
            cap.read()
            
        # Write frames until trim_end
        for _ in range(target_duration):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            
    else:
        # Need to pad
        pad_amount = target_duration - frame_count
        pad_start = pad_amount // 2
        pad_end = pad_amount - pad_start
        
        print(f"Padding video from {frame_count} to {target_duration} frames")
        print(f"Adding {pad_start} frames at start and {pad_end} frames at end")
        
        # Read first and last frames
        ret, first_frame = cap.read()
        if not ret:
            raise ValueError("Could not read first frame")
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        ret, last_frame = cap.read()
        if not ret:
            raise ValueError("Could not read last frame")
        
        # Write padding frames at start
        for _ in range(pad_start):
            out.write(first_frame)
            
        # Reset to start and write original frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            
        # Write padding frames at end
        for _ in range(pad_end):
            out.write(last_frame)
    
    # Release resources
    cap.release()
    out.release()
