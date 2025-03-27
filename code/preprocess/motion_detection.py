import cv2
import numpy as np
import matplotlib.pyplot as plt

def measure_motion_basic(input_video_path, motion_threshold=30):
    '''
    Calculates the absolute difference between consecutive frames to measure motion.
    The larger the difference, the more motion is present.
    '''
    # Open video file
    cap = cv2.VideoCapture(input_video_path)
    
    ret, prev_frame = cap.read()
    if not ret:
        print("Error reading video.")
        return
    
    # Convert the first frame to grayscale
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    frame_count = 0
    motion_measurements = []  # Store motion data for each frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate the absolute difference between the current frame and previous frame
        frame_diff = cv2.absdiff(prev_frame_gray, frame_gray)
        
        # Threshold the difference to create a binary image where motion is highlighted
        _, thresh = cv2.threshold(frame_diff, motion_threshold, 255, cv2.THRESH_BINARY)

        # Calculate the number of non-zero pixels (which indicates motion)
        non_zero_count = np.count_nonzero(thresh)
        motion_measurements.append(non_zero_count)  # Store the motion count for this frame

        # Update previous frame for the next iteration
        prev_frame_gray = frame_gray

    cap.release()
    return motion_measurements

def measure_motion_optical_flow(input_video_path):
    '''
    Optical flow is a more advanced technique for tracking motion across frames.
    It calculates the flow (motion) of objects between two consecutive frames using their pixel movements.
    '''
    # Open video file
    cap = cv2.VideoCapture(input_video_path)

    ret, prev_frame = cap.read()
    if not ret:
        print("Error reading video.")
        return

    # Convert the first frame to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    frame_count = 0
    motion_measurements = []  # Store motion data for each frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow (movement) between previous and current frame
        flow = cv2.calcOpticalFlowFarneback(prev_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Calculate the magnitude of motion (speed of movement) in each direction
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Calculate the total magnitude of motion for the entire frame
        total_magnitude = np.sum(magnitude)

        motion_measurements.append(total_magnitude)  # Store the total motion magnitude for this frame

        # Update the previous frame for the next iteration
        prev_gray = frame_gray

    cap.release()
    return motion_measurements

def measure_motion_background_subtraction(input_video_path):
    '''
    If the background is relatively static, you can use background subtraction methods to detect motion.
    '''
    # Open video file
    cap = cv2.VideoCapture(input_video_path)
    
    # Initialize background subtractor (e.g., MOG2)
    back_sub = cv2.createBackgroundSubtractorMOG2()
    
    frame_count = 0
    motion_measurements = []  # Store motion data for each frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Apply background subtraction
        fg_mask = back_sub.apply(frame)

        # Count the number of non-zero pixels in the foreground mask (indicating motion)
        non_zero_count = np.count_nonzero(fg_mask)
        motion_measurements.append(non_zero_count)  # Store the motion count for this frame

    cap.release()
    return motion_measurements[1:]

def normalize_list_of_data(data):
    """Normalize a list of data to the range [0, 1]."""
    min_val = min(data)
    max_val = max(data)
    if max_val == min_val:  # Handle case where all values are the same
        normalized_data = [0.5 for _ in data]  # Set all values to 0.5
    else:
        normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
    return normalized_data


def normalize_lists_of_data(data):
    """Normalize each list in data to the range [0, 1]."""
    normalized_data = []
    for series in data:
        normalized_data.append(normalize_list_of_data(series))
    return normalized_data

def weighted_average_motion(motion_series_list, weights=None):
    '''
    Combines multiple lists of motion measurements with weighted averaging.
    Note: Input lists should be pre-normalized if normalization is desired.
    
    Args:
        motion_series_list: List of lists, where each inner list contains motion values
        weights: List of weights corresponding to each motion series. 
                If None, uses equal weights for all series.
        
    Returns:
        List of combined motion values
    '''
    # If weights not provided, use equal weights
    if weights is None:
        weights = [1.0 / len(motion_series_list)] * len(motion_series_list)
    
    # Validate inputs
    if len(motion_series_list) != len(weights):
        raise ValueError("Number of motion series must match number of weights")
    if not motion_series_list:
        raise ValueError("At least one motion series must be provided")
    
    # Convert all series to numpy arrays
    motion_arrays = [np.array(series) for series in motion_series_list]
    
    # Find minimum length among all series
    min_length = min(len(arr) for arr in motion_arrays)
    
    # Trim all arrays to minimum length
    motion_arrays = [arr[:min_length] for arr in motion_arrays]
    
    # Combine methods with weights
    combined_motion = np.zeros(min_length)
    for arr, weight in zip(motion_arrays, weights):
        combined_motion += weight * arr
    
    return combined_motion.tolist()

def moving_average(data, fps, window_duration=0.334, verbose=True):
    """
    Apply moving average to a time series with configurable window duration.
    
    Args:
        data: List of values to smooth
        fps: Frames per second of the video
        window_duration: Duration in seconds for the moving average window (default: 0.2s)
        
    Returns:
        List of smoothed values
    """
    # Convert time-based window to frame-based window
    window_size = int(window_duration * fps)
    window_size = max(1, window_size)  # Ensure minimum window size of 1 frame
    if verbose:
        print(f"window_duration: {window_duration} seconds, at {fps} fps = {window_size} frame window_size")
    
    # Convert to numpy array and apply moving average
    data_array = np.array(data)
    smoothed = np.convolve(data_array, np.ones(window_size)/window_size, mode='same')
    
    return smoothed.tolist()

def get_frame(input_video_path, frame_number):
    cap = cv2.VideoCapture(input_video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    return np.array(frame)

def create_frame_with_motion_graph(frame, motion_data, frame_number, legend_labels=None, target_width=1280, target_height=720, graph_height=300, figsize=(12, 4), dpi=100, alpha=1):
    '''
    Creates a single frame with video and motion graph.
    Returns the combined frame.
    
    Args:
        frame: The video frame to display
        motion_data: List of motion measurement series
        frame_number: Current frame number for the vertical line
        legend_labels: List of labels for the legend. If None, uses "Series 1", "Series 2", etc.
        target_width: Width of the output frame
        target_height: Height of the video portion of the frame
        graph_height: Height of the graph portion of the frame
        figsize: Size of the matplotlib figure
        dpi: DPI of the matplotlib figure
        alpha: Transparency for all lines except the last one (default: 0.5)
    '''
    # Get original video dimensions
    original_height, original_width = frame.shape[:2]

    # Set up the plot for motion graph
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlim(0, len(motion_data[0]))
    ax.set_ylim(0, 1)  # Normalized data is between 0 and 1
    ax.set_title("Motion Over Time (Normalized)")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Motion Value")
    
    # Plot each motion data series with a legend
    colors = plt.cm.tab10.colors  # Use a color map for distinct colors
    lines = []
    labels = []
    for i, series in enumerate(motion_data):
        label = legend_labels[i] if legend_labels and i < len(legend_labels) else f"Series {i + 1}"
        # Make all lines except the last one semi-transparent and dotted
        line_alpha = alpha if i < len(motion_data) - 1 else 1.0
        line_style = '--' if i < len(motion_data) - 1 else '-'
        line_width = 1 if i < len(motion_data) - 1 else 2
        line, = ax.plot(range(len(series)), series, color=colors[i % len(colors)], 
                       lw=line_width, label=label, alpha=line_alpha, linestyle=line_style)
        lines.append(line)
        labels.append(label)
    
    # Add a legend to the plot
    ax.legend(lines, labels, loc="upper right")

    # Create the vertical line that will move with the frames
    vertical_line, = ax.plot([0, 0], [0, 1], color="black", lw=2)
    vertical_line.set_xdata([frame_number, frame_number])

    # Disable the toolbar and set the figure to not block
    plt.tight_layout()
    plt.ion()

    # Resize the input video to match the target width while maintaining aspect ratio
    aspect_ratio = original_width / original_height
    resized_width = target_width
    resized_height = int(target_width / aspect_ratio)

    # If the resized height exceeds the target height, adjust to fit
    if resized_height > target_height:
        resized_height = target_height
        resized_width = int(target_height * aspect_ratio)

    # Resize the video frame
    resized_frame = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    # Create a blank canvas with the target dimensions
    combined_frame = np.zeros((target_height + graph_height, target_width, 3), dtype=np.uint8)

    # Calculate padding to center the resized video on the canvas
    x_offset = (target_width - resized_width) // 2
    y_offset = (target_height - resized_height) // 2

    # Place the resized video on the canvas
    combined_frame[y_offset:y_offset + resized_height, x_offset:x_offset + resized_width] = resized_frame

    # Draw the plot
    fig.canvas.draw()

    # Convert the plot to an image
    graph_img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    graph_img = graph_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    graph_img = cv2.cvtColor(graph_img, cv2.COLOR_RGBA2BGR)

    # Resize the graph to match the target width
    graph_img = cv2.resize(graph_img, (target_width, graph_height), interpolation=cv2.INTER_AREA)

    # Place the graph at the bottom of the canvas
    combined_frame[target_height:, :] = graph_img

    # Clean up matplotlib
    plt.close(fig)
    
    return combined_frame

def play_video_with_motion_graph(input_video_path, motion_data, legend_labels=None, graph_height=300, figsize=(12, 4), dpi=100, alpha=1, output_video_path=None):
    # Define target dimensions for the output video
    target_width = 1280  # Set your desired width
    target_height = 720  # Set your desired height

    # Open the video
    cap = cv2.VideoCapture(input_video_path)
    ret, frame = cap.read()

    if not ret:
        print("Error reading video.")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer if output path is provided
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (target_width, target_height + graph_height))
        print(f"Saving video to: {output_video_path}")

    # Create the OpenCV window
    cv2.namedWindow("Video with Motion Graph", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video with Motion Graph", target_width, target_height + graph_height)

    while True:  # Outer loop to allow replaying
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to the first frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Get the current frame number
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # Create the combined frame using the new function
            combined_frame = create_frame_with_motion_graph(
                frame, motion_data, frame_number,
                legend_labels=legend_labels,
                target_width=target_width,
                target_height=target_height,
                graph_height=graph_height,
                figsize=figsize,
                dpi=dpi,
                alpha=alpha
            )

            # Show the combined frame
            cv2.imshow("Video with Motion Graph", combined_frame)

            # Write frame to output video if output path is provided
            if output_video_path:
                out.write(combined_frame)
                # Print progress
                if frame_number % 30 == 0:  # Update every 30 frames
                    progress = (frame_number / total_frames) * 100
                    print(f"Progress: {progress:.1f}%", end='\r')

            # Check if the window is closed
            if cv2.getWindowProperty("Video with Motion Graph", cv2.WND_PROP_VISIBLE) < 1:
                if output_video_path:
                    out.release()
                cap.release()
                cv2.destroyAllWindows()
                return

            # Wait for a key press to exit or move to next frame
            if cv2.waitKey(1) & 0xFF == ord('q'):
                if output_video_path:
                    out.release()
                cap.release()
                cv2.destroyAllWindows()
                return

        # Video has ended, wait for user input
        print("\nVideo ended. Press 'r' to replay or 'q' to quit.")
        print("CLOSING THE WINDOW BY CLICKING X WILL KEEP THE CELL RUNNING!")
        while True:
            # Check if the window is closed
            if cv2.getWindowProperty("Video with Motion Graph", cv2.WND_PROP_VISIBLE) < 1:
                if output_video_path:
                    out.release()
                cap.release()
                cv2.destroyAllWindows()
                return

            key = cv2.waitKey(0) & 0xFF
            if key == ord('r'):  # Replay the video
                break
            elif key == ord('q'):  # Quit the program
                if output_video_path:
                    out.release()
                cap.release()
                cv2.destroyAllWindows()
                return

def show_multiple_frames_one_plot(path, frame_numbers, motion_data, figsize=(20, 3), alpha=1, legend_labels=None):
    fig, axs = plt.subplots(1, len(frame_numbers), figsize=figsize)

    for i, frame_number in enumerate(frame_numbers):
        frame = get_frame(path, frame_number)
        axs[i].imshow(frame[:,:,::-1])
        axs[i].axis('off')
        axs[i].set_title(f'Frame {frame_number}', fontweight='bold')
    plt.suptitle('Video Frames')
    plt.show()

    plt.figure(figsize=figsize)
    for i, series in enumerate(motion_data):
        # Make all lines except the last one semi-transparent and dotted
        line_alpha = alpha if i < len(motion_data) - 1 else 1.0
        line_style = '--' if i < len(motion_data) - 1 else '-'
        line_width = 1 if i < len(motion_data) - 1 else 2
        plt.plot(series, alpha=line_alpha, linestyle=line_style, linewidth=line_width)
    if legend_labels:
        plt.legend(legend_labels)
    else:
        if len(motion_data) == 4:
            plt.legend(['Basic', 'Optical Flow', 'Background Sub', 'Weighted Avg'])
        if len(motion_data) == 3:
            plt.legend(['Basic', 'Background Sub', 'Weighted Avg'])
            
    for i, frame_number in enumerate(frame_numbers):
        plt.axvline(x=frame_number, linestyle='--', linewidth=2, color='black')
        # annotate the frame number
        plt.annotate(f'Frame {frame_number}', (frame_number+.2, i/len(frame_numbers)), fontweight='bold')
    plt.xlabel('Frame Number')
    plt.ylabel('Normalized Motion Value')
    plt.title('Motion Detection Comparison')
    plt.show()


def show_multiple_frames_multiple_plots(path, frame_numbers, motion_data, figsize=(20, 3), alpha=1):
    fig, axs = plt.subplots(1, len(frame_numbers), figsize=figsize)

    for i, frame_number in enumerate(frame_numbers):
        frame = get_frame(path, frame_number)

        # Create the plot
        fig_ = plt.figure(dpi=150)
        for j, series in enumerate(motion_data):
            # Make all lines except the last one semi-transparent and dotted
            line_alpha = alpha if j < len(motion_data) - 1 else 1.0
            line_style = '--' if j < len(motion_data) - 1 else '-'
            line_width = 1 if j < len(motion_data) - 1 else 2
            plt.plot(series, alpha=line_alpha, linestyle=line_style, linewidth=line_width)
        plt.axvline(x=frame_number, linestyle='-', linewidth=5, color='black')
        plt.axis('off')
        plt.tight_layout()
        fig_.canvas.draw()
        plot_img = np.frombuffer(fig_.canvas.buffer_rgba(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig_.canvas.get_width_height()[::-1] + (4,))
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)
        plot_img = cv2.copyMakeBorder(plot_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0,0,0])
        plt.close(fig_)

        # vertical stack the frame and the plot after resizing the plot to the same width as the plot
        plot_img = cv2.resize(plot_img, (frame.shape[1], int(frame.shape[0]/4)))
        combined_frame = np.vstack((frame, plot_img))

        axs[i].imshow(combined_frame[:,:,::-1])
        axs[i].axis('off')
        axs[i].set_title(f'Frame {frame_number}', fontweight='bold')
    plt.suptitle('Video Frames')
    plt.show()

def find_motion_boundary_simple(motion_data, threshold, direction='forward'):
    """
    Find the first frame where motion crosses a threshold.
    Note: Input motion_data should be pre-normalized if normalization is desired.
    
    Args:
        motion_data: List of motion values
        threshold: Threshold value to detect (between 0 and 1 for normalized data)
        direction: 'forward' to find first frame above threshold, 'backward' to find last frame above threshold
        
    Returns:
        int: Frame number where the threshold is crossed, or None if not found
    """
    if direction not in ['forward', 'backward']:
        raise ValueError("direction must be 'forward' or 'backward'")
    
    # Convert to numpy array for easier manipulation
    motion = np.array(motion_data)
    
    if direction == 'forward':
        # Find first frame where motion goes above threshold
        frames = np.where(motion > threshold)[0]
        return frames[0] if len(frames) > 0 else None
    else:
        # Find last frame where motion goes above threshold
        frames = np.where(motion[::-1] > threshold)[0]
        return len(motion) - 1 - frames[0] if len(frames) > 0 else None


def find_motion_boundaries_simple(motion_data, start_threshold, end_threshold=None):
    """
    Find both start and end points of a sign based on a simple threshold.
    Note: Input motion_data should be pre-normalized if normalization is desired.
    
    Args:
        motion_data: List of motion values
        start_threshold: Threshold value to detect start (between 0 and 1 for normalized data)
        end_threshold: Threshold value to detect end. If None, uses start_threshold.
        
    Returns:
        tuple: (start_frame, end_frame) where the sign boundaries are detected
    """
    # If end_threshold not specified, use start_threshold
    if end_threshold is None:
        end_threshold = start_threshold
    
    start_frame = find_motion_boundary_simple(
        motion_data, 
        start_threshold,
        direction='forward'
    )
    
    end_frame = find_motion_boundary_simple(
        motion_data,
        end_threshold,
        direction='backward'
    )
    
    return start_frame, end_frame

def find_motion_boundary_complex(motion_data, fps, direction='forward', threshold=0.1, min_motion_duration=0.3):
    """
    Find the start or end point of a sign based on motion data analysis.
    Note: Input motion_data should be pre-normalized if normalization is desired.
    
    Args:
        motion_data: List of motion values
        fps: Frames per second of the video
        direction: 'forward' to find start point, 'backward' to find end point
        threshold: Minimum increase/decrease in motion to consider as sign start/end (default: 0.1)
        min_motion_duration: Minimum duration in seconds of significant motion (default: 0.3s)
        
    Returns:
        int: Frame number where the sign boundary is detected
    """
    if direction not in ['forward', 'backward']:
        raise ValueError("direction must be 'forward' or 'backward'")
    
    # Convert time-based parameters to frame-based parameters
    min_frames = int(min_motion_duration * fps)
    min_frames = max(5, min_frames)    # At least 5 frames for minimum motion duration
    
    # Convert to numpy array for easier manipulation
    motion = np.array(motion_data)
    
    # Calculate the difference between consecutive values
    if direction == 'forward':
        diff = np.diff(motion)
        frames = range(len(diff))
        # For forward direction, look for increases in motion
        significant_motion = diff > threshold
    else:
        diff = np.diff(motion[::-1])[::-1]  # Reverse for backward analysis
        frames = range(len(diff)-1, -1, -1)
        # For backward direction, look for decreases in motion
        significant_motion = diff < -threshold
    
    # Find continuous segments of significant motion
    current_segment_length = 0
    for i, has_motion in enumerate(significant_motion):
        if has_motion:
            current_segment_length += 1
            if current_segment_length >= min_frames:
                # Return the frame where the significant motion started
                return frames[i - min_frames + 1]
        else:
            current_segment_length = 0
    
    # If no significant motion segment found, return None
    return None

def find_motion_boundaries_complex(motion_data, fps, threshold=0.1, min_motion_duration=0.3):
    """
    Find both start and end points of a sign based on motion data analysis.
    Note: Input motion_data should be pre-normalized if normalization is desired.
    
    Args:
        motion_data: List of motion values
        fps: Frames per second of the video
        threshold: Minimum increase/decrease in motion to consider as sign start/end (default: 0.1)
        min_motion_duration: Minimum duration in seconds of significant motion (default: 0.3s)
        
    Returns:
        tuple: (start_frame, end_frame) where the sign boundaries are detected
    """
    start_frame = find_motion_boundary_complex(
        motion_data, 
        fps,
        direction='forward',
        threshold=threshold,
        min_motion_duration=min_motion_duration
    )
    
    end_frame = find_motion_boundary_complex(
        motion_data,
        fps,
        direction='backward',
        threshold=threshold,
        min_motion_duration=min_motion_duration
    )
    
    return start_frame, end_frame

            