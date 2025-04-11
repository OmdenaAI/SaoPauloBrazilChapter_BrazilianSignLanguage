import matplotlib.pyplot as plt
from typing import List, Tuple
import numpy as np
import matplotlib.patches as patches

POSE_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),  # left eye and ear
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),  # right eye and ear
    (9, 10),  # mouth
    (11, 12),  # shoulders
    (11, 13),
    (13, 15),  # left arm
    (12, 14),
    (14, 16),  # right arm
    (15, 17),
    (15, 19),
    (15, 21),  # left wrist to fingers
    (16, 18),
    (16, 20),
    (16, 22),  # right wrist to fingers
    (11, 23),
    (12, 24),  # torso sides
    (23, 24),  # hips
    (23, 25),
    (25, 27),
    (27, 29),
    (29, 31),  # left leg
    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32),  # right leg
]

HAND_CONNECTIONS = [
    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    # Index Finger
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    # Middle Finger
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    # Ring Finger
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    # Pinky Finger
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


def draw_angle_arc(ax, a, b, c, color="orange", radius=0.05, linewidth=1):
    # Vectors BA and BC
    v1 = np.array([a[0] - b[0], a[1] - b[1]])
    v2 = np.array([c[0] - b[0], c[1] - b[1]])

    # Normalize vectors
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)

    # # Get angle in radians and degrees
    # angle_rad = np.arccos(np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0))
    # angle_deg = np.degrees(angle_rad)

    # Calculate start and end angles for the arc (in degrees)
    start_angle = np.degrees(np.arctan2(v1_norm[1], v1_norm[0]))
    end_angle = np.degrees(np.arctan2(v2_norm[1], v2_norm[0]))

    # Adjust angles so that the arc is drawn in the correct direction
    # (matplotlib always goes counterclockwise)
    if end_angle < start_angle:
        end_angle += 360

    arc = patches.Arc(
        b,
        width=2 * radius,
        height=2 * radius,
        angle=0,
        theta1=start_angle,
        theta2=end_angle,
        color=color,
        linewidth=linewidth,
    )
    ax.add_patch(arc)


def visualize_landmark_features(
    landmarks: List,
    landmark_type: str,
    distances: List[Tuple[int, int]] = None,
    angles: List[Tuple[int, int, int]] = None,
    title: str = "Landmark Feature Visualization",
):
    """
    Visualizes a set of landmarks with optional skeleton, distances, and angles.

    Parameters:
    -----------
    landmarks : mp.framework.formats.landmark_pb2.NormalizedLandmarkList
        The list of MediaPipe landmarks to visualize. Each landmark should have `x` and `y` attributes.

    landmark_type : str
        Either 'pose' or 'hand'. Determines which default skeleton connections to draw.

    distances : List[Tuple[int, int]], optional
        A list of (start_index, end_index) pairs representing custom distances to draw as green lines.

    angles : List[Tuple[int, int, int]], optional
        A list of (point_a, point_b, point_c) triplets representing angle connections (V-shapes), drawn in orange.
        Each triplet defines an angle at point_b between segments AB and CB.

    title : str
        Title of the plot.

    Notes:
    ------
    - The function automatically inverts the Y-axis to match image coordinates.
    - Uses predefined `POSE_CONNECTIONS` and `HAND_CONNECTIONS` for skeleton drawing. These should be defined globally.
    - Landmark indices are shown in red for easy identification.

    Returns:
    --------
    None. Displays the plot.
    """
    assert landmark_type in ["hand", "pose"], "landmark_type must be 'hand' or 'pose'"

    landmarks = landmarks.landmark
    x = [lm.x for lm in landmarks]
    y = [lm.y for lm in landmarks]

    plt.figure(figsize=(8, 10))
    plt.scatter(x, y)

    # Draw main skeleton
    if landmark_type == "pose":
        connections = POSE_CONNECTIONS
        radius = 0.015
    else:
        connections = HAND_CONNECTIONS
        radius = 0.005

    for start, end in connections:
        plt.plot([x[start], x[end]], [y[start], y[end]], "b-", linewidth=1)

    # Draw distances (green lines)
    if distances:
        for start, end in distances:
            plt.plot([x[start], x[end]], [y[start], y[end]], "g--", linewidth=2)

    # Draw angle connections (orange V-shapes)
    if angles:
        ax = plt.gca()  # get current axis
        for a_idx, b_idx, c_idx in angles:
            a = (x[a_idx], y[a_idx])
            b = (x[b_idx], y[b_idx])
            c = (x[c_idx], y[c_idx])

            draw_angle_arc(ax, a, b, c, color="orange", radius=radius)

    # Landmark indices
    for i, (xi, yi) in enumerate(zip(x, y)):
        plt.text(xi, yi, str(i), fontsize=8, color="red")

    plt.gca().invert_yaxis()
    plt.title(title)
    plt.axis("equal")
    plt.show()


def visualize_differences(
    prev_landmarks: List,
    next_landmarks: List,
    landmark_type: str,
    landmark_differences: List[int],
    title: str = "2D Landmark Differences",
):
    """
    Visualizes motion between two 2D landmark frames using red arrows.

    Parameters:
    -----------
    prev_landmarks : List of landmarks (with x, y)
        Previous frame landmarks.

    next_landmarks : List of landmarks (with x, y)
        Current frame landmarks.

    landmark_type : str
        Either 'pose' or 'hand'. Determines which default skeleton connections to draw.

    landmarks_differences: List of landmarks

    title : str
        Title of the plot.
    """
    prev_landmarks = prev_landmarks.landmark
    next_landmarks = next_landmarks.landmark
    x1 = [lm.x for lm in prev_landmarks]
    y1 = [lm.y for lm in prev_landmarks]
    x2 = [lm.x for lm in next_landmarks]
    y2 = [lm.y for lm in next_landmarks]

    plt.figure(figsize=(8, 10))
    plt.scatter(x1, y1, c="blue", label="Prev Frame")
    plt.scatter(x2, y2, c="green", label="Next Frame")

    if landmark_type == "pose":
        connections = POSE_CONNECTIONS
        head_width = 0.01
    else:
        connections = HAND_CONNECTIONS
        head_width = 0.005

    # Skeleton connections (prev frame)
    for start, end in connections:
        plt.plot([x1[start], x1[end]], [y1[start], y1[end]], "b--", linewidth=1)

    # Skeleton connections (next frame)
    for start, end in connections:
        plt.plot([x2[start], x2[end]], [y2[start], y2[end]], "g--", linewidth=1)

    # Arrows: movement from prev to next
    for i in landmark_differences:
        plt.arrow(
            x1[i],
            y1[i],
            x2[i] - x1[i],
            y2[i] - y1[i],
            head_width=head_width,
            color="red",
            alpha=0.6,
        )

    plt.gca().invert_yaxis()
    plt.title(title)
    plt.axis("equal")
    plt.legend()
    plt.show()
