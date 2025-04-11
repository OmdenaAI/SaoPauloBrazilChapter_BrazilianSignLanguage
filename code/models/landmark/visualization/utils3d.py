from typing import List, Tuple
import matplotlib.pyplot as plt
from models.landmark.visualization.utils2d import POSE_CONNECTIONS, HAND_CONNECTIONS
import numpy as np


def draw_angle_arc_3d(
    ax, a, b, c, radius=0.05, color="orange", linewidth=1, resolution=30
):
    """
    Draws an arc in 3D between vectors BA and BC, centered at B.

    Parameters:
    - ax: 3D matplotlib axis
    - a, b, c: points (tuples of 3 floats)
    - radius: arc radius
    - color: arc color
    - linewidth: arc line width
    - resolution: number of segments in arc
    """
    # Vectors from B to A and C
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)

    # Normalize vectors
    ba /= np.linalg.norm(ba)
    bc /= np.linalg.norm(bc)

    # Compute normal of the plane
    normal = np.cross(ba, bc)
    normal /= np.linalg.norm(normal)

    # Create a rotation axis from ba to bc
    angle = np.arccos(np.clip(np.dot(ba, bc), -1.0, 1.0))
    arc_points = []

    for t in np.linspace(0, angle, resolution):
        # Rodrigues' rotation formula to rotate ba around normal
        v = ba
        k = normal
        rotated = (
            v * np.cos(t)
            + np.cross(k, v) * np.sin(t)
            + k * np.dot(k, v) * (1 - np.cos(t))
        )
        arc_points.append(np.array(b) + radius * rotated)

    arc_points = np.array(arc_points)
    ax.plot(
        arc_points[:, 0],
        arc_points[:, 1],
        arc_points[:, 2],
        color=color,
        linewidth=linewidth,
    )


def visualize_landmark_features(
    landmarks: List,
    landmark_type: str,
    distances: List[Tuple[int, int]] = None,
    angles: List[Tuple[int, int, int]] = None,
    title: str = "3D Landmark Feature Visualization",
):
    """
    Visualizes a set of 3D landmarks with optional skeleton, distances, and angles.

    Parameters:
    -----------
    landmarks : List
        List of landmark objects with `x`, `y`, `z` attributes.

    landmark_type : str
        Either 'pose' or 'hand'. Determines which default skeleton connections to draw.

    distances : List[Tuple[int, int]], optional
        Pairs of landmark indices to connect with green lines representing custom distances.

    angles : List[Tuple[int, int, int]], optional
        Triplets of indices (a, b, c) to draw angle V-shapes at point b in orange.

    title : str
        Title of the plot.

    Returns:
    --------
    None. Displays a 3D plot.
    """
    assert landmark_type in ["hand", "pose"], "landmark_type must be 'hand' or 'pose'"
    landmarks = landmarks.landmark
    x = [lm.x for lm in landmarks]
    y = [lm.y for lm in landmarks]
    z = [lm.z for lm in landmarks]

    y, z = z, [-yi for yi in y]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, c="blue", label="Landmarks")

    # Define default connections
    if landmark_type == "pose":
        connections = POSE_CONNECTIONS
        radius = 0.03
    else:
        connections = HAND_CONNECTIONS
        radius = 0.005

    # Draw skeleton in blue
    for start, end in connections:
        ax.plot(
            [x[start], x[end]],
            [y[start], y[end]],
            [z[start], z[end]],
            c="blue",
            linewidth=1,
        )

    # Draw distances in green
    if distances:
        for start, end in distances:
            ax.plot(
                [x[start], x[end]],
                [y[start], y[end]],
                [z[start], z[end]],
                "g--",
                linewidth=2,
            )

    # Draw angles in orange (V-shapes)
    if angles:
        for a_idx, b_idx, c_idx in angles:
            a = (x[a_idx], y[a_idx], z[a_idx])
            b = (x[b_idx], y[b_idx], z[b_idx])
            c = (x[c_idx], y[c_idx], z[c_idx])

            draw_angle_arc_3d(ax, a, b, c, radius=radius, color="orange", linewidth=1)

    # Add landmark indices
    for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
        ax.text(xi, yi, zi, str(i), color="red", fontsize=8)
    ax.view_init(elev=10, azim=-70)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")

    ax.legend()
    plt.tight_layout()
    plt.show()


def visualize_differences(
    prev_landmarks: List,
    next_landmarks: List,
    landmark_type: str,
    landmark_differences: List[int],
    title: str = "3D Hand Landmark Differences",
):
    """
    Visualizes 3D hand motion using arrows from selected key landmarks.

    Parameters:
    -----------
    prev_landmarks : List of landmarks (with x, y, z)
    next_landmarks : List of landmarks (with x, y, z)
    landmark_type : str
        Either 'pose' or 'hand'. Determines which default skeleton connections to draw.
    landmarks_differences: List of landmarks
    title : str
        Title for the 3D plot.
    """
    prev_landmarks = prev_landmarks.landmark
    next_landmarks = next_landmarks.landmark

    x1 = [lm.x for lm in prev_landmarks]
    y1 = [lm.y for lm in prev_landmarks]
    z1 = [lm.z for lm in prev_landmarks]

    x2 = [lm.x for lm in next_landmarks]
    y2 = [lm.y for lm in next_landmarks]
    z2 = [lm.z for lm in next_landmarks]

    y1, z1 = z1, [-yi for yi in y1]
    y2, z2 = z2, [-yi for yi in y2]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x1, y1, z1, c="blue", label="Prev")
    ax.scatter(x2, y2, z2, c="green", label="Next")

    if landmark_type == "pose":
        connections = POSE_CONNECTIONS
    else:
        connections = HAND_CONNECTIONS

    # Draw prev-frame skeleton
    for start, end in connections:
        ax.plot([x1[start], x1[end]], [y1[start], y1[end]], [z1[start], z1[end]], "b--")

    # Draw next-frame skeleton
    for start, end in connections:
        ax.plot([x2[start], x2[end]], [y2[start], y2[end]], [z2[start], z2[end]], "g--")

    # Draw motion arrows
    for i in landmark_differences:
        ax.quiver(
            x1[i],
            y1[i],
            z1[i],
            x2[i] - x1[i],
            y2[i] - y1[i],
            z2[i] - z1[i],
            color="red",
            linewidth=2,
            alpha=0.6,
            arrow_length_ratio=0.4,
        )

    ax.view_init(elev=10, azim=-70)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    ax.legend()
    plt.tight_layout()
    plt.show()
