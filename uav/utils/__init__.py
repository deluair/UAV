"""
Utility functions for UAV systems.
"""

import numpy as np
from typing import List, Tuple, Optional
from ..core.vehicle import State


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert Euler angles to quaternion.
    
    Args:
        roll: Roll angle in radians
        pitch: Pitch angle in radians
        yaw: Yaw angle in radians
        
    Returns:
        Quaternion [w, x, y, z]
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([w, x, y, z])


def quaternion_to_euler(q: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert quaternion to Euler angles.
    
    Args:
        q: Quaternion [w, x, y, z]
        
    Returns:
        Tuple of (roll, pitch, yaw) in radians
    """
    w, x, y, z = q
    
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    pitch = np.arcsin(2 * (w * y - z * x))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    
    return roll, pitch, yaw


def wrap_angle(angle: float) -> float:
    """
    Wrap angle to [-pi, pi] range.
    
    Args:
        angle: Angle in radians
        
    Returns:
        Wrapped angle
    """
    return np.arctan2(np.sin(angle), np.cos(angle))


def distance_between_points(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        p1: First point
        p2: Second point
        
    Returns:
        Distance
    """
    return np.linalg.norm(p1 - p2)


def bearing_between_points(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Calculate bearing from point 1 to point 2.
    
    Args:
        p1: First point
        p2: Second point
        
    Returns:
        Bearing in radians
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.arctan2(dy, dx)


def interpolate_trajectory(
    waypoints: List[np.ndarray],
    num_points: int
) -> np.ndarray:
    """
    Interpolate waypoints to create smooth trajectory.
    
    Args:
        waypoints: List of waypoints
        num_points: Number of points in interpolated trajectory
        
    Returns:
        Array of interpolated positions
    """
    if len(waypoints) < 2:
        return np.array(waypoints)
    
    trajectory = []
    total_distance = 0.0
    
    # Calculate cumulative distances
    distances = [0.0]
    for i in range(1, len(waypoints)):
        dist = distance_between_points(waypoints[i-1], waypoints[i])
        total_distance += dist
        distances.append(total_distance)
    
    # Interpolate
    for i in range(num_points):
        alpha = i / (num_points - 1) if num_points > 1 else 0.0
        target_distance = alpha * total_distance
        
        # Find segment
        segment_idx = 0
        for j in range(len(distances) - 1):
            if distances[j] <= target_distance <= distances[j + 1]:
                segment_idx = j
                break
        
        # Interpolate within segment
        if segment_idx < len(waypoints) - 1:
            seg_start = distances[segment_idx]
            seg_end = distances[segment_idx + 1]
            seg_alpha = (target_distance - seg_start) / (seg_end - seg_start) if seg_end > seg_start else 0.0
            
            point = waypoints[segment_idx] + seg_alpha * (
                waypoints[segment_idx + 1] - waypoints[segment_idx]
            )
            trajectory.append(point)
        else:
            trajectory.append(waypoints[-1])
    
    return np.array(trajectory)


def calculate_trajectory_statistics(
    trajectory: np.ndarray,
    timestamps: Optional[np.ndarray] = None
) -> dict:
    """
    Calculate statistics for a trajectory.
    
    Args:
        trajectory: Array of positions
        timestamps: Optional timestamps
        
    Returns:
        Dictionary with statistics
    """
    if len(trajectory) < 2:
        return {}
    
    # Calculate distances between consecutive points
    distances = []
    for i in range(1, len(trajectory)):
        dist = distance_between_points(trajectory[i-1], trajectory[i])
        distances.append(dist)
    
    distances = np.array(distances)
    
    # Calculate velocities if timestamps provided
    velocities = []
    if timestamps is not None and len(timestamps) == len(trajectory):
        for i in range(1, len(trajectory)):
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                velocity = distances[i-1] / dt
                velocities.append(velocity)
    
    stats = {
        "total_distance": np.sum(distances),
        "mean_distance": np.mean(distances),
        "max_distance": np.max(distances),
        "min_distance": np.min(distances),
        "num_waypoints": len(trajectory),
        "trajectory_length": len(trajectory)
    }
    
    if velocities:
        stats["mean_velocity"] = np.mean(velocities)
        stats["max_velocity"] = np.max(velocities)
        stats["min_velocity"] = np.min(velocities)
    
    return stats


def calculate_heading_error(
    current_heading: float,
    target_heading: float
) -> float:
    """
    Calculate heading error (wrapped to [-pi, pi]).
    
    Args:
        current_heading: Current heading in radians
        target_heading: Target heading in radians
        
    Returns:
        Heading error in radians
    """
    error = target_heading - current_heading
    return wrap_angle(error)


def smooth_control_input(
    control_input: np.ndarray,
    previous_control: Optional[np.ndarray] = None,
    max_rate: float = 5.0
) -> np.ndarray:
    """
    Smooth control input to limit rate of change.
    
    Args:
        control_input: Desired control input
        previous_control: Previous control input
        max_rate: Maximum rate of change
        
    Returns:
        Smoothed control input
    """
    if previous_control is None:
        return control_input
    
    rate = control_input - previous_control
    rate = np.clip(rate, -max_rate, max_rate)
    
    return previous_control + rate

