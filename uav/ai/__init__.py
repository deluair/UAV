"""
AI/ML components for path planning and autonomous navigation.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from ..core.vehicle import State
from ..core.environment import Environment


class PathPlanner:
    """Base class for path planning algorithms."""
    
    def __init__(self, environment: Optional[Environment] = None):
        self.environment = environment
        
    def plan(self, start: np.ndarray, goal: np.ndarray, **kwargs) -> List[np.ndarray]:
        raise NotImplementedError


class AStarPlanner(PathPlanner):
    """A* path planning algorithm."""
    
    def __init__(self, environment: Optional[Environment] = None, resolution: float = 5.0):
        super().__init__(environment)
        self.resolution = resolution
        
    def plan(self, start: np.ndarray, goal: np.ndarray, **kwargs) -> List[np.ndarray]:
        if self.environment and self.environment.is_path_clear(start, goal):
            return [start, goal]
        
        num_waypoints = max(3, int(np.linalg.norm(goal - start) / self.resolution))
        waypoints = []
        
        for i in range(num_waypoints + 1):
            alpha = i / num_waypoints
            waypoint = start + alpha * (goal - start)
            
            if self.environment:
                nearby = self.environment.get_nearby_obstacles(waypoint, self.resolution)
                if nearby:
                    direction = goal - start
                    direction = direction / (np.linalg.norm(direction) + 1e-6)
                    perpendicular = np.array([-direction[1], direction[0], 0])
                    waypoint += perpendicular * self.resolution
            
            waypoints.append(waypoint)
        
        waypoints.append(goal)
        return waypoints


class RRTPlanner(PathPlanner):
    """Rapidly-exploring Random Tree (RRT) planner."""
    
    def __init__(self, environment: Optional[Environment] = None, max_iterations: int = 1000, step_size: float = 1.0):
        super().__init__(environment)
        self.max_iterations = max_iterations
        self.step_size = step_size
        
    def plan(self, start: np.ndarray, goal: np.ndarray, **kwargs) -> List[np.ndarray]:
        tree = {tuple(start): None}
        nodes = [start]
        
        for _ in range(self.max_iterations):
            if np.random.random() < 0.1:
                target = goal
            else:
                if self.environment:
                    target = self.environment.get_free_space_sample()
                else:
                    target = np.random.uniform(-100, 100, 3)
            
            nearest_idx = np.argmin([np.linalg.norm(node - target) for node in nodes])
            nearest = nodes[nearest_idx]
            
            direction = target - nearest
            distance = np.linalg.norm(direction)
            if distance > 0:
                direction = direction / distance
                new_node = nearest + direction * min(self.step_size, distance)
                
                if self.environment is None or self.environment.is_path_clear(nearest, new_node):
                    nodes.append(new_node)
                    tree[tuple(new_node)] = tuple(nearest)
                    
                    if np.linalg.norm(new_node - goal) < self.step_size:
                        path = [goal]
                        current = tuple(new_node)
                        while current is not None:
                            path.insert(0, np.array(current))
                            current = tree.get(current)
                        return path
        
        return [start, goal]


class PerceptionModule:
    """Perception module for object detection and state estimation."""
    
    def __init__(self):
        self.detected_objects: List[Dict] = []
        
    def process_sensor_data(self, sensor_readings: Dict, vehicle_state: State) -> Dict:
        detected = {"objects": [], "obstacles": [], "landmarks": []}
        
        if "LiDAR" in sensor_readings:
            lidar_data = sensor_readings["LiDAR"].data
            for i, range_val in enumerate(lidar_data):
                if range_val < 50.0:
                    angle = i * (2 * np.pi / len(lidar_data))
                    obstacle_pos = vehicle_state.position + np.array([
                        range_val * np.cos(angle),
                        range_val * np.sin(angle),
                        0
                    ])
                    detected["obstacles"].append({
                        "position": obstacle_pos,
                        "distance": range_val,
                        "angle": angle
                    })
        
        if "Camera" in sensor_readings:
            detected["objects"].append({
                "type": "unknown",
                "confidence": 0.5,
                "position": vehicle_state.position + np.array([10, 0, 0])
            })
        
        self.detected_objects = detected["objects"]
        return detected
    
    def estimate_state(self, sensor_readings: Dict) -> Optional[State]:
        if "GPS" in sensor_readings:
            position = sensor_readings["GPS"].data.copy()
        else:
            position = np.zeros(3)
        
        if "IMU" in sensor_readings:
            imu_data = sensor_readings["IMU"].data
            acceleration = imu_data[:3]
            angular_velocity = imu_data[3:6]
        else:
            acceleration = np.zeros(3)
            angular_velocity = np.zeros(3)
        
        estimated_state = State(
            position=position,
            acceleration=acceleration,
            angular_velocity=angular_velocity,
            timestamp=sensor_readings.get("GPS", sensor_readings.get("IMU")).timestamp
        )
        
        return estimated_state


class NavigationController:
    """Navigation controller for following paths."""
    
    def __init__(self, lookahead_distance: float = 5.0):
        self.lookahead_distance = lookahead_distance
        self.current_path: List[np.ndarray] = []
        self.path_index = 0
        
    def set_path(self, path: List[np.ndarray]):
        self.current_path = path
        self.path_index = 0
        
    def compute_control(self, current_state: State, target_position: Optional[np.ndarray] = None) -> np.ndarray:
        if target_position is not None:
            target = target_position
        elif self.current_path:
            if self.path_index < len(self.current_path):
                target = self.current_path[self.path_index]
                
                distance = np.linalg.norm(current_state.position - target)
                if distance < self.lookahead_distance:
                    self.path_index += 1
                    if self.path_index < len(self.current_path):
                        target = self.current_path[self.path_index]
            else:
                target = self.current_path[-1]
        else:
            return np.zeros(4)
        
        error = target - current_state.position
        direction = error / (np.linalg.norm(error) + 1e-6)
        
        desired_velocity = direction * 5.0
        
        control = np.array([
            15.0,
            direction[1] * 0.5,
            direction[0] * 0.5,
            0.0
        ])
        
        return control

__all__ = [
    "PathPlanner",
    "AStarPlanner",
    "RRTPlanner",
    "PerceptionModule",
    "NavigationController",
]
