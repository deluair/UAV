"""
Environment representation for UAV simulations.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Obstacle:
    """Obstacle representation."""
    position: np.ndarray
    radius: float
    height: Optional[float] = None  # For 3D obstacles
    obstacle_type: str = "sphere"  # "sphere", "cylinder", "box"
    
    def contains(self, point: np.ndarray) -> bool:
        """Check if point is inside obstacle."""
        if self.obstacle_type == "sphere":
            distance = np.linalg.norm(point - self.position)
            return distance <= self.radius
        elif self.obstacle_type == "cylinder":
            horizontal_distance = np.linalg.norm(point[:2] - self.position[:2])
            if horizontal_distance > self.radius:
                return False
            if self.height is not None:
                z_min = self.position[2] - self.height / 2
                z_max = self.position[2] + self.height / 2
                return z_min <= point[2] <= z_max
            return True
        elif self.obstacle_type == "box":
            half_size = self.radius  # Treat radius as half-size
            return np.all(np.abs(point - self.position) <= half_size)
        return False


class Environment:
    """Environment representation with obstacles and boundaries."""
    
    def __init__(
        self,
        boundaries: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        obstacles: Optional[List[Obstacle]] = None
    ):
        """
        Initialize environment.
        
        Args:
            boundaries: (min_bound, max_bound) tuple defining space limits
            obstacles: List of obstacles in the environment
        """
        self.boundaries = boundaries or (
            np.array([-1000, -1000, -500]),
            np.array([1000, 1000, 400])
        )
        self.obstacles = obstacles or []
        
    def add_obstacle(self, obstacle: Obstacle):
        """Add obstacle to environment."""
        self.obstacles.append(obstacle)
    
    def check_collision(self, position: np.ndarray) -> bool:
        """
        Check if position collides with any obstacle or boundary.
        
        Args:
            position: Position to check
            
        Returns:
            True if collision detected
        """
        # Check boundaries
        min_bound, max_bound = self.boundaries
        if np.any(position < min_bound) or np.any(position > max_bound):
            return True
        
        # Check obstacles
        for obstacle in self.obstacles:
            if obstacle.contains(position):
                return True
        
        return False
    
    def get_nearby_obstacles(
        self,
        position: np.ndarray,
        radius: float
    ) -> List[Obstacle]:
        """
        Get obstacles within specified radius of position.
        
        Args:
            position: Position to check around
            radius: Search radius
            
        Returns:
            List of nearby obstacles
        """
        nearby = []
        for obstacle in self.obstacles:
            distance = np.linalg.norm(position - obstacle.position)
            if distance <= radius + obstacle.radius:
                nearby.append(obstacle)
        return nearby
    
    def is_path_clear(
        self,
        start: np.ndarray,
        end: np.ndarray,
        num_samples: int = 100
    ) -> bool:
        """
        Check if straight-line path is clear of obstacles.
        
        Args:
            start: Start position
            end: End position
            num_samples: Number of points to sample along path
            
        Returns:
            True if path is clear
        """
        for i in range(num_samples + 1):
            alpha = i / num_samples
            point = start + alpha * (end - start)
            if self.check_collision(point):
                return False
        return True
    
    def get_free_space_sample(self, num_samples: int = 1) -> np.ndarray:
        """
        Sample random positions in free space.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Array of sampled positions
        """
        min_bound, max_bound = self.boundaries
        samples = []
        
        while len(samples) < num_samples:
            sample = np.random.uniform(min_bound, max_bound)
            if not self.check_collision(sample):
                samples.append(sample)
        
        return np.array(samples) if num_samples > 1 else np.array(samples[0])

