"""
Visualization tools for UAV simulations.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Optional, Tuple
from ..core.vehicle import State


class TrajectoryVisualizer:
    """Visualize vehicle trajectories."""
    
    @staticmethod
    def plot_2d_trajectory(
        trajectory: np.ndarray,
        ax: Optional[plt.Axes] = None,
        label: Optional[str] = None,
        color: str = 'blue'
    ):
        """
        Plot 2D trajectory (x-y projection).
        
        Args:
            trajectory: Array of positions
            ax: Matplotlib axes (creates new if None)
            label: Label for trajectory
            color: Color for trajectory
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, label=label, linewidth=2)
        ax.scatter(trajectory[0, 0], trajectory[0, 1], color='green', s=100, marker='o', label='Start')
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', s=100, marker='s', label='End')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Vehicle Trajectory')
        ax.grid(True)
        ax.legend()
        ax.set_aspect('equal')
        
        return ax
    
    @staticmethod
    def plot_3d_trajectory(
        trajectory: np.ndarray,
        ax: Optional[Axes3D] = None,
        label: Optional[str] = None,
        color: str = 'blue'
    ):
        """
        Plot 3D trajectory.
        
        Args:
            trajectory: Array of positions
            ax: Matplotlib 3D axes (creates new if None)
            label: Label for trajectory
            color: Color for trajectory
        """
        if ax is None:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
        
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                color=color, label=label, linewidth=2)
        ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2],
                  color='green', s=100, marker='o', label='Start')
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2],
                  color='red', s=100, marker='s', label='End')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Vehicle Trajectory (3D)')
        ax.legend()
        
        return ax
    
    @staticmethod
    def plot_multiple_trajectories(
        trajectories: Dict[str, np.ndarray],
        plot_3d: bool = False
    ):
        """
        Plot multiple trajectories.
        
        Args:
            trajectories: Dictionary of vehicle_id -> trajectory
            plot_3d: Whether to plot in 3D
        """
        if plot_3d:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
        
        for (vehicle_id, trajectory), color in zip(trajectories.items(), colors):
            if plot_3d:
                TrajectoryVisualizer.plot_3d_trajectory(
                    trajectory, ax, label=vehicle_id, color=color
                )
            else:
                TrajectoryVisualizer.plot_2d_trajectory(
                    trajectory, ax, label=vehicle_id, color=color
                )
        
        plt.tight_layout()
        return fig, ax


class StateVisualizer:
    """Visualize vehicle states over time."""
    
    @staticmethod
    def plot_state_history(
        states: List[State],
        figsize: Tuple[int, int] = (15, 10)
    ):
        """
        Plot vehicle state history.
        
        Args:
            states: List of states
            figsize: Figure size
        """
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        
        timestamps = [s.timestamp for s in states]
        positions = np.array([s.position for s in states])
        velocities = np.array([s.velocity for s in states])
        orientations = np.array([s.orientation for s in states])
        
        # Position plots
        axes[0, 0].plot(timestamps, positions[:, 0], label='X')
        axes[0, 0].plot(timestamps, positions[:, 1], label='Y')
        axes[0, 0].plot(timestamps, positions[:, 2], label='Z')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Position (m)')
        axes[0, 0].set_title('Position vs Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Velocity plots
        axes[0, 1].plot(timestamps, velocities[:, 0], label='Vx')
        axes[0, 1].plot(timestamps, velocities[:, 1], label='Vy')
        axes[0, 1].plot(timestamps, velocities[:, 2], label='Vz')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Velocity (m/s)')
        axes[0, 1].set_title('Velocity vs Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Orientation plots
        axes[1, 0].plot(timestamps, orientations[:, 0], label='Roll')
        axes[1, 0].plot(timestamps, orientations[:, 1], label='Pitch')
        axes[1, 0].plot(timestamps, orientations[:, 2], label='Yaw')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Angle (rad)')
        axes[1, 0].set_title('Orientation vs Time')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Speed plot
        speeds = np.linalg.norm(velocities, axis=1)
        axes[1, 1].plot(timestamps, speeds)
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Speed (m/s)')
        axes[1, 1].set_title('Speed vs Time')
        axes[1, 1].grid(True)
        
        # Trajectory 2D
        axes[2, 0].plot(positions[:, 0], positions[:, 1])
        axes[2, 0].scatter(positions[0, 0], positions[0, 1], color='green', s=100, marker='o')
        axes[2, 0].scatter(positions[-1, 0], positions[-1, 1], color='red', s=100, marker='s')
        axes[2, 0].set_xlabel('X (m)')
        axes[2, 0].set_ylabel('Y (m)')
        axes[2, 0].set_title('Trajectory (X-Y)')
        axes[2, 0].grid(True)
        axes[2, 0].set_aspect('equal')
        
        # Altitude plot
        axes[2, 1].plot(timestamps, positions[:, 2])
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].set_ylabel('Altitude (m)')
        axes[2, 1].set_title('Altitude vs Time')
        axes[2, 1].grid(True)
        
        plt.tight_layout()
        return fig, axes

