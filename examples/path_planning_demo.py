"""
Example: Path planning and navigation
"""

import numpy as np
import matplotlib.pyplot as plt
from uav.core.vehicle import AerialVehicle, State
from uav.core.environment import Environment, Obstacle
from uav.ai import AStarPlanner, RRTPlanner, NavigationController
from uav.simulation.simulator import Simulator


def path_planning_demo():
    """Demonstrate path planning algorithms."""
    print("Running path planning demo...")
    
    # Create environment with obstacles
    obstacles = [
        Obstacle(position=np.array([30, 20, 5]), radius=8.0, height=20.0),
        Obstacle(position=np.array([60, 50, 5]), radius=10.0, height=25.0),
        Obstacle(position=np.array([80, 30, 5]), radius=6.0, height=15.0)
    ]
    
    environment = Environment(obstacles=obstacles)
    
    # Start and goal positions
    start = np.array([0, 0, 10])
    goal = np.array([100, 80, 15])
    
    # Try A* planner
    print("Planning with A*...")
    astar_planner = AStarPlanner(environment, resolution=5.0)
    astar_path = astar_planner.plan(start, goal)
    
    # Try RRT planner
    print("Planning with RRT...")
    rrt_planner = RRTPlanner(environment, max_iterations=500, step_size=3.0)
    rrt_path = rrt_planner.plan(start, goal)
    
    # Create vehicle
    drone = AerialVehicle(
        initial_state=State(position=start),
        vehicle_id="drone_path"
    )
    
    # Create navigation controller
    nav_controller = NavigationController(lookahead_distance=5.0)
    nav_controller.set_path(astar_path)
    
    # Control policy using navigation controller
    def control_policy(state, t):
        return nav_controller.compute_control(state)
    
    # Simulate following the path
    simulator = Simulator(drone, control_policy=control_policy, environment=environment)
    history = simulator.run(duration=60.0, dt=0.1, verbose=True)
    
    trajectory = simulator.get_trajectory()
    
    # Plot results
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot obstacles
    for obs in obstacles:
        circle = plt.Circle(
            (obs.position[0], obs.position[1]),
            obs.radius,
            color='red',
            alpha=0.5,
            label='Obstacle'
        )
        ax.add_patch(circle)
    
    # Plot planned paths
    astar_path_array = np.array(astar_path)
    ax.plot(astar_path_array[:, 0], astar_path_array[:, 1], 
            'g--', linewidth=2, label='A* Path', markersize=8)
    ax.scatter(astar_path_array[:, 0], astar_path_array[:, 1], 
              color='green', s=50, marker='x')
    
    rrt_path_array = np.array(rrt_path)
    ax.plot(rrt_path_array[:, 0], rrt_path_array[:, 1], 
            'm--', linewidth=2, label='RRT Path', alpha=0.7)
    
    # Plot actual trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], 
            'b-', linewidth=2, label='Actual Trajectory')
    
    # Start and goal
    ax.scatter(start[0], start[1], color='green', s=200, marker='o', label='Start', zorder=5)
    ax.scatter(goal[0], goal[1], color='red', s=200, marker='s', label='Goal', zorder=5)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Path Planning Demo')
    ax.grid(True)
    ax.legend()
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('examples/path_planning_demo.png', dpi=150)
    
    print(f"A* path length: {len(astar_path)} waypoints")
    print(f"RRT path length: {len(rrt_path)} waypoints")
    print(f"Saved plot to examples/path_planning_demo.png")


if __name__ == "__main__":
    import os
    os.makedirs('examples', exist_ok=True)
    
    path_planning_demo()
    print("\nPath planning demo completed!")

