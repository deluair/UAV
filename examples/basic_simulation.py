"""
Example: Basic vehicle simulation
"""

import numpy as np
import matplotlib.pyplot as plt
from uav.core.vehicle import AerialVehicle, GroundVehicle, UnderwaterVehicle, State, VehicleConfig
from uav.simulation.simulator import Simulator
from uav.core.environment import Environment, Obstacle


def basic_aerial_simulation():
    """Basic aerial vehicle simulation."""
    print("Running basic aerial vehicle simulation...")
    
    # Create vehicle configuration
    config = VehicleConfig(
        mass=1.5,
        max_speed=20.0,
        battery_level=100.0
    )
    
    # Create initial state
    initial_state = State(
        position=np.array([0, 0, 10]),
        velocity=np.array([0, 0, 0]),
        orientation=np.array([0, 0, 0])
    )
    
    # Create aerial vehicle
    drone = AerialVehicle(
        initial_state=initial_state,
        config=config,
        vehicle_id="drone_1"
    )
    
    # Simple control policy: move forward
    def control_policy(state, t):
        # Simple forward motion
        return np.array([15.0, 0.0, 0.1, 0.0])  # [thrust, roll, pitch, yaw]
    
    # Create simulator
    simulator = Simulator(drone, control_policy=control_policy)
    
    # Run simulation
    history = simulator.run(duration=30.0, dt=0.1, verbose=True)
    
    # Get trajectory
    trajectory = simulator.get_trajectory()
    
    # Plot trajectory
    fig = plt.figure(figsize=(12, 5))
    
    # 2D top view
    ax1 = fig.add_subplot(121)
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2)
    ax1.scatter(trajectory[0, 0], trajectory[0, 1], color='green', s=100, marker='o', label='Start')
    ax1.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', s=100, marker='s', label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Trajectory (Top View)')
    ax1.grid(True)
    ax1.legend()
    ax1.set_aspect('equal')
    
    # Altitude plot
    ax2 = fig.add_subplot(122)
    times = [h['timestamp'] for h in history]
    altitudes = [h['state'].position[2] for h in history]
    ax2.plot(times, altitudes, 'b-', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Altitude (m)')
    ax2.set_title('Altitude vs Time')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('examples/basic_aerial_simulation.png', dpi=150)
    print(f"Simulation complete. Final position: {drone.get_position()}")
    print(f"Saved plot to examples/basic_aerial_simulation.png")


def simulation_with_obstacles():
    """Simulation with obstacles."""
    print("\nRunning simulation with obstacles...")
    
    # Create environment with obstacles
    obstacles = [
        Obstacle(position=np.array([50, 0, 5]), radius=10.0, height=20.0),
        Obstacle(position=np.array([100, 50, 5]), radius=15.0, height=25.0)
    ]
    
    environment = Environment(obstacles=obstacles)
    
    # Create vehicle
    drone = AerialVehicle(
        initial_state=State(position=np.array([0, 0, 10])),
        vehicle_id="drone_2"
    )
    
    # Control policy: move forward
    def control_policy(state, t):
        return np.array([15.0, 0.0, 0.1, 0.0])
    
    # Create simulator with environment
    simulator = Simulator(drone, control_policy=control_policy, environment=environment)
    
    # Run simulation
    history = simulator.run(duration=60.0, dt=0.1, verbose=True)
    
    trajectory = simulator.get_trajectory()
    
    # Plot with obstacles
    fig, ax = plt.subplots(figsize=(10, 10))
    
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
    
    # Plot trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Trajectory')
    ax.scatter(trajectory[0, 0], trajectory[0, 1], color='green', s=100, marker='o', label='Start')
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', s=100, marker='s', label='End')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Trajectory with Obstacles')
    ax.grid(True)
    ax.legend()
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('examples/simulation_with_obstacles.png', dpi=150)
    print(f"Saved plot to examples/simulation_with_obstacles.png")


if __name__ == "__main__":
    # Create examples directory if it doesn't exist
    import os
    os.makedirs('examples', exist_ok=True)
    
    # Run examples
    basic_aerial_simulation()
    simulation_with_obstacles()
    
    print("\nExamples completed!")

