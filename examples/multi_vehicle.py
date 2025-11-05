"""
Example: Multi-vehicle simulation
"""

import numpy as np
import matplotlib.pyplot as plt
from uav.core.vehicle import AerialVehicle, State, VehicleConfig
from uav.simulation.simulator import MultiVehicleSimulator
from uav.communication import FleetManager, CommunicationChannel


def multi_vehicle_simulation():
    """Simulate multiple vehicles."""
    print("Running multi-vehicle simulation...")
    
    # Create multiple vehicles
    vehicles = []
    num_vehicles = 3
    
    for i in range(num_vehicles):
        initial_state = State(
            position=np.array([i * 10, 0, 10]),
            velocity=np.array([0, 0, 0])
        )
        
        vehicle = AerialVehicle(
            initial_state=initial_state,
            config=VehicleConfig(max_speed=15.0),
            vehicle_id=f"drone_{i+1}"
        )
        vehicles.append(vehicle)
    
    # Create communication channel
    comm_channel = CommunicationChannel(range=500.0)
    
    # Create fleet manager
    fleet_manager = FleetManager(vehicles, comm_channel)
    
    # Control policy for each vehicle
    def control_policy(state, t, all_vehicles):
        # Simple forward motion with slight offset
        vehicle_id = None
        for vid, v in all_vehicles.items():
            if np.allclose(v.get_position(), state.position, atol=0.1):
                vehicle_id = vid
                break
        
        # Different behavior for each vehicle
        if vehicle_id == "drone_1":
            return np.array([15.0, 0.0, 0.1, 0.0])
        elif vehicle_id == "drone_2":
            return np.array([15.0, 0.0, 0.1, 0.05])
        else:
            return np.array([15.0, 0.0, 0.1, -0.05])
    
    control_policies = {v.vehicle_id: control_policy for v in vehicles}
    
    # Create multi-vehicle simulator
    simulator = MultiVehicleSimulator(
        vehicles=vehicles,
        control_policies=control_policies
    )
    
    # Run simulation
    history = simulator.run(duration=30.0, dt=0.1, verbose=True)
    
    # Get trajectories
    trajectories = simulator.get_trajectories()
    
    # Plot trajectories
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = ['blue', 'green', 'orange', 'purple', 'brown']
    
    for i, (vehicle_id, trajectory) in enumerate(trajectories.items()):
        color = colors[i % len(colors)]
        ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, linewidth=2, label=vehicle_id)
        ax.scatter(trajectory[0, 0], trajectory[0, 1], color=color, s=100, marker='o')
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color=color, s=100, marker='s')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Multi-Vehicle Trajectories')
    ax.grid(True)
    ax.legend()
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('examples/multi_vehicle_simulation.png', dpi=150)
    
    # Print fleet status
    status = fleet_manager.get_fleet_status()
    print("\nFleet Status:")
    for vid, stat in status.items():
        print(f"  {vid}: Position {stat['position']}, Battery {stat['battery_level']:.1f}%")
    
    print(f"\nSaved plot to examples/multi_vehicle_simulation.png")


if __name__ == "__main__":
    import os
    os.makedirs('examples', exist_ok=True)
    
    multi_vehicle_simulation()
    print("\nMulti-vehicle simulation completed!")

