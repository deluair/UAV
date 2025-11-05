"""
Example: Trajectory forecasting
"""

import numpy as np
import matplotlib.pyplot as plt
from uav.core.vehicle import AerialVehicle, State
from uav.projection import TrajectoryForecaster, TrajectoryPredictor
from uav.simulation.simulator import Simulator


def forecasting_demo():
    """Demonstrate trajectory forecasting."""
    print("Running trajectory forecasting demo...")
    
    # Create vehicle and simulate some history
    drone = AerialVehicle(
        initial_state=State(position=np.array([0, 0, 10])),
        vehicle_id="drone_forecast"
    )
    
    # Simulate vehicle motion
    def control_policy(state, t):
        # Circular motion
        omega = 0.1
        return np.array([
            15.0,
            0.0,
            0.1 * np.cos(omega * t),
            0.05
        ])
    
    simulator = Simulator(drone, control_policy=control_policy)
    history = simulator.run(duration=20.0, dt=0.1, verbose=False)
    
    # Create forecaster
    forecaster = TrajectoryForecaster(method="linear", horizon=10.0, dt=0.1)
    
    # Update forecaster with history
    for step in history[-20:]:  # Use last 20 states
        forecaster.update_history(step['state'])
    
    # Get current state
    current_state = drone.get_state()
    
    # Forecast using different methods
    print("Forecasting trajectories...")
    
    methods = ["constant_velocity", "linear", "polynomial"]
    forecasts = {}
    
    for method in methods:
        forecaster.method = method
        forecast = forecaster.forecast(current_state, horizon=10.0)
        forecasts[method] = forecast
    
    # Get actual future trajectory (for comparison)
    # Continue simulation for comparison
    actual_history = simulator.run(duration=10.0, dt=0.1, verbose=False)
    actual_future = np.array([h['state'].position for h in actual_history])
    
    # Plot results
    fig = plt.figure(figsize=(15, 5))
    
    # 2D top view
    ax1 = fig.add_subplot(131)
    trajectory = simulator.get_trajectory()
    
    # Plot past trajectory
    ax1.plot(trajectory[:-len(actual_future), 0], 
            trajectory[:-len(actual_future), 1], 
            'b-', linewidth=2, label='Past Trajectory')
    
    # Plot actual future
    ax1.plot(actual_future[:, 0], actual_future[:, 1], 
            'r-', linewidth=2, label='Actual Future', linestyle='--')
    
    # Plot forecasts
    colors = {'constant_velocity': 'green', 'linear': 'orange', 'polynomial': 'purple'}
    for method, forecast in forecasts.items():
        ax1.plot(forecast[:, 0], forecast[:, 1], 
                color=colors[method], linewidth=2, 
                label=f'{method.capitalize()} Forecast', linestyle=':', alpha=0.7)
    
    ax1.scatter(current_state.position[0], current_state.position[1], 
               color='black', s=150, marker='*', label='Current', zorder=5)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Trajectory Forecasting (Top View)')
    ax1.grid(True)
    ax1.legend()
    ax1.set_aspect('equal')
    
    # Altitude comparison
    ax2 = fig.add_subplot(132)
    times = np.arange(len(actual_future)) * 0.1
    
    ax2.plot(times, actual_future[:, 2], 'r-', linewidth=2, label='Actual Future')
    
    for method, forecast in forecasts.items():
        ax2.plot(times, forecast[:, 2], 
                color=colors[method], linewidth=2, 
                label=f'{method.capitalize()} Forecast', linestyle=':', alpha=0.7)
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Altitude (m)')
    ax2.set_title('Altitude Forecasting')
    ax2.grid(True)
    ax2.legend()
    
    # Error analysis
    ax3 = fig.add_subplot(133)
    
    errors = {}
    for method, forecast in forecasts.items():
        if len(forecast) == len(actual_future):
            error = np.linalg.norm(forecast - actual_future, axis=1)
            errors[method] = error
            ax3.plot(times, error, color=colors[method], linewidth=2, label=method)
    
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Forecast Error (m)')
    ax3.set_title('Forecasting Error')
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('examples/forecasting_demo.png', dpi=150)
    
    # Print error statistics
    print("\nForecasting Error Statistics:")
    for method, error in errors.items():
        print(f"  {method}: Mean={np.mean(error):.2f}m, Max={np.max(error):.2f}m")
    
    print(f"Saved plot to examples/forecasting_demo.png")


if __name__ == "__main__":
    import os
    os.makedirs('examples', exist_ok=True)
    
    forecasting_demo()
    print("\nForecasting demo completed!")

