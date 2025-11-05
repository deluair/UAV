"""
Example: Sensor integration demo
"""

import numpy as np
import matplotlib.pyplot as plt
from uav.core.vehicle import AerialVehicle, State
from uav.sensors import GPS, IMU, LiDAR, SensorSuite
from uav.simulation.simulator import Simulator


def sensor_integration_demo():
    """Demonstrate sensor integration."""
    print("Running sensor integration demo...")
    
    # Create vehicle
    drone = AerialVehicle(
        initial_state=State(position=np.array([0, 0, 10])),
        vehicle_id="drone_sensors"
    )
    
    # Create sensors
    gps = GPS(noise_std=1.0, update_rate=1.0)
    imu = IMU(accel_noise_std=0.1, gyro_noise_std=0.01, update_rate=100.0)
    lidar = LiDAR(max_range=100.0, num_beams=180, update_rate=10.0)
    
    # Create sensor suite
    sensor_suite = SensorSuite([gps, imu, lidar])
    
    # Simulate vehicle motion
    def control_policy(state, t):
        return np.array([15.0, 0.0, 0.1, 0.0])
    
    simulator = Simulator(drone, control_policy=control_policy)
    history = simulator.run(duration=20.0, dt=0.01, verbose=False)
    
    # Collect sensor readings
    sensor_readings = []
    true_states = []
    
    for step in history:
        state = step['state']
        timestamp = step['timestamp']
        
        true_states.append(state.position.copy())
        
        # Read sensors
        readings = sensor_suite.read_all(state, timestamp)
        if readings:
            sensor_readings.append({
                'timestamp': timestamp,
                'readings': readings
            })
    
    # Extract GPS readings
    gps_positions = []
    gps_times = []
    
    for reading_data in sensor_readings:
        if 'GPS' in reading_data['readings']:
            gps_positions.append(reading_data['readings']['GPS'].data)
            gps_times.append(reading_data['timestamp'])
    
    gps_positions = np.array(gps_positions)
    gps_times = np.array(gps_times)
    
    # Extract IMU readings
    imu_readings = []
    imu_times = []
    
    for reading_data in sensor_readings:
        if 'IMU' in reading_data['readings']:
            imu_data = reading_data['readings']['IMU'].data
            imu_readings.append(imu_data)
            imu_times.append(reading_data['timestamp'])
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # GPS vs True Position
    true_times = [s['timestamp'] for s in history]
    true_positions = np.array(true_states)
    
    ax = axes[0, 0]
    ax.plot(true_times, true_positions[:, 0], 'b-', label='True X', linewidth=2)
    ax.plot(true_times, true_positions[:, 1], 'g-', label='True Y', linewidth=2)
    ax.plot(gps_times, gps_positions[:, 0], 'b--', label='GPS X', marker='o', markersize=4)
    ax.plot(gps_times, gps_positions[:, 1], 'g--', label='GPS Y', marker='o', markersize=4)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (m)')
    ax.set_title('GPS vs True Position')
    ax.legend()
    ax.grid(True)
    
    # Position Error
    ax = axes[0, 1]
    # Interpolate GPS to match true times
    errors = []
    for i, (t, true_pos) in enumerate(zip(true_times, true_positions)):
        # Find nearest GPS reading
        if len(gps_times) > 0:
            nearest_idx = np.argmin(np.abs(gps_times - t))
            gps_pos = gps_positions[nearest_idx]
            error = np.linalg.norm(true_pos - gps_pos)
            errors.append(error)
    
    if errors:
        ax.plot(true_times[:len(errors)], errors, 'r-', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position Error (m)')
        ax.set_title('GPS Position Error')
        ax.grid(True)
    
    # IMU Acceleration
    ax = axes[1, 0]
    if imu_readings:
        imu_array = np.array(imu_readings)
        imu_times_array = np.array(imu_times)
        
        ax.plot(imu_times_array, imu_array[:, 0], label='Accel X')
        ax.plot(imu_times_array, imu_array[:, 1], label='Accel Y')
        ax.plot(imu_times_array, imu_array[:, 2], label='Accel Z')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Acceleration (m/s²)')
        ax.set_title('IMU Acceleration')
        ax.legend()
        ax.grid(True)
    
    # Trajectory Comparison
    ax = axes[1, 1]
    ax.plot(true_positions[:, 0], true_positions[:, 1], 'b-', label='True Trajectory', linewidth=2)
    ax.plot(gps_positions[:, 0], gps_positions[:, 1], 'r--', label='GPS Trajectory', marker='o', markersize=4)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Trajectory Comparison')
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('examples/sensor_integration_demo.png', dpi=150)
    
    print(f"GPS readings: {len(gps_positions)}")
    print(f"IMU readings: {len(imu_readings)}")
    print(f"Average GPS error: {np.mean(errors):.2f}m")
    print(f"Saved plot to examples/sensor_integration_demo.png")


if __name__ == "__main__":
    import os
    os.makedirs('examples', exist_ok=True)
    
    sensor_integration_demo()
    print("\nSensor integration demo completed!")

