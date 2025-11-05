# Quick Start Guide

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd UAV

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Examples

### 1. Basic Simulation

```python
from uav.core.vehicle import AerialVehicle, State
from uav.simulation.simulator import Simulator
import numpy as np

# Create drone
drone = AerialVehicle(
    initial_state=State(position=np.array([0, 0, 10]))
)

# Simple control: move forward
def control_policy(state, t):
    return np.array([15.0, 0.0, 0.1, 0.0])

# Simulate
simulator = Simulator(drone, control_policy=control_policy)
history = simulator.run(duration=30.0, dt=0.1)

# Visualize
trajectory = simulator.get_trajectory()
print(f"Final position: {drone.get_position()}")
```

### 2. Path Planning

```python
from uav.ai import AStarPlanner
from uav.core.environment import Environment

# Create environment
env = Environment()

# Plan path
planner = AStarPlanner(env)
path = planner.plan(
    start=np.array([0, 0, 10]),
    goal=np.array([100, 100, 15])
)

print(f"Path planned: {len(path)} waypoints")
```

### 3. Multi-Vehicle Simulation

```python
from uav.simulation.simulator import MultiVehicleSimulator
from uav.core.vehicle import AerialVehicle, State

# Create multiple vehicles
vehicles = [
    AerialVehicle(initial_state=State(position=np.array([0, 0, 10])), vehicle_id="drone_1"),
    AerialVehicle(initial_state=State(position=np.array([10, 0, 10])), vehicle_id="drone_2")
]

# Control policy
def control_policy(state, t, all_vehicles):
    return np.array([15.0, 0.0, 0.1, 0.0])

control_policies = {v.vehicle_id: control_policy for v in vehicles}

# Simulate
simulator = MultiVehicleSimulator(vehicles, control_policies=control_policies)
history = simulator.run(duration=30.0, dt=0.1)
```

### 4. Trajectory Forecasting

```python
from uav.projection import TrajectoryForecaster

# Create forecaster
forecaster = TrajectoryForecaster(method="linear", horizon=10.0)

# Update with history
for state in state_history:
    forecaster.update_history(state)

# Forecast
forecast = forecaster.forecast(current_state, horizon=10.0)
print(f"Forecasted {len(forecast)} future positions")
```

### 5. Sensor Integration

```python
from uav.sensors import GPS, IMU, SensorSuite

# Create sensors
gps = GPS(noise_std=1.0, update_rate=1.0)
imu = IMU(accel_noise_std=0.1, update_rate=100.0)

# Create sensor suite
sensor_suite = SensorSuite([gps, imu])

# Read sensors
readings = sensor_suite.read_all(vehicle_state, timestamp)
print(f"GPS reading: {readings.get('GPS').data}")
```

## Running Examples

Run the example scripts:

```bash
# Basic simulation
python examples/basic_simulation.py

# Multi-vehicle simulation
python examples/multi_vehicle.py

# Path planning demo
python examples/path_planning_demo.py

# Forecasting demo
python examples/forecasting_demo.py

# Sensor integration
python examples/sensor_integration.py
```

## Next Steps

1. Read the [Simulation Guide](docs/SIMULATION_GUIDE.md)
2. Explore [AI Navigation](docs/AI_NAVIGATION.md)
3. Check [API Reference](docs/API_REFERENCE.md)
4. Review [Projection Models](docs/PROJECTION_MODELS.md)

## Getting Help

- Check documentation in `docs/` directory
- Review example scripts in `examples/`
- Open an issue on GitHub

