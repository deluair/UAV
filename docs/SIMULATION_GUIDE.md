# Simulation Guide

## Overview

The UAV simulation system provides comprehensive simulation capabilities for aerial, ground, and underwater vehicles. This guide covers how to use the simulation components.

## Basic Simulation

### Single Vehicle Simulation

```python
from uav.core.vehicle import AerialVehicle, State
from uav.simulation.simulator import Simulator

# Create vehicle
drone = AerialVehicle(
    initial_state=State(position=np.array([0, 0, 10]))
)

# Define control policy
def control_policy(state, t):
    # Return control inputs [thrust, roll_rate, pitch_rate, yaw_rate]
    return np.array([15.0, 0.0, 0.1, 0.0])

# Create simulator
simulator = Simulator(drone, control_policy=control_policy)

# Run simulation
history = simulator.run(duration=30.0, dt=0.1)

# Get results
trajectory = simulator.get_trajectory()
velocities = simulator.get_velocities()
```

### Multi-Vehicle Simulation

```python
from uav.simulation.simulator import MultiVehicleSimulator

# Create multiple vehicles
vehicles = [
    AerialVehicle(initial_state=State(position=np.array([0, 0, 10])), vehicle_id="drone_1"),
    AerialVehicle(initial_state=State(position=np.array([10, 0, 10])), vehicle_id="drone_2")
]

# Define control policies
def control_policy(state, t, all_vehicles):
    return np.array([15.0, 0.0, 0.1, 0.0])

control_policies = {v.vehicle_id: control_policy for v in vehicles}

# Create simulator
simulator = MultiVehicleSimulator(vehicles, control_policies=control_policies)

# Run simulation
history = simulator.run(duration=30.0, dt=0.1)

# Get trajectories for all vehicles
trajectories = simulator.get_trajectories()
```

## Environment Setup

### Creating Environments with Obstacles

```python
from uav.core.environment import Environment, Obstacle

# Create obstacles
obstacles = [
    Obstacle(position=np.array([50, 0, 5]), radius=10.0, height=20.0),
    Obstacle(position=np.array([100, 50, 5]), radius=15.0, height=25.0)
]

# Create environment
environment = Environment(obstacles=obstacles)

# Check collisions
collision = environment.check_collision(position)

# Check if path is clear
is_clear = environment.is_path_clear(start_position, end_position)
```

### Custom Boundaries

```python
min_bound = np.array([-1000, -1000, -500])
max_bound = np.array([1000, 1000, 400])
environment = Environment(boundaries=(min_bound, max_bound))
```

## Control Policies

### Simple Control Policy

```python
def simple_control(state, t):
    # Move forward at constant speed
    return np.array([15.0, 0.0, 0.1, 0.0])
```

### Waypoint Following

```python
from uav.control import FlightController

controller = FlightController()

def waypoint_control(state, t):
    target = np.array([100, 100, 15])
    return controller.compute_control(state, target, dt=0.1)
```

### Dynamic Control Based on Conditions

```python
def adaptive_control(state, t):
    # Adjust control based on battery level
    if state.vehicle.config.battery_level < 20:
        # Return to base
        return return_to_base_control(state)
    else:
        # Continue mission
        return mission_control(state, t)
```

## Sensor Integration

### Adding Sensors to Simulation

```python
from uav.sensors import GPS, IMU, SensorSuite

# Create sensors
gps = GPS(noise_std=1.0, update_rate=1.0)
imu = IMU(accel_noise_std=0.1, update_rate=100.0)

sensor_suite = SensorSuite([gps, imu])

# In simulation loop
for step in simulator.simulation_history:
    readings = sensor_suite.read_all(step['state'], step['timestamp'])
    # Process sensor readings
```

## Advanced Features

### Custom Vehicle Dynamics

You can subclass vehicle classes to implement custom dynamics:

```python
class CustomVehicle(AerialVehicle):
    def dynamics(self, state, control_input, dt):
        # Custom dynamics implementation
        new_state = state.copy()
        # ... custom dynamics ...
        return new_state
```

### Event Handling

```python
class EventSimulator(Simulator):
    def __init__(self, vehicle, control_policy, events=None):
        super().__init__(vehicle, control_policy)
        self.events = events or []
    
    def run(self, duration, dt=0.1):
        # Override to handle events
        for event in self.events:
            if event.should_trigger(self.vehicle.state):
                event.execute(self.vehicle)
        return super().run(duration, dt)
```

## Performance Tips

1. **Time Step**: Use appropriate `dt` values (0.01-0.1 seconds typically)
2. **History Size**: Limit history size for long simulations
3. **Parallelization**: Use multi-vehicle simulator for parallel processing
4. **Vectorization**: Use NumPy vectorized operations when possible

## Common Issues

### Collision Detection

If vehicles are colliding unexpectedly:
- Check obstacle sizes and positions
- Verify vehicle dimensions
- Adjust collision margins

### Control Instability

If control is unstable:
- Reduce control gains
- Increase time step
- Add control smoothing

### Performance Issues

For performance optimization:
- Reduce simulation frequency
- Use simpler dynamics models
- Limit visualization updates

