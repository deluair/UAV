# API Reference

## Core Modules

### Vehicle Classes

#### `Vehicle`
Base class for all autonomous vehicles.

**Methods:**
- `get_state() -> State`: Get current vehicle state
- `update_state(new_state: State)`: Update vehicle state
- `dynamics(state: State, control_input: np.ndarray, dt: float) -> State`: Compute vehicle dynamics
- `get_constraints() -> Dict`: Get vehicle-specific constraints

#### `AerialVehicle(Vehicle)`
Aerial vehicle (drone) implementation.

**Initialization:**
```python
drone = AerialVehicle(
    initial_state=State(position=np.array([0, 0, 10])),
    config=VehicleConfig(max_speed=20.0),
    vehicle_id="drone_1"
)
```

#### `GroundVehicle(Vehicle)`
Ground vehicle (UGV) implementation.

#### `UnderwaterVehicle(Vehicle)`
Underwater vehicle (UUV) implementation.

### State

Represents vehicle state with position, velocity, acceleration, orientation, and angular velocity.

**Attributes:**
- `position: np.ndarray`: 3D position [x, y, z]
- `velocity: np.ndarray`: 3D velocity [vx, vy, vz]
- `acceleration: np.ndarray`: 3D acceleration [ax, ay, az]
- `orientation: np.ndarray`: Euler angles [roll, pitch, yaw]
- `angular_velocity: np.ndarray`: Angular velocity [wx, wy, wz]
- `timestamp: float`: Timestamp

## Simulation

### `Simulator`

Simulation engine for single vehicle.

**Usage:**
```python
simulator = Simulator(vehicle, control_policy=control_policy)
history = simulator.run(duration=30.0, dt=0.1)
trajectory = simulator.get_trajectory()
```

### `MultiVehicleSimulator`

Simulation engine for multiple vehicles.

**Usage:**
```python
simulator = MultiVehicleSimulator(vehicles, control_policies=control_policies)
history = simulator.run(duration=30.0, dt=0.1)
trajectories = simulator.get_trajectories()
```

## Sensors

### Available Sensors

- `GPS`: GPS sensor for position
- `IMU`: Inertial Measurement Unit
- `LiDAR`: LiDAR sensor for distance measurement
- `Camera`: Camera sensor
- `Sonar`: Sonar sensor for underwater vehicles
- `SensorSuite`: Collection of sensors

**Usage:**
```python
from uav.sensors import GPS, IMU, SensorSuite

gps = GPS(noise_std=1.0, update_rate=1.0)
imu = IMU(accel_noise_std=0.1, update_rate=100.0)

sensor_suite = SensorSuite([gps, imu])
readings = sensor_suite.read_all(vehicle_state, timestamp)
```

## AI/ML Components

### Path Planning

#### `AStarPlanner`
A* path planning algorithm.

**Usage:**
```python
planner = AStarPlanner(environment, resolution=5.0)
path = planner.plan(start_position, goal_position)
```

#### `RRTPlanner`
Rapidly-exploring Random Tree planner.

**Usage:**
```python
planner = RRTPlanner(environment, max_iterations=1000, step_size=1.0)
path = planner.plan(start_position, goal_position)
```

### Navigation

#### `NavigationController`
Controller for following paths.

**Usage:**
```python
controller = NavigationController(lookahead_distance=5.0)
controller.set_path(path)
control_input = controller.compute_control(current_state)
```

### Perception

#### `PerceptionModule`
Process sensor data for object detection and state estimation.

## Control Systems

### `PIDController`
Proportional-Integral-Derivative controller.

### `FlightController`
Complete flight controller for aerial vehicles.

**Usage:**
```python
controller = FlightController(hover_thrust=15.0)
control = controller.compute_control(
    current_state,
    target_position,
    dt=0.1
)
```

## Projection and Forecasting

### `TrajectoryForecaster`
Forecast vehicle trajectories using historical data.

**Methods:**
- `constant_velocity`: Constant velocity model
- `linear`: Linear regression
- `polynomial`: Polynomial regression

**Usage:**
```python
forecaster = TrajectoryForecaster(method="linear", horizon=10.0)
forecaster.update_history(state)
forecast = forecaster.forecast(current_state, horizon=10.0)
```

## Communication

### `FleetManager`
Manage multiple vehicles.

**Usage:**
```python
fleet = FleetManager(vehicles, communication_channel)
fleet.assign_mission(vehicle_id, mission)
status = fleet.get_fleet_status()
```

## Environment

### `Environment`
Environment representation with obstacles.

**Usage:**
```python
obstacles = [Obstacle(position=np.array([50, 0, 5]), radius=10.0)]
environment = Environment(obstacles=obstacles)
collision = environment.check_collision(position)
```

## Visualization

### `TrajectoryVisualizer`
Visualize vehicle trajectories.

**Usage:**
```python
TrajectoryVisualizer.plot_2d_trajectory(trajectory)
TrajectoryVisualizer.plot_3d_trajectory(trajectory)
```

### `StateVisualizer`
Visualize vehicle states over time.

**Usage:**
```python
StateVisualizer.plot_state_history(states)
```

