# Unmanned Autonomous Vehicle (UAV) System

A comprehensive Python framework for simulating, controlling, and projecting the behavior of unmanned autonomous vehicles across multiple domains: aerial (drones), ground-based (UGVs), and underwater (UUVs).

## Features

- **Multi-Domain Simulation**: Support for aerial, ground, and underwater vehicles
- **Sensor Integration**: GPS, IMU, LiDAR, cameras, sonar, and more
- **AI/ML Navigation**: Perception, path planning, and autonomous decision-making
- **Trajectory Projection**: Forecasting and prediction models for vehicle paths
- **Control Systems**: Flight control, motion planning, and dynamics simulation
- **Fleet Management**: Multi-vehicle coordination and communication
- **Visualization**: Real-time and post-processing visualization tools

## Project Structure

```
UAV/
├── uav/                      # Main package
│   ├── core/                 # Core vehicle classes
│   ├── simulation/           # Simulation engines
│   ├── sensors/              # Sensor modules
│   ├── ai/                   # AI/ML components
│   ├── control/              # Control systems
│   ├── communication/        # Communication protocols
│   ├── projection/           # Forecasting and prediction
│   └── utils/                # Utility functions
├── examples/                 # Example scripts and demos
├── tests/                    # Unit tests
├── docs/                     # Documentation
├── data/                     # Data files and results
└── requirements.txt          # Python dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Simulation

```python
from uav.core.vehicle import AerialVehicle
from uav.simulation.simulator import Simulator

# Create a drone
drone = AerialVehicle(
    position=[0, 0, 100],
    velocity=[10, 0, 0],
    mass=1.5,
    max_speed=30
)

# Run simulation
simulator = Simulator(drone)
simulator.run(duration=60, dt=0.1)
```

### Path Planning

```python
from uav.ai.path_planner import AStarPlanner
from uav.core.environment import Environment

env = Environment()
planner = AStarPlanner(env)
path = planner.plan(start=[0, 0, 0], goal=[100, 100, 50])
```

### Trajectory Projection

```python
from uav.projection.forecaster import TrajectoryForecaster

forecaster = TrajectoryForecaster()
predicted_path = forecaster.forecast(
    current_state=drone.get_state(),
    horizon=30  # seconds
)
```

## Documentation

See `docs/` directory for detailed documentation:
- `API_REFERENCE.md` - Complete API documentation
- `SIMULATION_GUIDE.md` - Simulation usage guide
- `AI_NAVIGATION.md` - AI/ML navigation guide
- `PROJECTION_MODELS.md` - Forecasting models documentation

## Interactive Dashboard

🚀 **Launch the interactive dashboard:**

```bash
streamlit run dashboard.py
```

Or on Windows:
```bash
run_dashboard.bat
```

The dashboard provides:
- Real-time vehicle simulation and visualization
- Interactive path planning with obstacle avoidance
- Multi-vehicle fleet management
- Trajectory forecasting and analytics
- 3D visualization of vehicle trajectories

## Examples

Check `examples/` directory for:
- `basic_simulation.py` - Basic vehicle simulation
- `multi_vehicle.py` - Fleet coordination
- `path_planning_demo.py` - Path planning examples
- `sensor_integration.py` - Sensor data processing
- `forecasting_demo.py` - Trajectory projection

## License

MIT License

## Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.

## Authors

UAV Research Team

