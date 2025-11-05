"""
Core vehicle classes for UAV, UGV, and UUV.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class VehicleType(Enum):
    """Vehicle type enumeration."""
    AERIAL = "aerial"
    GROUND = "ground"
    UNDERWATER = "underwater"


@dataclass
class State:
    """Vehicle state representation."""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))  # Roll, Pitch, Yaw
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    timestamp: float = 0.0
    
    def copy(self) -> 'State':
        """Create a copy of the state."""
        return State(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            acceleration=self.acceleration.copy(),
            orientation=self.orientation.copy(),
            angular_velocity=self.angular_velocity.copy(),
            timestamp=self.timestamp
        )


@dataclass
class VehicleConfig:
    """Vehicle configuration parameters."""
    mass: float = 1.0
    max_speed: float = 10.0
    max_acceleration: float = 5.0
    max_angular_velocity: float = 1.0
    battery_capacity: float = 100.0
    battery_level: float = 100.0
    sensor_range: float = 100.0
    communication_range: float = 1000.0
    dimensions: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 0.5]))


class Vehicle(ABC):
    """Base class for all autonomous vehicles."""
    
    def __init__(
        self,
        vehicle_type: VehicleType,
        initial_state: Optional[State] = None,
        config: Optional[VehicleConfig] = None,
        vehicle_id: str = "vehicle_1"
    ):
        """
        Initialize vehicle.
        
        Args:
            vehicle_type: Type of vehicle (aerial, ground, underwater)
            initial_state: Initial state of the vehicle
            config: Vehicle configuration
            vehicle_id: Unique identifier for the vehicle
        """
        self.vehicle_type = vehicle_type
        self.vehicle_id = vehicle_id
        self.config = config or VehicleConfig()
        self.state = initial_state or State()
        self.history: List[State] = [self.state.copy()]
        self.commands: List[Dict] = []
        
    def get_state(self) -> State:
        """Get current state."""
        return self.state.copy()
    
    def update_state(self, new_state: State):
        """Update vehicle state."""
        self.state = new_state
        self.history.append(self.state.copy())
    
    @abstractmethod
    def dynamics(self, state: State, control_input: np.ndarray, dt: float) -> State:
        """
        Compute vehicle dynamics.
        
        Args:
            state: Current state
            control_input: Control input vector
            dt: Time step
            
        Returns:
            New state after dynamics update
        """
        pass
    
    @abstractmethod
    def get_constraints(self) -> Dict:
        """Get vehicle-specific constraints."""
        pass
    
    def get_position(self) -> np.ndarray:
        """Get current position."""
        return self.state.position.copy()
    
    def get_velocity(self) -> np.ndarray:
        """Get current velocity."""
        return self.state.velocity.copy()
    
    def get_distance_to(self, target_position: np.ndarray) -> float:
        """Calculate distance to target position."""
        return np.linalg.norm(self.state.position - target_position)
    
    def is_battery_low(self, threshold: float = 20.0) -> bool:
        """Check if battery level is below threshold."""
        return self.config.battery_level < threshold


class AerialVehicle(Vehicle):
    """Aerial vehicle (drone) implementation."""
    
    def __init__(
        self,
        initial_state: Optional[State] = None,
        config: Optional[VehicleConfig] = None,
        vehicle_id: str = "drone_1"
    ):
        """
        Initialize aerial vehicle.
        
        Args:
            initial_state: Initial state (position should be 3D with altitude)
            config: Vehicle configuration
            vehicle_id: Unique identifier
        """
        super().__init__(VehicleType.AERIAL, initial_state, config, vehicle_id)
        if initial_state is None:
            self.state.position[2] = 10.0  # Default altitude
        
        # Aerial-specific parameters
        self.max_altitude = 400.0  # meters (typical drone limit)
        self.min_altitude = 0.5
        self.hover_thrust = self.config.mass * 9.81  # Newton
        
    def dynamics(self, state: State, control_input: np.ndarray, dt: float) -> State:
        """
        Aerial vehicle dynamics with simplified quadcopter model.
        
        Control input: [thrust, roll_rate, pitch_rate, yaw_rate]
        """
        new_state = state.copy()
        
        # Extract control inputs
        thrust = control_input[0] if len(control_input) > 0 else self.hover_thrust
        roll_rate = control_input[1] if len(control_input) > 1 else 0.0
        pitch_rate = control_input[2] if len(control_input) > 2 else 0.0
        yaw_rate = control_input[3] if len(control_input) > 3 else 0.0
        
        # Orientation update
        new_state.orientation += np.array([roll_rate, pitch_rate, yaw_rate]) * dt
        
        # Gravity
        gravity = np.array([0, 0, -9.81])
        
        # Thrust direction (simplified - assumes orientation affects thrust direction)
        roll, pitch, yaw = new_state.orientation
        thrust_vector = np.array([
            np.sin(pitch) * np.cos(yaw),
            np.sin(pitch) * np.sin(yaw),
            np.cos(pitch)
        ]) * (thrust / self.config.mass)
        
        # Acceleration
        new_state.acceleration = thrust_vector + gravity
        
        # Update velocity
        new_state.velocity += new_state.acceleration * dt
        
        # Update position
        new_state.position += new_state.velocity * dt
        
        # Altitude constraints
        new_state.position[2] = np.clip(
            new_state.position[2],
            self.min_altitude,
            self.max_altitude
        )
        
        # Update timestamp
        new_state.timestamp += dt
        
        return new_state
    
    def get_constraints(self) -> Dict:
        """Get aerial vehicle constraints."""
        return {
            "min_altitude": self.min_altitude,
            "max_altitude": self.max_altitude,
            "max_speed": self.config.max_speed,
            "max_vertical_speed": 5.0,
            "max_angular_velocity": self.config.max_angular_velocity
        }


class GroundVehicle(Vehicle):
    """Ground vehicle (UGV) implementation."""
    
    def __init__(
        self,
        initial_state: Optional[State] = None,
        config: Optional[VehicleConfig] = None,
        vehicle_id: str = "ugv_1"
    ):
        """
        Initialize ground vehicle.
        
        Args:
            initial_state: Initial state (z position should be 0 or ground level)
            config: Vehicle configuration
            vehicle_id: Unique identifier
        """
        super().__init__(VehicleType.GROUND, initial_state, config, vehicle_id)
        if initial_state is None:
            self.state.position[2] = 0.0  # Ground level
        
        # Ground-specific parameters
        self.wheelbase = 2.0  # meters
        self.max_steering_angle = np.pi / 6  # 30 degrees
        
    def dynamics(self, state: State, control_input: np.ndarray, dt: float) -> State:
        """
        Ground vehicle dynamics with bicycle model.
        
        Control input: [speed, steering_angle]
        """
        new_state = state.copy()
        
        # Extract control inputs
        speed = control_input[0] if len(control_input) > 0 else 0.0
        steering_angle = control_input[1] if len(control_input) > 1 else 0.0
        
        # Limit steering angle
        steering_angle = np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)
        
        # Limit speed
        speed = np.clip(speed, -self.config.max_speed, self.config.max_speed)
        
        # Current orientation (yaw)
        yaw = state.orientation[2]
        
        # Update orientation based on steering
        if abs(speed) > 0.01:
            angular_velocity = (speed / self.wheelbase) * np.tan(steering_angle)
            new_state.orientation[2] += angular_velocity * dt
        
        # Update position
        new_state.velocity[0] = speed * np.cos(yaw)
        new_state.velocity[1] = speed * np.sin(yaw)
        new_state.velocity[2] = 0.0  # Ground vehicle stays on ground
        
        new_state.position += new_state.velocity * dt
        new_state.position[2] = 0.0  # Constrain to ground
        
        # Acceleration
        new_state.acceleration = (new_state.velocity - state.velocity) / dt if dt > 0 else np.zeros(3)
        
        # Angular velocity
        new_state.angular_velocity[2] = (speed / self.wheelbase) * np.tan(steering_angle) if abs(speed) > 0.01 else 0.0
        
        # Update timestamp
        new_state.timestamp += dt
        
        return new_state
    
    def get_constraints(self) -> Dict:
        """Get ground vehicle constraints."""
        return {
            "max_speed": self.config.max_speed,
            "max_steering_angle": self.max_steering_angle,
            "min_turning_radius": self.wheelbase / np.tan(self.max_steering_angle),
            "ground_level": 0.0
        }


class UnderwaterVehicle(Vehicle):
    """Underwater vehicle (UUV) implementation."""
    
    def __init__(
        self,
        initial_state: Optional[State] = None,
        config: Optional[VehicleConfig] = None,
        vehicle_id: str = "uuv_1"
    ):
        """
        Initialize underwater vehicle.
        
        Args:
            initial_state: Initial state (position should be 3D with depth)
            config: Vehicle configuration
            vehicle_id: Unique identifier
        """
        super().__init__(VehicleType.UNDERWATER, initial_state, config, vehicle_id)
        if initial_state is None:
            self.state.position[2] = -10.0  # Default depth
        
        # Underwater-specific parameters
        self.max_depth = 500.0  # meters
        self.buoyancy = 0.0  # Neutral buoyancy
        self.water_density = 1025.0  # kg/m^3 (seawater)
        self.drag_coefficient = 0.5
        
    def dynamics(self, state: State, control_input: np.ndarray, dt: float) -> State:
        """
        Underwater vehicle dynamics with hydrodynamic effects.
        
        Control input: [thrust_x, thrust_y, thrust_z, roll_rate, pitch_rate, yaw_rate]
        """
        new_state = state.copy()
        
        # Extract control inputs
        thrust = control_input[:3] if len(control_input) >= 3 else np.zeros(3)
        angular_rates = control_input[3:6] if len(control_input) >= 6 else np.zeros(3)
        
        # Update orientation
        new_state.orientation += angular_rates * dt
        
        # Buoyancy force
        volume = self.config.mass / self.water_density
        buoyancy_force = np.array([0, 0, self.water_density * 9.81 * volume])
        
        # Drag force (proportional to velocity squared)
        drag_force = -self.drag_coefficient * np.linalg.norm(new_state.velocity) * new_state.velocity
        
        # Total force
        total_force = thrust + buoyancy_force + drag_force
        
        # Acceleration
        new_state.acceleration = total_force / self.config.mass
        
        # Update velocity
        new_state.velocity += new_state.acceleration * dt
        
        # Update position
        new_state.position += new_state.velocity * dt
        
        # Depth constraints
        new_state.position[2] = np.clip(
            new_state.position[2],
            -self.max_depth,
            0.0  # Surface
        )
        
        # Update angular velocity
        new_state.angular_velocity = angular_rates
        
        # Update timestamp
        new_state.timestamp += dt
        
        return new_state
    
    def get_constraints(self) -> Dict:
        """Get underwater vehicle constraints."""
        return {
            "max_depth": self.max_depth,
            "max_speed": self.config.max_speed,
            "max_angular_velocity": self.config.max_angular_velocity,
            "surface_level": 0.0
        }

