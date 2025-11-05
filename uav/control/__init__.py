"""
Control systems for vehicle dynamics.
"""

import numpy as np
from typing import Optional
from ..core.vehicle import State


class PIDController:
    """Proportional-Integral-Derivative controller."""
    
    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.0,
        kd: float = 0.0,
        setpoint: float = 0.0
    ):
        """
        Initialize PID controller.
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            setpoint: Target value
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.integral = 0.0
        self.last_error = 0.0
        
    def update(self, current_value: float, dt: float) -> float:
        """
        Update controller and compute output.
        
        Args:
            current_value: Current measured value
            dt: Time step
            
        Returns:
            Control output
        """
        error = self.setpoint - current_value
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        self.integral += error * dt
        i_term = self.ki * self.integral
        
        # Derivative term
        derivative = (error - self.last_error) / dt if dt > 0 else 0.0
        d_term = self.kd * derivative
        
        self.last_error = error
        
        return p_term + i_term + d_term
    
    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.last_error = 0.0


class AttitudeController:
    """Attitude controller for aerial vehicles."""
    
    def __init__(
        self,
        kp_roll: float = 2.0,
        kp_pitch: float = 2.0,
        kp_yaw: float = 1.0
    ):
        """
        Initialize attitude controller.
        
        Args:
            kp_roll: Roll proportional gain
            kp_pitch: Pitch proportional gain
            kp_yaw: Yaw proportional gain
        """
        self.roll_controller = PIDController(kp=kp_roll)
        self.pitch_controller = PIDController(kp=kp_pitch)
        self.yaw_controller = PIDController(kp=kp_yaw)
        
    def compute_control(
        self,
        current_orientation: np.ndarray,
        desired_orientation: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """
        Compute attitude control inputs.
        
        Args:
            current_orientation: Current [roll, pitch, yaw]
            desired_orientation: Desired [roll, pitch, yaw]
            dt: Time step
            
        Returns:
            Control inputs [roll_rate, pitch_rate, yaw_rate]
        """
        self.roll_controller.setpoint = desired_orientation[0]
        self.pitch_controller.setpoint = desired_orientation[1]
        self.yaw_controller.setpoint = desired_orientation[2]
        
        roll_rate = self.roll_controller.update(current_orientation[0], dt)
        pitch_rate = self.pitch_controller.update(current_orientation[1], dt)
        yaw_rate = self.yaw_controller.update(current_orientation[2], dt)
        
        return np.array([roll_rate, pitch_rate, yaw_rate])


class PositionController:
    """Position controller for waypoint navigation."""
    
    def __init__(
        self,
        kp: float = 1.0,
        max_speed: float = 10.0
    ):
        """
        Initialize position controller.
        
        Args:
            kp: Proportional gain
            max_speed: Maximum allowed speed
        """
        self.kp = kp
        self.max_speed = max_speed
        
    def compute_control(
        self,
        current_position: np.ndarray,
        target_position: np.ndarray
    ) -> np.ndarray:
        """
        Compute position control (desired velocity).
        
        Args:
            current_position: Current position
            target_position: Target position
            
        Returns:
            Desired velocity vector
        """
        error = target_position - current_position
        distance = np.linalg.norm(error)
        
        if distance < 0.1:
            return np.zeros(3)
        
        direction = error / distance
        desired_speed = min(self.kp * distance, self.max_speed)
        
        return direction * desired_speed


class FlightController:
    """Complete flight controller for aerial vehicles."""
    
    def __init__(
        self,
        hover_thrust: float = 15.0,
        max_thrust: float = 30.0
    ):
        """
        Initialize flight controller.
        
        Args:
            hover_thrust: Thrust required for hover
            max_thrust: Maximum thrust
        """
        self.hover_thrust = hover_thrust
        self.max_thrust = max_thrust
        self.attitude_controller = AttitudeController()
        self.position_controller = PositionController()
        
    def compute_control(
        self,
        current_state: State,
        target_position: np.ndarray,
        target_altitude: Optional[float] = None,
        dt: float = 0.1
    ) -> np.ndarray:
        """
        Compute complete flight control inputs.
        
        Args:
            current_state: Current vehicle state
            target_position: Target position
            target_altitude: Optional target altitude (uses target_position[2] if None)
            dt: Time step
            
        Returns:
            Control inputs [thrust, roll_rate, pitch_rate, yaw_rate]
        """
        # Use target altitude if provided, otherwise use target_position z
        target_pos = target_position.copy()
        if target_altitude is not None:
            target_pos[2] = target_altitude
        
        # Compute desired velocity
        desired_velocity = self.position_controller.compute_control(
            current_state.position,
            target_pos
        )
        
        # Compute desired orientation from desired velocity
        # For aerial vehicle, pitch forward to move forward
        speed = np.linalg.norm(desired_velocity)
        if speed > 0.1:
            yaw = np.arctan2(desired_velocity[1], desired_velocity[0])
            pitch = -np.arctan2(desired_velocity[2], np.linalg.norm(desired_velocity[:2]))
        else:
            yaw = current_state.orientation[2]
            pitch = 0.0
        
        desired_orientation = np.array([0.0, pitch, yaw])
        
        # Compute attitude control
        angular_rates = self.attitude_controller.compute_control(
            current_state.orientation,
            desired_orientation,
            dt
        )
        
        # Compute thrust (simplified)
        altitude_error = target_pos[2] - current_state.position[2]
        thrust = self.hover_thrust + altitude_error * 2.0
        thrust = np.clip(thrust, 0.0, self.max_thrust)
        
        return np.array([thrust, angular_rates[0], angular_rates[1], angular_rates[2]])

