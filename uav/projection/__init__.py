"""
Trajectory projection and forecasting models.
"""

import numpy as np
from typing import List, Optional, Dict
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from ..core.vehicle import State


class TrajectoryForecaster:
    """Forecast vehicle trajectories using historical data."""
    
    def __init__(
        self,
        method: str = "linear",
        horizon: float = 10.0,
        dt: float = 0.1
    ):
        """
        Initialize trajectory forecaster.
        
        Args:
            method: Forecasting method ("linear", "polynomial", "constant_velocity")
            horizon: Forecast horizon in seconds
            dt: Time step for forecast
        """
        self.method = method
        self.horizon = horizon
        self.dt = dt
        self.history: List[State] = []
        
    def update_history(self, state: State):
        """Update historical states."""
        self.history.append(state.copy())
        # Keep only recent history (last 100 states)
        if len(self.history) > 100:
            self.history.pop(0)
    
    def forecast(
        self,
        current_state: Optional[State] = None,
        horizon: Optional[float] = None
    ) -> np.ndarray:
        """
        Forecast future trajectory.
        
        Args:
            current_state: Current state (uses last state in history if None)
            horizon: Forecast horizon (uses self.horizon if None)
            
        Returns:
            Array of forecasted positions
        """
        if current_state is None:
            if not self.history:
                raise ValueError("No state history available")
            current_state = self.history[-1]
        
        horizon = horizon or self.horizon
        num_steps = int(horizon / self.dt)
        
        if self.method == "constant_velocity":
            return self._constant_velocity_forecast(current_state, num_steps)
        elif self.method == "linear":
            return self._linear_forecast(current_state, num_steps)
        elif self.method == "polynomial":
            return self._polynomial_forecast(current_state, num_steps)
        else:
            return self._constant_velocity_forecast(current_state, num_steps)
    
    def _constant_velocity_forecast(
        self,
        state: State,
        num_steps: int
    ) -> np.ndarray:
        """Forecast using constant velocity model."""
        trajectory = []
        position = state.position.copy()
        velocity = state.velocity.copy()
        
        for _ in range(num_steps):
            position = position + velocity * self.dt
            trajectory.append(position.copy())
        
        return np.array(trajectory)
    
    def _linear_forecast(
        self,
        state: State,
        num_steps: int
    ) -> np.ndarray:
        """Forecast using linear regression on history."""
        if len(self.history) < 3:
            return self._constant_velocity_forecast(state, num_steps)
        
        # Extract position history
        positions = np.array([s.position for s in self.history[-20:]])
        times = np.array([s.timestamp for s in self.history[-20:]])
        
        trajectory = []
        current_time = state.timestamp
        
        for dim in range(3):
            # Fit linear model
            X = times.reshape(-1, 1)
            y = positions[:, dim]
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Forecast
            forecast_times = current_time + np.arange(1, num_steps + 1) * self.dt
            forecast_positions = model.predict(forecast_times.reshape(-1, 1))
            
            if dim == 0:
                trajectory = forecast_positions.reshape(-1, 1)
            else:
                trajectory = np.hstack([trajectory, forecast_positions.reshape(-1, 1)])
        
        return trajectory
    
    def _polynomial_forecast(
        self,
        state: State,
        num_steps: int
    ) -> np.ndarray:
        """Forecast using polynomial regression."""
        if len(self.history) < 5:
            return self._constant_velocity_forecast(state, num_steps)
        
        positions = np.array([s.position for s in self.history[-20:]])
        times = np.array([s.timestamp for s in self.history[-20:]])
        
        trajectory = []
        current_time = state.timestamp
        
        for dim in range(3):
            X = times.reshape(-1, 1)
            y = positions[:, dim]
            
            # Use polynomial features
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)
            
            model = LinearRegression()
            model.fit(X_poly, y)
            
            # Forecast
            forecast_times = current_time + np.arange(1, num_steps + 1) * self.dt
            forecast_times_poly = poly.transform(forecast_times.reshape(-1, 1))
            forecast_positions = model.predict(forecast_times_poly)
            
            if dim == 0:
                trajectory = forecast_positions.reshape(-1, 1)
            else:
                trajectory = np.hstack([trajectory, forecast_positions.reshape(-1, 1)])
        
        return trajectory


class TrajectoryPredictor:
    """Predict trajectories using physics-based models."""
    
    def __init__(self, dt: float = 0.1):
        """
        Initialize trajectory predictor.
        
        Args:
            dt: Time step for prediction
        """
        self.dt = dt
        
    def predict_constant_acceleration(
        self,
        initial_state: State,
        duration: float,
        acceleration: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict trajectory with constant acceleration model.
        
        Args:
            initial_state: Initial state
            duration: Prediction duration
            acceleration: Constant acceleration (uses state acceleration if None)
            
        Returns:
            Array of predicted positions
        """
        if acceleration is None:
            acceleration = initial_state.acceleration.copy()
        
        num_steps = int(duration / self.dt)
        trajectory = []
        
        position = initial_state.position.copy()
        velocity = initial_state.velocity.copy()
        
        for _ in range(num_steps):
            velocity = velocity + acceleration * self.dt
            position = position + velocity * self.dt
            trajectory.append(position.copy())
        
        return np.array(trajectory)
    
    def predict_with_control(
        self,
        initial_state: State,
        control_sequence: np.ndarray,
        dynamics_function
    ) -> np.ndarray:
        """
        Predict trajectory given control sequence.
        
        Args:
            initial_state: Initial state
            control_sequence: Sequence of control inputs
            dynamics_function: Function(state, control, dt) -> new_state
            
        Returns:
            Array of predicted positions
        """
        trajectory = []
        state = initial_state.copy()
        
        for control in control_sequence:
            state = dynamics_function(state, control, self.dt)
            trajectory.append(state.position.copy())
        
        return np.array(trajectory)

