# Projection Models Documentation

## Overview

Trajectory projection and forecasting models predict future vehicle positions based on current state and historical data.

## Trajectory Forecasting

### Constant Velocity Model

Simplest forecasting model assuming constant velocity.

```python
from uav.projection import TrajectoryForecaster

forecaster = TrajectoryForecaster(method="constant_velocity", horizon=10.0)
forecast = forecaster.forecast(current_state, horizon=10.0)
```

**Assumptions:**
- Vehicle maintains current velocity
- No acceleration
- Suitable for short-term predictions

### Linear Regression Model

Uses linear regression on historical data.

```python
forecaster = TrajectoryForecaster(method="linear", horizon=10.0)

# Update with history
for state in state_history:
    forecaster.update_history(state)

# Forecast
forecast = forecaster.forecast(current_state)
```

**Features:**
- Learns from historical trajectory
- Accounts for trends
- Better for medium-term predictions

### Polynomial Regression Model

Uses polynomial regression for non-linear trajectories.

```python
forecaster = TrajectoryForecaster(method="polynomial", horizon=10.0)
forecast = forecaster.forecast(current_state)
```

**Features:**
- Captures non-linear motion patterns
- Requires more historical data
- Can overfit with limited data

## Trajectory Prediction

### Physics-Based Prediction

Predict trajectories using physics models.

```python
from uav.projection import TrajectoryPredictor

predictor = TrajectoryPredictor(dt=0.1)

# Constant acceleration model
predicted = predictor.predict_constant_acceleration(
    initial_state,
    duration=10.0,
    acceleration=np.array([0, 0, -9.81])  # Gravity
)
```

### Control-Based Prediction

Predict trajectory given control sequence.

```python
# Define control sequence
control_sequence = np.array([
    [15.0, 0.0, 0.1, 0.0],  # Control at t=0
    [15.0, 0.0, 0.1, 0.0],  # Control at t=0.1
    # ... more controls ...
])

# Predict trajectory
predicted = predictor.predict_with_control(
    initial_state,
    control_sequence,
    dynamics_function=vehicle.dynamics
)
```

## Advanced Forecasting

### Ensemble Forecasting

Combine multiple forecasting methods for better accuracy.

```python
def ensemble_forecast(state, history):
    # Get forecasts from multiple methods
    cv_forecast = TrajectoryForecaster(method="constant_velocity").forecast(state)
    linear_forecast = TrajectoryForecaster(method="linear").forecast(state)
    poly_forecast = TrajectoryForecaster(method="polynomial").forecast(state)
    
    # Weighted average
    weights = [0.3, 0.4, 0.3]
    ensemble = (weights[0] * cv_forecast + 
                weights[1] * linear_forecast + 
                weights[2] * poly_forecast)
    
    return ensemble
```

## Evaluation Metrics

### Forecast Accuracy

```python
def evaluate_forecast(predicted, actual):
    # Mean Squared Error
    mse = np.mean((predicted - actual)**2)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(predicted - actual))
    
    # Maximum Error
    max_error = np.max(np.linalg.norm(predicted - actual, axis=1))
    
    return {
        'mse': mse,
        'mae': mae,
        'max_error': max_error
    }
```

## Use Cases

### Collision Avoidance

```python
def predict_collision(vehicle1, vehicle2, horizon=5.0):
    forecaster1 = TrajectoryForecaster()
    forecaster2 = TrajectoryForecaster()
    
    forecast1 = forecaster1.forecast(vehicle1.get_state(), horizon)
    forecast2 = forecaster2.forecast(vehicle2.get_state(), horizon)
    
    # Check minimum distance
    distances = [np.linalg.norm(p1 - p2) 
                 for p1, p2 in zip(forecast1, forecast2)]
    
    min_distance = np.min(distances)
    collision_risk = min_distance < 5.0  # 5m threshold
    
    return collision_risk, min_distance
```

## Best Practices

1. **Model Selection**: Choose model based on prediction horizon
2. **History Length**: Use sufficient historical data for learning models
3. **Uncertainty Quantification**: Include uncertainty bounds in predictions
4. **Model Updating**: Update models as new data arrives
5. **Validation**: Validate predictions against actual trajectories

## Limitations

- **Short-term**: Constant velocity works well
- **Medium-term**: Linear/polynomial regression better
- **Long-term**: Physics-based models more reliable
- **Uncertainty**: All models have uncertainty that increases with horizon

