"""
Sensor modules for UAV systems.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from ..core.vehicle import State


@dataclass
class SensorReading:
    """Sensor reading data structure."""
    timestamp: float
    data: np.ndarray
    sensor_type: str
    noise_level: float = 0.0


class Sensor:
    """Base sensor class."""
    
    def __init__(
        self,
        sensor_type: str,
        noise_std: float = 0.0,
        update_rate: float = 10.0
    ):
        """
        Initialize sensor.
        
        Args:
            sensor_type: Type of sensor
            noise_std: Standard deviation of sensor noise
            update_rate: Update rate in Hz
        """
        self.sensor_type = sensor_type
        self.noise_std = noise_std
        self.update_rate = update_rate
        self.update_period = 1.0 / update_rate
        self.last_update_time = 0.0
        self.readings: List[SensorReading] = []
        
    def add_noise(self, measurement: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to measurement."""
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, measurement.shape)
            return measurement + noise
        return measurement
    
    def read(self, state: State, timestamp: float) -> Optional[SensorReading]:
        """
        Read sensor data.
        
        Args:
            state: Current vehicle state
            timestamp: Current time
            
        Returns:
            Sensor reading or None if not ready
        """
        if timestamp - self.last_update_time < self.update_period:
            return None
        
        measurement = self._measure(state)
        noisy_measurement = self.add_noise(measurement)
        
        reading = SensorReading(
            timestamp=timestamp,
            data=noisy_measurement,
            sensor_type=self.sensor_type,
            noise_level=self.noise_std
        )
        
        self.readings.append(reading)
        self.last_update_time = timestamp
        
        return reading
    
    def _measure(self, state: State) -> np.ndarray:
        """Internal measurement method (to be overridden)."""
        raise NotImplementedError


class GPS(Sensor):
    """GPS sensor."""
    
    def __init__(self, noise_std: float = 1.0, update_rate: float = 1.0):
        """
        Initialize GPS sensor.
        
        Args:
            noise_std: Position noise standard deviation in meters
            update_rate: Update rate in Hz (typically 1 Hz for GPS)
        """
        super().__init__("GPS", noise_std, update_rate)
        
    def _measure(self, state: State) -> np.ndarray:
        """Measure position."""
        return state.position.copy()


class IMU(Sensor):
    """Inertial Measurement Unit."""
    
    def __init__(
        self,
        accel_noise_std: float = 0.1,
        gyro_noise_std: float = 0.01,
        update_rate: float = 100.0
    ):
        """
        Initialize IMU sensor.
        
        Args:
            accel_noise_std: Accelerometer noise standard deviation
            gyro_noise_std: Gyroscope noise standard deviation
            update_rate: Update rate in Hz (typically 100+ Hz)
        """
        super().__init__("IMU", 0.0, update_rate)
        self.accel_noise_std = accel_noise_std
        self.gyro_noise_std = gyro_noise_std
        
    def _measure(self, state: State) -> np.ndarray:
        """Measure acceleration and angular velocity."""
        # Combine accelerometer and gyroscope readings
        accel_with_noise = self.add_noise(state.acceleration.copy())
        gyro_with_noise = np.random.normal(
            state.angular_velocity,
            self.gyro_noise_std,
            state.angular_velocity.shape
        )
        return np.concatenate([accel_with_noise, gyro_with_noise])
    
    def add_noise(self, measurement: np.ndarray) -> np.ndarray:
        """Add noise to accelerometer measurement."""
        if self.accel_noise_std > 0:
            noise = np.random.normal(0, self.accel_noise_std, measurement.shape)
            return measurement + noise
        return measurement


class LiDAR(Sensor):
    """LiDAR sensor."""
    
    def __init__(
        self,
        max_range: float = 100.0,
        fov: float = 360.0,
        num_beams: int = 360,
        noise_std: float = 0.05,
        update_rate: float = 10.0
    ):
        """
        Initialize LiDAR sensor.
        
        Args:
            max_range: Maximum detection range in meters
            fov: Field of view in degrees
            num_beams: Number of LiDAR beams
            noise_std: Range measurement noise standard deviation
            update_rate: Update rate in Hz
        """
        super().__init__("LiDAR", noise_std, update_rate)
        self.max_range = max_range
        self.fov = np.deg2rad(fov)
        self.num_beams = num_beams
        self.beam_angles = np.linspace(0, self.fov, num_beams)
        
    def _measure(self, state: State) -> np.ndarray:
        """
        Measure distances (simplified - returns max_range for all beams).
        In real implementation, this would raycast against environment.
        """
        # Simplified: return max_range for all beams
        # Real implementation would raycast against obstacles
        ranges = np.full(self.num_beams, self.max_range)
        return ranges


class Camera(Sensor):
    """Camera sensor."""
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (640, 480),
        fov: float = 60.0,
        update_rate: float = 30.0
    ):
        """
        Initialize camera sensor.
        
        Args:
            resolution: Image resolution (width, height)
            fov: Field of view in degrees
            update_rate: Frame rate in Hz
        """
        super().__init__("Camera", 0.0, update_rate)
        self.resolution = resolution
        self.fov = np.deg2rad(fov)
        
    def _measure(self, state: State) -> np.ndarray:
        """
        Capture image (simplified - returns dummy array).
        Real implementation would render scene from vehicle's perspective.
        """
        # Simplified: return dummy image array
        return np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)


class Sonar(Sensor):
    """Sonar sensor for underwater vehicles."""
    
    def __init__(
        self,
        max_range: float = 200.0,
        fov: float = 60.0,
        noise_std: float = 0.1,
        update_rate: float = 1.0
    ):
        """
        Initialize sonar sensor.
        
        Args:
            max_range: Maximum detection range in meters
            fov: Field of view in degrees
            noise_std: Range measurement noise standard deviation
            update_rate: Update rate in Hz
        """
        super().__init__("Sonar", noise_std, update_rate)
        self.max_range = max_range
        self.fov = np.deg2rad(fov)
        
    def _measure(self, state: State) -> np.ndarray:
        """
        Measure distance (simplified).
        Real implementation would account for underwater acoustics.
        """
        # Simplified: return max_range
        return np.array([self.max_range])


class SensorSuite:
    """Collection of sensors on a vehicle."""
    
    def __init__(self, sensors: Optional[List[Sensor]] = None):
        """
        Initialize sensor suite.
        
        Args:
            sensors: List of sensors
        """
        self.sensors = sensors or []
        self.sensor_dict = {s.sensor_type: s for s in self.sensors}
        
    def add_sensor(self, sensor: Sensor):
        """Add sensor to suite."""
        self.sensors.append(sensor)
        self.sensor_dict[sensor.sensor_type] = sensor
        
    def read_all(self, state: State, timestamp: float) -> Dict[str, SensorReading]:
        """
        Read all sensors.
        
        Args:
            state: Current vehicle state
            timestamp: Current time
            
        Returns:
            Dictionary of sensor readings
        """
        readings = {}
        for sensor in self.sensors:
            reading = sensor.read(state, timestamp)
            if reading is not None:
                readings[sensor.sensor_type] = reading
        return readings
    
    def get_sensor(self, sensor_type: str) -> Optional[Sensor]:
        """Get sensor by type."""
        return self.sensor_dict.get(sensor_type)

__all__ = [
    "Sensor",
    "GPS",
    "IMU",
    "LiDAR",
    "Camera",
    "Sonar",
    "SensorSuite",
    "SensorReading",
]
