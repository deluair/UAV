"""
Simulation engine for UAV systems.
"""

import numpy as np
from typing import List, Dict, Optional, Callable
from tqdm import tqdm
from ..core.vehicle import Vehicle, State


class Simulator:
    """Simulation engine for autonomous vehicles."""
    
    def __init__(
        self,
        vehicle: Vehicle,
        control_policy: Optional[Callable] = None,
        environment: Optional['Environment'] = None
    ):
        """
        Initialize simulator.
        
        Args:
            vehicle: Vehicle to simulate
            control_policy: Optional control policy function(state, t) -> control_input
            environment: Optional environment for obstacles and interactions
        """
        self.vehicle = vehicle
        self.control_policy = control_policy
        self.environment = environment
        self.simulation_history: List[Dict] = []
        
    def run(
        self,
        duration: float,
        dt: float = 0.1,
        verbose: bool = True,
        progress_bar: bool = True
    ) -> List[Dict]:
        """
        Run simulation for specified duration.
        
        Args:
            duration: Simulation duration in seconds
            dt: Time step in seconds
            verbose: Print simulation info
            progress_bar: Show progress bar
            
        Returns:
            List of simulation states at each time step
        """
        num_steps = int(duration / dt)
        self.simulation_history = []
        
        iterator = tqdm(range(num_steps), desc="Simulating") if progress_bar else range(num_steps)
        
        for step in iterator:
            t = step * dt
            
            # Get current state
            current_state = self.vehicle.get_state()
            
            # Compute control input
            if self.control_policy:
                control_input = self.control_policy(current_state, t)
            else:
                control_input = np.zeros(4)  # Default: no control
            
            # Update vehicle dynamics
            new_state = self.vehicle.dynamics(current_state, control_input, dt)
            
            # Check collisions with environment
            if self.environment:
                collision = self.environment.check_collision(new_state.position)
                if collision:
                    if verbose:
                        print(f"Collision detected at t={t:.2f}s, position={new_state.position}")
                    break
            
            # Update vehicle state
            self.vehicle.update_state(new_state)
            
            # Record simulation step
            step_data = {
                "timestamp": t,
                "state": new_state.copy(),
                "control_input": control_input.copy() if isinstance(control_input, np.ndarray) else control_input,
                "vehicle_id": self.vehicle.vehicle_id
            }
            self.simulation_history.append(step_data)
        
        if verbose:
            print(f"\nSimulation completed: {len(self.simulation_history)} steps, "
                  f"final position: {self.vehicle.get_position()}")
        
        return self.simulation_history
    
    def get_trajectory(self) -> np.ndarray:
        """Get vehicle trajectory as array of positions."""
        return np.array([step["state"].position for step in self.simulation_history])
    
    def get_velocities(self) -> np.ndarray:
        """Get vehicle velocities over time."""
        return np.array([step["state"].velocity for step in self.simulation_history])
    
    def get_control_inputs(self) -> np.ndarray:
        """Get control inputs over time."""
        return np.array([step["control_input"] for step in self.simulation_history])


class MultiVehicleSimulator:
    """Simulator for multiple vehicles."""
    
    def __init__(
        self,
        vehicles: List[Vehicle],
        control_policies: Optional[Dict[str, Callable]] = None,
        environment: Optional['Environment'] = None
    ):
        """
        Initialize multi-vehicle simulator.
        
        Args:
            vehicles: List of vehicles to simulate
            control_policies: Dictionary mapping vehicle_id to control policy
            environment: Optional environment
        """
        self.vehicles = {v.vehicle_id: v for v in vehicles}
        self.control_policies = control_policies or {}
        self.environment = environment
        self.simulation_history: List[Dict] = []
        
    def run(
        self,
        duration: float,
        dt: float = 0.1,
        verbose: bool = True,
        progress_bar: bool = True
    ) -> List[Dict]:
        """
        Run multi-vehicle simulation.
        
        Args:
            duration: Simulation duration
            dt: Time step
            verbose: Print info
            progress_bar: Show progress bar
            
        Returns:
            List of simulation states for all vehicles
        """
        num_steps = int(duration / dt)
        self.simulation_history = []
        
        iterator = tqdm(range(num_steps), desc="Multi-Vehicle Simulation") if progress_bar else range(num_steps)
        
        for step in iterator:
            t = step * dt
            step_data = {"timestamp": t, "vehicles": {}}
            
            # Update each vehicle
            for vehicle_id, vehicle in self.vehicles.items():
                current_state = vehicle.get_state()
                
                # Get control policy for this vehicle
                control_policy = self.control_policies.get(vehicle_id)
                if control_policy:
                    control_input = control_policy(current_state, t, self.vehicles)
                else:
                    control_input = np.zeros(4)
                
                # Update dynamics
                new_state = vehicle.dynamics(current_state, control_input, dt)
                
                # Check collisions
                collision = False
                if self.environment:
                    collision = self.environment.check_collision(new_state.position)
                
                # Check inter-vehicle collisions
                for other_id, other_vehicle in self.vehicles.items():
                    if other_id != vehicle_id:
                        distance = np.linalg.norm(
                            new_state.position - other_vehicle.get_position()
                        )
                        if distance < 2.0:  # Minimum safe distance
                            collision = True
                            break
                
                if not collision:
                    vehicle.update_state(new_state)
                
                step_data["vehicles"][vehicle_id] = {
                    "state": new_state.copy(),
                    "control_input": control_input.copy() if isinstance(control_input, np.ndarray) else control_input,
                    "collision": collision
                }
            
            self.simulation_history.append(step_data)
        
        if verbose:
            print(f"\nMulti-vehicle simulation completed: {len(self.simulation_history)} steps")
        
        return self.simulation_history
    
    def get_trajectories(self) -> Dict[str, np.ndarray]:
        """Get trajectories for all vehicles."""
        trajectories = {}
        for vehicle_id in self.vehicles.keys():
            trajectory = []
            for step in self.simulation_history:
                if vehicle_id in step["vehicles"]:
                    trajectory.append(step["vehicles"][vehicle_id]["state"].position)
            trajectories[vehicle_id] = np.array(trajectory)
        return trajectories

