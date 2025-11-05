"""
Communication protocols for multi-vehicle systems.
"""

import numpy as np
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from ..core.vehicle import State


@dataclass
class Message:
    """Message structure for vehicle communication."""
    sender_id: str
    receiver_id: Optional[str]  # None for broadcast
    message_type: str
    data: Dict
    timestamp: float
    
    def to_dict(self) -> Dict:
        """Convert message to dictionary."""
        return {
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type,
            "data": self.data,
            "timestamp": self.timestamp
        }


class CommunicationChannel:
    """Communication channel between vehicles."""
    
    def __init__(
        self,
        range: float = 1000.0,
        bandwidth: float = 1.0,  # Messages per second
        packet_loss_rate: float = 0.0
    ):
        """
        Initialize communication channel.
        
        Args:
            range: Communication range in meters
            bandwidth: Maximum messages per second
            packet_loss_rate: Probability of packet loss (0-1)
        """
        self.range = range
        self.bandwidth = bandwidth
        self.packet_loss_rate = packet_loss_rate
        self.message_queue: List[Message] = []
        self.sent_messages: List[Message] = []
        self.received_messages: Dict[str, List[Message]] = {}
        
    def send_message(
        self,
        sender_position: np.ndarray,
        receiver_position: np.ndarray,
        message: Message
    ) -> bool:
        """
        Send message from sender to receiver.
        
        Args:
            sender_position: Position of sender
            receiver_position: Position of receiver
            message: Message to send
            
        Returns:
            True if message sent successfully
        """
        # Check range
        distance = np.linalg.norm(receiver_position - sender_position)
        if distance > self.range:
            return False
        
        # Check packet loss
        if np.random.random() < self.packet_loss_rate:
            return False
        
        # Check bandwidth
        if len(self.message_queue) >= self.bandwidth:
            return False
        
        self.message_queue.append(message)
        self.sent_messages.append(message)
        
        # Deliver message
        receiver_id = message.receiver_id or "broadcast"
        if receiver_id not in self.received_messages:
            self.received_messages[receiver_id] = []
        self.received_messages[receiver_id].append(message)
        
        return True
    
    def broadcast_message(
        self,
        sender_position: np.ndarray,
        vehicle_positions: Dict[str, np.ndarray],
        message: Message
    ) -> List[str]:
        """
        Broadcast message to all vehicles in range.
        
        Args:
            sender_position: Position of sender
            vehicle_positions: Dictionary of vehicle_id -> position
            message: Message to broadcast
            
        Returns:
            List of vehicle IDs that received the message
        """
        received_by = []
        
        for vehicle_id, receiver_position in vehicle_positions.items():
            if vehicle_id == message.sender_id:
                continue
            
            if self.send_message(sender_position, receiver_position, message):
                received_by.append(vehicle_id)
        
        return received_by
    
    def get_messages(self, vehicle_id: str) -> List[Message]:
        """Get messages for a vehicle."""
        messages = self.received_messages.get(vehicle_id, [])
        broadcast_messages = self.received_messages.get("broadcast", [])
        return messages + broadcast_messages


class FleetManager:
    """Fleet management for multiple vehicles."""
    
    def __init__(
        self,
        vehicles: List,
        communication_channel: Optional[CommunicationChannel] = None
    ):
        """
        Initialize fleet manager.
        
        Args:
            vehicles: List of vehicles
            communication_channel: Communication channel
        """
        self.vehicles = {v.vehicle_id: v for v in vehicles}
        self.communication_channel = communication_channel or CommunicationChannel()
        self.mission_assignments: Dict[str, Dict] = {}
        
    def assign_mission(
        self,
        vehicle_id: str,
        mission: Dict
    ):
        """
        Assign mission to vehicle.
        
        Args:
            vehicle_id: Vehicle ID
            mission: Mission dictionary with waypoints, objectives, etc.
        """
        self.mission_assignments[vehicle_id] = mission
        
        # Send mission assignment via communication
        if vehicle_id in self.vehicles:
            vehicle = self.vehicles[vehicle_id]
            message = Message(
                sender_id="fleet_manager",
                receiver_id=vehicle_id,
                message_type="mission_assignment",
                data=mission,
                timestamp=0.0
            )
            
            # In real implementation, this would use actual positions
            sender_pos = np.zeros(3)
            receiver_pos = vehicle.get_position()
            self.communication_channel.send_message(sender_pos, receiver_pos, message)
    
    def get_fleet_status(self) -> Dict:
        """Get status of all vehicles in fleet."""
        status = {}
        
        for vehicle_id, vehicle in self.vehicles.items():
            status[vehicle_id] = {
                "position": vehicle.get_position().tolist(),
                "velocity": vehicle.get_velocity().tolist(),
                "battery_level": vehicle.config.battery_level,
                "mission": self.mission_assignments.get(vehicle_id, {}),
                "state": "active" if vehicle.config.battery_level > 0 else "inactive"
            }
        
        return status
    
    def coordinate_flight(
        self,
        waypoints: List[np.ndarray],
        num_vehicles: Optional[int] = None
    ):
        """
        Coordinate multiple vehicles to reach waypoints.
        
        Args:
            waypoints: List of waypoints to visit
            num_vehicles: Number of vehicles to use (uses all if None)
        """
        vehicle_ids = list(self.vehicles.keys())
        if num_vehicles:
            vehicle_ids = vehicle_ids[:num_vehicles]
        
        # Simple assignment: divide waypoints among vehicles
        waypoints_per_vehicle = len(waypoints) // len(vehicle_ids)
        
        for i, vehicle_id in enumerate(vehicle_ids):
            start_idx = i * waypoints_per_vehicle
            end_idx = start_idx + waypoints_per_vehicle if i < len(vehicle_ids) - 1 else len(waypoints)
            
            mission = {
                "waypoints": waypoints[start_idx:end_idx],
                "objective": "patrol"
            }
            
            self.assign_mission(vehicle_id, mission)
    
    def update(self, timestamp: float):
        """
        Update fleet manager (process communications, check status).
        
        Args:
            timestamp: Current simulation time
        """
        # Check for low battery vehicles
        for vehicle_id, vehicle in self.vehicles.items():
            if vehicle.is_battery_low():
                # Send return-to-base message
                message = Message(
                    sender_id="fleet_manager",
                    receiver_id=vehicle_id,
                    message_type="return_to_base",
                    data={"reason": "low_battery"},
                    timestamp=timestamp
                )
                
                sender_pos = np.zeros(3)
                receiver_pos = vehicle.get_position()
                self.communication_channel.send_message(sender_pos, receiver_pos, message)

