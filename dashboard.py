"""
Interactive Dashboard for UAV System
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from typing import List, Dict

from uav.core.vehicle import AerialVehicle, GroundVehicle, UnderwaterVehicle, State, VehicleConfig
from uav.simulation.simulator import Simulator, MultiVehicleSimulator
from uav.core.environment import Environment, Obstacle
from uav.ai import AStarPlanner, RRTPlanner, NavigationController
from uav.sensors import GPS, IMU, SensorSuite
from uav.projection import TrajectoryForecaster
from uav.communication import FleetManager


st.set_page_config(page_title="UAV System Dashboard", layout="wide", initial_sidebar_state="expanded")

# Initialize session state
if 'simulation_history' not in st.session_state:
    st.session_state.simulation_history = []
if 'vehicles' not in st.session_state:
    st.session_state.vehicles = []
if 'environment' not in st.session_state:
    st.session_state.environment = Environment()


def main():
    st.title("🚁 UAV System Interactive Dashboard")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("🎛️ Control Panel")
    
    page = st.sidebar.selectbox(
        "Select Page",
        ["🏠 Overview", "🚁 Single Vehicle", "👥 Multi-Vehicle", "🗺️ Path Planning", "📊 Analytics", "⚙️ Settings"]
    )
    
    if page == "🏠 Overview":
        show_overview()
    elif page == "🚁 Single Vehicle":
        show_single_vehicle()
    elif page == "👥 Multi-Vehicle":
        show_multi_vehicle()
    elif page == "🗺️ Path Planning":
        show_path_planning()
    elif page == "📊 Analytics":
        show_analytics()
    elif page == "⚙️ Settings":
        show_settings()


def show_overview():
    st.header("📊 System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Vehicles", len(st.session_state.vehicles))
    
    with col2:
        st.metric("Simulation Steps", len(st.session_state.simulation_history))
    
    with col3:
        if st.session_state.vehicles:
            avg_battery = np.mean([v.config.battery_level for v in st.session_state.vehicles])
            st.metric("Avg Battery", f"{avg_battery:.1f}%")
        else:
            st.metric("Avg Battery", "N/A")
    
    with col4:
        st.metric("Environment", "Active" if st.session_state.environment else "None")
    
    st.markdown("---")
    
    # Quick start
    st.subheader("🚀 Quick Start")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Create Aerial Vehicle**")
        if st.button("Create Drone", type="primary"):
            drone = AerialVehicle(
                initial_state=State(position=np.array([0, 0, 10])),
                vehicle_id=f"drone_{len(st.session_state.vehicles) + 1}"
            )
            st.session_state.vehicles.append(drone)
            st.success(f"Created {drone.vehicle_id}")
            st.rerun()
    
    with col2:
        st.write("**Run Simulation**")
        if st.button("Run Basic Simulation", type="primary"):
            if st.session_state.vehicles:
                run_basic_simulation()
                st.success("Simulation completed!")
                st.rerun()
            else:
                st.warning("Please create a vehicle first")
    
    # Recent simulations
    if st.session_state.simulation_history:
        st.subheader("📈 Recent Simulation")
        latest = st.session_state.simulation_history[-1]
        
        if isinstance(latest, dict) and 'vehicles' in latest:
            # Multi-vehicle simulation
            fig = go.Figure()
            for vehicle_id, vehicle_data in latest['vehicles'].items():
                pos = vehicle_data['state'].position
                fig.add_trace(go.Scatter3d(
                    x=[pos[0]],
                    y=[pos[1]],
                    z=[pos[2]],
                    mode='markers',
                    name=vehicle_id,
                    marker=dict(size=10)
                ))
            fig.update_layout(title="Latest Vehicle Positions", scene=dict(aspectmode='cube'))
            st.plotly_chart(fig, use_container_width=True)


def show_single_vehicle():
    st.header("🚁 Single Vehicle Simulation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Vehicle Configuration")
        
        vehicle_type = st.selectbox("Vehicle Type", ["Aerial", "Ground", "Underwater"])
        
        col_x, col_y, col_z = st.columns(3)
        with col_x:
            init_x = st.number_input("Initial X (m)", value=0.0)
        with col_y:
            init_y = st.number_input("Initial Y (m)", value=0.0)
        with col_z:
            init_z = st.number_input("Initial Z (m)", value=10.0 if vehicle_type == "Aerial" else 0.0)
        
        max_speed = st.slider("Max Speed (m/s)", 1.0, 50.0, 20.0)
        mass = st.slider("Mass (kg)", 0.1, 10.0, 1.5)
        
        if st.button("Create Vehicle", type="primary"):
            config = VehicleConfig(max_speed=max_speed, mass=mass)
            initial_state = State(position=np.array([init_x, init_y, init_z]))
            
            if vehicle_type == "Aerial":
                vehicle = AerialVehicle(initial_state=initial_state, config=config)
            elif vehicle_type == "Ground":
                vehicle = GroundVehicle(initial_state=initial_state, config=config)
            else:
                vehicle = UnderwaterVehicle(initial_state=initial_state, config=config)
            
            st.session_state.vehicles.append(vehicle)
            st.success(f"Created {vehicle.vehicle_id}")
            st.rerun()
    
    with col2:
        st.subheader("Simulation Parameters")
        duration = st.slider("Duration (s)", 1.0, 60.0, 30.0)
        dt = st.slider("Time Step (s)", 0.01, 0.5, 0.1)
        
        if st.session_state.vehicles:
            selected_vehicle = st.selectbox(
                "Select Vehicle",
                [v.vehicle_id for v in st.session_state.vehicles]
            )
            
            if st.button("Run Simulation", type="primary"):
                vehicle = next(v for v in st.session_state.vehicles if v.vehicle_id == selected_vehicle)
                
                def control_policy(state, t):
                    return np.array([15.0, 0.0, 0.1, 0.0])
                
                simulator = Simulator(vehicle, control_policy=control_policy)
                history = simulator.run(duration=duration, dt=dt, verbose=False)
                
                st.session_state.simulation_history.extend(history)
                st.success(f"Simulation completed: {len(history)} steps")
                st.rerun()
    
    # Display vehicle info
    if st.session_state.vehicles:
        st.subheader("📋 Vehicle Status")
        
        for vehicle in st.session_state.vehicles:
            with st.expander(f"🚁 {vehicle.vehicle_id}"):
                col1, col2, col3 = st.columns(3)
                
                pos = vehicle.get_position()
                vel = vehicle.get_velocity()
                
                col1.metric("Position", f"[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]")
                col2.metric("Velocity", f"{np.linalg.norm(vel):.2f} m/s")
                col3.metric("Battery", f"{vehicle.config.battery_level:.1f}%")
        
        # Plot trajectory if available
        if st.session_state.simulation_history:
            plot_trajectories()


def show_multi_vehicle():
    st.header("👥 Multi-Vehicle Simulation")
    
    st.write("**Create Multiple Vehicles**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_vehicles = st.number_input("Number of Vehicles", 1, 10, 3)
        
        if st.button("Create Fleet", type="primary"):
            vehicles = []
            for i in range(num_vehicles):
                initial_state = State(position=np.array([i * 10, 0, 10]))
                vehicle = AerialVehicle(
                    initial_state=initial_state,
                    vehicle_id=f"drone_{i+1}"
                )
                vehicles.append(vehicle)
            
            st.session_state.vehicles = vehicles
            st.success(f"Created {num_vehicles} vehicles")
            st.rerun()
    
    with col2:
        duration = st.slider("Simulation Duration (s)", 1.0, 60.0, 30.0)
        dt = st.slider("Time Step (s)", 0.01, 0.5, 0.1)
        
        if st.button("Run Multi-Vehicle Simulation", type="primary"):
            if len(st.session_state.vehicles) > 1:
                def control_policy(state, t, all_vehicles):
                    return np.array([15.0, 0.0, 0.1, 0.0])
                
                control_policies = {v.vehicle_id: control_policy for v in st.session_state.vehicles}
                simulator = MultiVehicleSimulator(
                    st.session_state.vehicles,
                    control_policies=control_policies
                )
                
                history = simulator.run(duration=duration, dt=dt, verbose=False)
                st.session_state.simulation_history.extend(history)
                st.success("Multi-vehicle simulation completed!")
                st.rerun()
            else:
                st.warning("Please create multiple vehicles first")
    
    # Fleet visualization
    if len(st.session_state.vehicles) > 1:
        st.subheader("📍 Fleet Positions")
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3
        
        for i, vehicle in enumerate(st.session_state.vehicles):
            pos = vehicle.get_position()
            fig.add_trace(go.Scatter3d(
                x=[pos[0]],
                y=[pos[1]],
                z=[pos[2]],
                mode='markers+text',
                name=vehicle.vehicle_id,
                marker=dict(size=10, color=colors[i % len(colors)]),
                text=[vehicle.vehicle_id],
                textposition="top center"
            ))
        
        fig.update_layout(
            title="Fleet Positions",
            scene=dict(
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                zaxis_title="Z (m)",
                aspectmode='cube'
            ),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)


def show_path_planning():
    st.header("🗺️ Path Planning")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Environment Setup")
        
        num_obstacles = st.number_input("Number of Obstacles", 0, 10, 3)
        
        obstacles = []
        for i in range(num_obstacles):
            with st.expander(f"Obstacle {i+1}"):
                col_x, col_y, col_z = st.columns(3)
                with col_x:
                    obs_x = st.number_input(f"X {i+1}", value=30.0 + i*20, key=f"obs_x_{i}")
                with col_y:
                    obs_y = st.number_input(f"Y {i+1}", value=20.0 + i*15, key=f"obs_y_{i}")
                with col_z:
                    obs_z = st.number_input(f"Z {i+1}", value=5.0, key=f"obs_z_{i}")
                
                radius = st.number_input(f"Radius {i+1}", 1.0, 20.0, 10.0, key=f"obs_r_{i}")
                
                obstacles.append(Obstacle(
                    position=np.array([obs_x, obs_y, obs_z]),
                    radius=radius
                ))
        
        if st.button("Update Environment"):
            st.session_state.environment = Environment(obstacles=obstacles)
            st.success("Environment updated")
    
    with col2:
        st.subheader("Path Planning")
        
        planner_type = st.selectbox("Planner Type", ["A*", "RRT"])
        
        col_sx, col_sy, col_sz = st.columns(3)
        with col_sx:
            start_x = st.number_input("Start X", value=0.0)
        with col_sy:
            start_y = st.number_input("Start Y", value=0.0)
        with col_sz:
            start_z = st.number_input("Start Z", value=10.0)
        
        col_gx, col_gy, col_gz = st.columns(3)
        with col_gx:
            goal_x = st.number_input("Goal X", value=100.0)
        with col_gy:
            goal_y = st.number_input("Goal Y", value=100.0)
        with col_gz:
            goal_z = st.number_input("Goal Z", value=15.0)
        
        start = np.array([start_x, start_y, start_z])
        goal = np.array([goal_x, goal_y, goal_z])
        
        if st.button("Plan Path", type="primary"):
            if planner_type == "A*":
                planner = AStarPlanner(st.session_state.environment, resolution=5.0)
            else:
                planner = RRTPlanner(st.session_state.environment, max_iterations=500)
            
            with st.spinner("Planning path..."):
                path = planner.plan(start, goal)
            
            st.success(f"Path planned: {len(path)} waypoints")
            
            # Visualize path
            if path:
                fig = go.Figure()
                
                path_array = np.array(path)
                
                # Plot obstacles
                for obs in obstacles:
                    fig.add_trace(go.Scatter3d(
                        x=[obs.position[0]],
                        y=[obs.position[1]],
                        z=[obs.position[2]],
                        mode='markers',
                        marker=dict(size=obs.radius*2, color='red', opacity=0.5),
                        name=f"Obstacle at [{obs.position[0]:.0f}, {obs.position[1]:.0f}]"
                    ))
                
                # Plot path
                fig.add_trace(go.Scatter3d(
                    x=path_array[:, 0],
                    y=path_array[:, 1],
                    z=path_array[:, 2],
                    mode='lines+markers',
                    name='Planned Path',
                    line=dict(color='blue', width=4),
                    marker=dict(size=5)
                ))
                
                # Start and goal
                fig.add_trace(go.Scatter3d(
                    x=[start[0], goal[0]],
                    y=[start[1], goal[1]],
                    z=[start[2], goal[2]],
                    mode='markers',
                    marker=dict(size=10, color=['green', 'red']),
                    name='Start/Goal'
                ))
                
                fig.update_layout(
                    title="Path Planning Result",
                    scene=dict(aspectmode='cube'),
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)


def show_analytics():
    st.header("📊 Analytics & Visualization")
    
    if not st.session_state.simulation_history:
        st.info("No simulation data available. Run a simulation first.")
        return
    
    tab1, tab2, tab3 = st.tabs(["Trajectories", "State History", "Forecasting"])
    
    with tab1:
        plot_trajectories()
    
    with tab2:
        if st.session_state.simulation_history:
            plot_state_history()
    
    with tab3:
        show_forecasting()


def plot_trajectories():
    if not st.session_state.simulation_history:
        return
    
    fig = go.Figure()
    
    # Get latest simulation
    latest = st.session_state.simulation_history[-1]
    
    if isinstance(latest, dict):
        if 'vehicles' in latest:
            # Multi-vehicle
            for vehicle_id, vehicle_data in latest['vehicles'].items():
                pos = vehicle_data['state'].position
                fig.add_trace(go.Scatter3d(
                    x=[pos[0]],
                    y=[pos[1]],
                    z=[pos[2]],
                    mode='markers',
                    name=vehicle_id
                ))
        elif 'state' in latest:
            # Single vehicle
            pos = latest['state'].position
            fig.add_trace(go.Scatter3d(
                x=[pos[0]],
                y=[pos[1]],
                z=[pos[2]],
                mode='markers',
                name='Vehicle'
            ))
    
    fig.update_layout(
        title="Vehicle Trajectories",
        scene=dict(aspectmode='cube'),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_state_history():
    if not st.session_state.simulation_history:
        return
    
    # Extract state data
    timestamps = []
    positions_x = []
    positions_y = []
    positions_z = []
    velocities = []
    
    for step in st.session_state.simulation_history:
        if isinstance(step, dict) and 'state' in step:
            timestamps.append(step['timestamp'])
            pos = step['state'].position
            positions_x.append(pos[0])
            positions_y.append(pos[1])
            positions_z.append(pos[2])
            velocities.append(np.linalg.norm(step['state'].velocity))
    
    if not timestamps:
        return
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Position X', 'Position Y', 'Position Z', 'Speed'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    fig.add_trace(go.Scatter(x=timestamps, y=positions_x, name='X'), row=1, col=1)
    fig.add_trace(go.Scatter(x=timestamps, y=positions_y, name='Y'), row=1, col=2)
    fig.add_trace(go.Scatter(x=timestamps, y=positions_z, name='Z'), row=2, col=1)
    fig.add_trace(go.Scatter(x=timestamps, y=velocities, name='Speed'), row=2, col=2)
    
    fig.update_layout(height=600, title_text="State History")
    st.plotly_chart(fig, use_container_width=True)


def show_forecasting():
    st.subheader("Trajectory Forecasting")
    
    if not st.session_state.vehicles:
        st.warning("Please create a vehicle first")
        return
    
    selected_vehicle = st.selectbox(
        "Select Vehicle",
        [v.vehicle_id for v in st.session_state.vehicles]
    )
    
    vehicle = next(v for v in st.session_state.vehicles if v.vehicle_id == selected_vehicle)
    
    method = st.selectbox("Forecasting Method", ["constant_velocity", "linear", "polynomial"])
    horizon = st.slider("Forecast Horizon (s)", 1.0, 30.0, 10.0)
    
    if st.button("Generate Forecast"):
        forecaster = TrajectoryForecaster(method=method, horizon=horizon)
        
        # Update with history if available
        if st.session_state.simulation_history:
            for step in st.session_state.simulation_history[-20:]:
                if isinstance(step, dict) and 'state' in step:
                    forecaster.update_history(step['state'])
        
        forecast = forecaster.forecast(vehicle.get_state(), horizon=horizon)
        
        # Plot forecast
        fig = go.Figure()
        
        current_pos = vehicle.get_position()
        forecast_array = np.array(forecast)
        
        # Current position
        fig.add_trace(go.Scatter3d(
            x=[current_pos[0]],
            y=[current_pos[1]],
            z=[current_pos[2]],
            mode='markers',
            marker=dict(size=10, color='green'),
            name='Current'
        ))
        
        # Forecast
        fig.add_trace(go.Scatter3d(
            x=forecast_array[:, 0],
            y=forecast_array[:, 1],
            z=forecast_array[:, 2],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='blue', width=4),
            marker=dict(size=4)
        ))
        
        fig.update_layout(
            title=f"Trajectory Forecast ({method})",
            scene=dict(aspectmode='cube'),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)


def show_settings():
    st.header("⚙️ Settings")
    
    st.subheader("System Configuration")
    
    st.write("**Default Parameters**")
    
    default_max_speed = st.slider("Default Max Speed (m/s)", 1.0, 50.0, 20.0)
    default_mass = st.slider("Default Mass (kg)", 0.1, 10.0, 1.5)
    
    if st.button("Reset Vehicles"):
        st.session_state.vehicles = []
        st.success("Vehicles reset")
        st.rerun()
    
    if st.button("Clear History"):
        st.session_state.simulation_history = []
        st.success("History cleared")
        st.rerun()


def run_basic_simulation():
    """Run a basic simulation for demonstration."""
    if not st.session_state.vehicles:
        # Create a default vehicle
        vehicle = AerialVehicle(
            initial_state=State(position=np.array([0, 0, 10])),
            vehicle_id="demo_drone"
        )
        st.session_state.vehicles.append(vehicle)
    
    vehicle = st.session_state.vehicles[0]
    
    def control_policy(state, t):
        return np.array([15.0, 0.0, 0.1, 0.0])
    
    simulator = Simulator(vehicle, control_policy=control_policy)
    history = simulator.run(duration=30.0, dt=0.1, verbose=False)
    
    st.session_state.simulation_history.extend(history)


if __name__ == "__main__":
    main()

