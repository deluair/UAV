"""
UAV Industry Economic Insights & Market Projections Dashboard
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List
import sys
import os

# Add parent directory to path to import analysis module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from analysis.china_market_analysis import ChinaUAVMarketSimulator
except ImportError:
    st.error("Market analysis module not found. Please ensure analysis/china_market_analysis.py exists.")
    st.stop()


st.set_page_config(
    page_title="UAV Economic Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'market_data' not in st.session_state:
    st.session_state.market_data = None
if 'simulator' not in st.session_state:
    st.session_state.simulator = ChinaUAVMarketSimulator()


def main():
    st.title("📊 UAV Industry Economic Insights & Market Projections")
    st.markdown("**Comprehensive Market Analysis, China Dominance, and 2026 Forecasts**")
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("📈 Navigation")
    
    page = st.sidebar.selectbox(
        "Select Analysis",
        [
            "🏠 Market Overview",
            "🇨🇳 China Analysis",
            "📊 Market Segments",
            "🌍 Regional Analysis",
            "💡 Investment Opportunities",
            "🔮 2026 Projections",
            "📈 Scenario Analysis",
            "⚙️ Custom Forecast"
        ]
    )
    
    if page == "🏠 Market Overview":
        show_market_overview()
    elif page == "🇨🇳 China Analysis":
        show_china_analysis()
    elif page == "📊 Market Segments":
        show_market_segments()
    elif page == "🌍 Regional Analysis":
        show_regional_analysis()
    elif page == "💡 Investment Opportunities":
        show_investment_opportunities()
    elif page == "🔮 2026 Projections":
        show_2026_projections()
    elif page == "📈 Scenario Analysis":
        show_scenario_analysis()
    elif page == "⚙️ Custom Forecast":
        show_custom_forecast()


def show_market_overview():
    st.header("🏠 Global UAV Market Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    projections_2026 = st.session_state.simulator.project_market(2026)
    projections_2024 = st.session_state.simulator.project_market(2024)
    
    with col1:
        st.metric(
            "2024 Market Size",
            f"${projections_2024['global_market']:.1f}B",
            delta=f"{projections_2024['global_market'] - 25.0:.1f}B vs 2023"
        )
    
    with col2:
        st.metric(
            "2026 Projected",
            f"${projections_2026['global_market']:.1f}B",
            delta=f"+{(projections_2026['global_market'] / projections_2024['global_market'] - 1) * 100:.1f}%"
        )
    
    with col3:
        cagr = 15.8
        st.metric(
            "CAGR (2024-2026)",
            f"{cagr:.1f}%",
            delta="Strong Growth"
        )
    
    with col4:
        units_2026 = projections_2026['global_units']
        st.metric(
            "Units Sold (2026)",
            f"{units_2026:.1f}M",
            delta=f"+{(units_2026 / projections_2024['global_units'] - 1) * 100:.1f}%"
        )
    
    st.markdown("---")
    
    # Market growth chart
    st.subheader("📈 Market Growth Trajectory")
    
    years = [2024, 2025, 2026]
    market_values = [
        st.session_state.simulator.project_market(y)['global_market'] 
        for y in years
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=years,
        y=market_values,
        mode='lines+markers',
        name='Global Market',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        title="UAV Market Value Growth (2024-2026)",
        xaxis_title="Year",
        yaxis_title="Market Value (Billion USD)",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Market drivers
    st.subheader("🚀 Key Market Drivers")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Commercial Applications**
        - Logistics & Delivery
        - Agriculture
        - Infrastructure Inspection
        - Real Estate
        """)
    
    with col2:
        st.markdown("""
        **Technology Advances**
        - AI & Machine Learning
        - Battery Improvements
        - Enhanced Sensors
        - 5G Connectivity
        """)
    
    with col3:
        st.markdown("""
        **Regulatory Evolution**
        - BVLOS Approvals
        - UTM Systems
        - Standardized Rules
        - Privacy Frameworks
        """)


def show_china_analysis():
    st.header("🇨🇳 China's Market Dominance Analysis")
    
    projections_2026 = st.session_state.simulator.project_market(2026)
    projections_2024 = st.session_state.simulator.project_market(2024)
    
    # China metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Market Share",
            f"{projections_2026['china_market_share']*100:.1f}%",
            delta="Dominant Position"
        )
    
    with col2:
        st.metric(
            "Total Market Value (2026)",
            f"${projections_2026['china_total']:.1f}B",
            delta=f"+{((projections_2026['china_total'] / projections_2024['china_total']) - 1) * 100:.1f}%"
        )
    
    with col3:
        st.metric(
            "Export Value (2026)",
            f"${projections_2026['china_export']:.1f}B",
            delta="Global Leader"
        )
    
    with col4:
        st.metric(
            "Manufacturing Capacity",
            f"{projections_2026['china_capacity']:.1f}M units",
            delta="World's Largest"
        )
    
    st.markdown("---")
    
    # China vs Global comparison
    st.subheader("📊 China vs Global Market")
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Market Value Comparison', 'Market Share Over Time'),
        specs=[[{"type": "bar"}, {"type": "scatter"}]]
    )
    
    years = [2024, 2025, 2026]
    global_markets = [st.session_state.simulator.project_market(y)['global_market'] for y in years]
    china_totals = [st.session_state.simulator.project_market(y)['china_total'] for y in years]
    china_shares = [st.session_state.simulator.project_market(y)['china_market_share'] * 100 for y in years]
    
    # Bar chart
    fig.add_trace(
        go.Bar(name='Global Market', x=years, y=global_markets, marker_color='#1f77b4'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(name='China Total', x=years, y=china_totals, marker_color='#ff7f0e'),
        row=1, col=1
    )
    
    # Line chart
    fig.add_trace(
        go.Scatter(x=years, y=china_shares, mode='lines+markers', name='China Share %', line=dict(color='red', width=3)),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=True, title_text="China Market Position")
    fig.update_yaxes(title_text="Billion USD", row=1, col=1)
    fig.update_yaxes(title_text="Market Share (%)", row=1, col=2)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Key Chinese players
    st.subheader("🏢 Key Chinese Players")
    
    players_data = {
        'Company': ['DJI', 'Autel Robotics', 'Ehang', 'Zero Zero Robotics'],
        'Market Share': [70, 4, 2, 1],
        'Focus': ['Consumer/Enterprise', 'Professional', 'Urban Air Mobility', 'Consumer Selfie'],
        'Revenue (Est.)': ['$3B+', '$200M+', '$50M+', '$30M+']
    }
    
    df_players = pd.DataFrame(players_data)
    st.dataframe(df_players, use_container_width=True, hide_index=True)
    
    # China advantages
    st.subheader("💪 China's Strategic Advantages")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Manufacturing Ecosystem**
        - Complete supply chain in Shenzhen
        - 30% cost advantage
        - Rapid prototyping
        - Component suppliers network
        """)
        
        st.markdown("""
        **Government Support**
        - "Made in China 2025" initiative
        - Development subsidies
        - Domestic market protection
        - Export promotion
        """)
    
    with col2:
        st.markdown("""
        **Technical Innovation**
        - Leading AI/ML integration
        - Advanced gimbal technology
        - Battery optimization
        - Compact design expertise
        """)
        
        st.markdown("""
        **Market Access**
        - 1.4B population domestic market
        - Established distribution
        - Strong e-commerce platforms
        - Global export network
        """)


def show_market_segments():
    st.header("📊 Market Segmentation Analysis")
    
    projections_2026 = st.session_state.simulator.project_market(2026)
    projections_2024 = st.session_state.simulator.project_market(2024)
    
    # Segment breakdown
    segments = {
        'Commercial': projections_2026['commercial'],
        'Consumer': projections_2026['consumer'],
        'Agriculture': projections_2026['agriculture'],
        'Military': 7.2,  # From industry data
    }
    
    # Pie chart
    fig = go.Figure(data=[go.Pie(
        labels=list(segments.keys()),
        values=list(segments.values()),
        hole=0.4,
        marker_colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    )])
    
    fig.update_layout(
        title="Market Share by Segment (2026)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Segment growth comparison
    st.subheader("📈 Segment Growth Comparison")
    
    segment_data = {
        'Segment': ['Commercial', 'Consumer', 'Agriculture', 'Military'],
        '2024 (B$)': [11.4, 10.0, 1.4, 5.7],
        '2026 (B$)': [17.8, 11.5, 2.1, 7.2],
        'CAGR (%)': [25.0, 7.2, 22.5, 12.3]
    }
    
    df_segments = pd.DataFrame(segment_data)
    df_segments['Growth'] = ((df_segments['2026 (B$)'] / df_segments['2024 (B$)']) - 1) * 100
    
    # Bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='2024',
        x=df_segments['Segment'],
        y=df_segments['2024 (B$)'],
        marker_color='#1f77b4'
    ))
    
    fig.add_trace(go.Bar(
        name='2026',
        x=df_segments['Segment'],
        y=df_segments['2026 (B$)'],
        marker_color='#ff7f0e'
    ))
    
    fig.update_layout(
        title="Market Segment Growth (2024-2026)",
        xaxis_title="Segment",
        yaxis_title="Market Value (Billion USD)",
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed segment table
    st.subheader("📋 Detailed Segment Analysis")
    st.dataframe(df_segments, use_container_width=True, hide_index=True)
    
    # Application breakdown
    st.subheader("🎯 Key Applications by Segment")
    
    applications = {
        'Commercial': {
            'Logistics & Delivery': 5.2,
            'Inspection & Monitoring': 4.8,
            'Construction': 1.9,
            'Energy': 1.6,
            'Real Estate': 1.2,
            'Others': 3.1
        },
        'Consumer': {
            'Photography/Videography': 6.5,
            'Racing/Hobby': 2.8,
            'Selfie Drones': 1.5,
            'Educational': 0.7
        },
        'Agriculture': {
            'Crop Monitoring': 0.8,
            'Spraying': 0.7,
            'Mapping': 0.4,
            'Livestock': 0.2
        }
    }
    
    for segment, apps in applications.items():
        with st.expander(f"{segment} Applications"):
            fig = go.Figure(data=[go.Bar(
                x=list(apps.keys()),
                y=list(apps.values()),
                marker_color='#2ca02c'
            )])
            fig.update_layout(
                title=f"{segment} Market Breakdown (2026)",
                xaxis_title="Application",
                yaxis_title="Market Value (Billion USD)",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)


def show_regional_analysis():
    st.header("🌍 Regional Market Analysis")
    
    # Regional data
    regional_data = {
        'Region': ['Asia-Pacific', 'North America', 'Europe', 'Rest of World'],
        '2024 (B$)': [12.8, 8.0, 5.1, 2.6],
        '2026 (B$)': [18.5, 10.4, 6.5, 3.8],
        'CAGR (%)': [18.5, 14.2, 12.8, 16.3],
        'Market Share (%)': [45, 28, 18, 9]
    }
    
    df_regions = pd.DataFrame(regional_data)
    
    # Regional comparison chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='2024',
        x=df_regions['Region'],
        y=df_regions['2024 (B$)'],
        marker_color='#1f77b4'
    ))
    
    fig.add_trace(go.Bar(
        name='2026',
        x=df_regions['Region'],
        y=df_regions['2026 (B$)'],
        marker_color='#ff7f0e'
    ))
    
    fig.update_layout(
        title="Regional Market Comparison",
        xaxis_title="Region",
        yaxis_title="Market Value (Billion USD)",
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Market share pie chart
    fig = go.Figure(data=[go.Pie(
        labels=df_regions['Region'],
        values=df_regions['Market Share (%)'],
        hole=0.3,
        marker_colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    )])
    
    fig.update_layout(
        title="Regional Market Share (2026)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Regional details
    st.subheader("📋 Regional Market Details")
    st.dataframe(df_regions, use_container_width=True, hide_index=True)
    
    # Key markets by region
    st.subheader("🗺️ Key Markets by Region")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Asia-Pacific**
        - China: Manufacturing hub, largest market
        - Japan: Advanced robotics, commercial apps
        - India: Emerging market, agriculture focus
        - South Korea: Defense, logistics
        """)
        
        st.markdown("""
        **North America**
        - United States: Largest commercial market
        - Canada: Resource extraction
        """)
    
    with col2:
        st.markdown("""
        **Europe**
        - UK: Advanced regulations
        - Germany: Industrial applications
        - France: Defense and security
        """)
        
        st.markdown("""
        **Rest of World**
        - Latin America: Agricultural growth
        - Middle East: Defense applications
        - Africa: Emerging opportunities
        """)


def show_investment_opportunities():
    st.header("💡 Investment Opportunities")
    
    # High-growth segments
    st.subheader("🚀 High-Growth Investment Segments")
    
    investment_data = {
        'Segment': [
            'Autonomous Delivery',
            'Urban Air Mobility',
            'Agricultural Drones',
            'Drone-as-a-Service',
            'Enterprise Software',
            'Sensor Technology'
        ],
        'CAGR (%)': [35, 45, 22.5, 28, 32, 18],
        'Market Size 2026 (B$)': [5.2, 2.8, 2.1, 1.5, 1.2, 6.8],
        'Investment Required': ['$500M+', '$2B+', '$200M+', '$100M+', '$150M+', '$300M+']
    }
    
    df_invest = pd.DataFrame(investment_data)
    
    # Investment opportunity chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_invest['Segment'],
        y=df_invest['CAGR (%)'],
        marker_color=df_invest['CAGR (%)'],
        marker_colorscale='Viridis',
        text=df_invest['CAGR (%)'],
        texttemplate='%{text:.1f}%',
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Investment Opportunities by Growth Rate",
        xaxis_title="Segment",
        yaxis_title="CAGR (%)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Investment table
    st.subheader("📊 Investment Opportunity Details")
    st.dataframe(df_invest, use_container_width=True, hide_index=True)
    
    # Key players by segment
    st.subheader("🏢 Key Players by Investment Segment")
    
    players_by_segment = {
        'Autonomous Delivery': ['Amazon Prime Air', 'Wing (Alphabet)', 'JD.com'],
        'Urban Air Mobility': ['Ehang', 'Joby Aviation', 'Volocopter'],
        'Agricultural Drones': ['DJI Agras', 'XAG', 'PrecisionHawk'],
        'Drone-as-a-Service': ['DroneDeploy', 'Measure', 'Skyward']
    }
    
    for segment, players in players_by_segment.items():
        with st.expander(segment):
            for player in players:
                st.write(f"- {player}")
    
    # Regional investment hotspots
    st.subheader("🌍 Regional Investment Hotspots")
    
    hotspots = {
        'China': 'Manufacturing, R&D, domestic market expansion',
        'United States': 'Commercial applications, urban air mobility',
        'Europe': 'Industrial automation, regulatory innovation',
        'India': 'Agricultural applications, logistics',
        'Southeast Asia': 'E-commerce delivery, infrastructure'
    }
    
    for region, focus in hotspots.items():
        st.markdown(f"**{region}**: {focus}")


def show_2026_projections():
    st.header("🔮 2026 Market Projections")
    
    projections_2026 = st.session_state.simulator.project_market(2026)
    
    # Key projections
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Global Market",
            f"${projections_2026['global_market']:.1f}B",
            delta="15.8% CAGR"
        )
    
    with col2:
        st.metric(
            "Units Sold",
            f"{projections_2026['global_units']:.1f}M",
            delta="19.2% CAGR"
        )
    
    with col3:
        st.metric(
            "China Share",
            f"{projections_2026['china_market_share']*100:.1f}%",
            delta="Maintained"
        )
    
    st.markdown("---")
    
    # Detailed forecast table
    st.subheader("📋 Detailed 2026 Forecast")
    
    forecast_df = st.session_state.simulator.generate_forecast_dataframe(2024, 2026)
    st.dataframe(forecast_df, use_container_width=True, hide_index=True)
    
    # Key predictions
    st.subheader("🎯 Key 2026 Predictions")
    
    predictions = [
        {
            'Category': 'Market Consolidation',
            'Prediction': 'Top 5 companies will control 75% of the market'
        },
        {
            'Category': 'Technology Breakthroughs',
            'Prediction': 'Autonomous delivery in 500+ cities, AI in 90% of commercial drones'
        },
        {
            'Category': 'Regulatory Evolution',
            'Prediction': 'BVLOS operations approved in 50+ countries, UTM systems deployed globally'
        },
        {
            'Category': 'New Applications',
            'Prediction': 'Last-mile delivery ($5.2B), Urban air taxis launch, 15% of farms using drones'
        },
        {
            'Category': "China's Position",
            'Prediction': 'Maintains 65-70% market share, $18-20B export value, 12M+ units annually'
        }
    ]
    
    for pred in predictions:
        with st.expander(pred['Category']):
            st.write(pred['Prediction'])
    
    # Technology impact projections
    st.subheader("🔬 Technology Impact Projections")
    
    tech_impact = {
        'Technology': ['AI & Autonomy', 'Battery Tech', 'Sensors', '5G Connectivity'],
        'Market Impact 2026 (B$)': [12.5, 8.2, 6.8, 4.5],
        'Key Benefit': [
            'Fully autonomous operations',
            '60+ min flight time',
            'Advanced environmental awareness',
            'Real-time data transmission'
        ]
    }
    
    df_tech = pd.DataFrame(tech_impact)
    
    fig = go.Figure(data=[go.Bar(
        x=df_tech['Technology'],
        y=df_tech['Market Impact 2026 (B$)'],
        marker_color='#2ca02c',
        text=df_tech['Market Impact 2026 (B$)'],
        texttemplate='$%{text:.1f}B',
        textposition='outside'
    )])
    
    fig.update_layout(
        title="Technology Market Impact (2026)",
        xaxis_title="Technology",
        yaxis_title="Market Value (Billion USD)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(df_tech, use_container_width=True, hide_index=True)


def show_scenario_analysis():
    st.header("📈 Scenario Analysis")
    
    scenarios = st.session_state.simulator.simulate_scenarios(2026)
    
    # Scenario comparison
    scenario_names = list(scenarios.keys())
    global_markets = [scenarios[s]['global_market'] for s in scenario_names]
    china_totals = [scenarios[s]['china_total'] for s in scenario_names]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Global Market by Scenario', 'China Market by Scenario'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    colors = {'baseline': '#1f77b4', 'optimistic': '#2ca02c', 'pessimistic': '#d62728'}
    
    fig.add_trace(
        go.Bar(x=scenario_names, y=global_markets, marker_color=[colors[s] for s in scenario_names], name='Global'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=scenario_names, y=china_totals, marker_color=[colors[s] for s in scenario_names], name='China'),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False, title_text="2026 Scenario Comparison")
    fig.update_yaxes(title_text="Billion USD", row=1, col=1)
    fig.update_yaxes(title_text="Billion USD", row=1, col=2)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Scenario details
    st.subheader("📊 Scenario Details")
    
    for scenario_name, scenario_data in scenarios.items():
        with st.expander(f"{scenario_name.upper()} Scenario"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Global Market", f"${scenario_data['global_market']:.2f}B")
            with col2:
                st.metric("China Total", f"${scenario_data['china_total']:.2f}B")
            with col3:
                st.metric("China Share", f"{scenario_data['china_market_share']*100:.1f}%")
            
            if scenario_name == 'optimistic':
                st.info("""
                **Assumptions:**
                - Faster market growth (20% higher CAGR)
                - Strong China export growth
                - Accelerated technology adoption
                - Favorable regulatory environment
                """)
            elif scenario_name == 'pessimistic':
                st.warning("""
                **Assumptions:**
                - Slower market growth (15% lower CAGR)
                - Increased competition reduces China share
                - Regulatory delays
                - Economic headwinds
                """)
            else:
                st.success("""
                **Assumptions:**
                - Current growth trends continue
                - China maintains market dominance
                - Steady technology adoption
                - Moderate regulatory evolution
                """)


def show_custom_forecast():
    st.header("⚙️ Custom Market Forecast")
    
    st.subheader("📊 Forecast Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_year = st.number_input("Start Year", 2024, 2030, 2024)
        end_year = st.number_input("End Year", 2024, 2030, 2026)
        
        if end_year <= start_year:
            st.error("End year must be greater than start year")
            return
    
    with col2:
        growth_multiplier = st.slider("Growth Rate Multiplier", 0.5, 2.0, 1.0, 0.1)
        china_share = st.slider("China Market Share (%)", 50.0, 80.0, 70.0, 1.0)
    
    if st.button("Generate Custom Forecast", type="primary"):
        # Adjust growth rates
        original_rates = st.session_state.simulator.growth_rates.copy()
        for key in st.session_state.simulator.growth_rates:
            st.session_state.simulator.growth_rates[key] *= growth_multiplier
        
        # Adjust China share
        original_share = st.session_state.simulator.baseline_2024['china_market_share']
        st.session_state.simulator.baseline_2024['china_market_share'] = china_share / 100
        
        # Generate forecast
        forecast_df = st.session_state.simulator.generate_forecast_dataframe(start_year, end_year)
        
        # Restore original values
        st.session_state.simulator.growth_rates = original_rates
        st.session_state.simulator.baseline_2024['china_market_share'] = original_share
        
        # Display results
        st.subheader("📈 Custom Forecast Results")
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)
        
        # Plot custom forecast
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=forecast_df['Year'],
            y=forecast_df['Global_Market_Billions'],
            mode='lines+markers',
            name='Global Market',
            line=dict(color='#1f77b4', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_df['Year'],
            y=forecast_df['China_Total_Billions'],
            mode='lines+markers',
            name='China Total',
            line=dict(color='#ff7f0e', width=3)
        ))
        
        fig.update_layout(
            title=f"Custom Forecast ({start_year}-{end_year})",
            xaxis_title="Year",
            yaxis_title="Market Value (Billion USD)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Download option
        csv = forecast_df.to_csv(index=False)
        st.download_button(
            label="Download Forecast CSV",
            data=csv,
            file_name=f"custom_forecast_{start_year}_{end_year}.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
