"""
China UAV Market Analysis and Simulation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
from datetime import datetime, timedelta


class ChinaUAVMarketSimulator:
    """Simulate China's UAV market growth and projections."""
    
    def __init__(self):
        """Initialize market simulator with current data."""
        # 2024 baseline data
        self.baseline_2024 = {
            'total_market_value': 28.5,  # billion USD
            'china_market_share': 0.70,  # 70%
            'china_domestic_market': 8.5,  # billion USD
            'china_export_value': 12.3,  # billion USD
            'global_units_sold': 8.5,  # million units
            'china_manufacturing_capacity': 8.0,  # million units
            'commercial_market': 11.4,  # billion USD
            'consumer_market': 10.0,  # billion USD
            'agriculture_market': 1.4,  # billion USD
        }
        
        # Growth rates (CAGR)
        self.growth_rates = {
            'total_market': 0.158,  # 15.8%
            'china_domestic': 0.185,  # 18.5%
            'china_export': 0.165,  # 16.5%
            'commercial': 0.250,  # 25.0%
            'consumer': 0.072,  # 7.2%
            'agriculture': 0.225,  # 22.5%
            'units_sold': 0.192,  # 19.2%
        }
        
        # China-specific factors
        self.china_factors = {
            'manufacturing_cost_advantage': 0.30,  # 30% cost advantage
            'government_support_multiplier': 1.15,  # 15% boost
            'supply_chain_efficiency': 1.20,  # 20% efficiency gain
            'r_and_d_investment': 2.0,  # billion USD annually
        }
    
    def project_market(self, target_year: int = 2026) -> Dict:
        """
        Project market values to target year.
        
        Args:
            target_year: Target year for projection
            
        Returns:
            Dictionary with projected values
        """
        years_ahead = target_year - 2024
        
        projections = {}
        
        # Total global market
        projections['global_market'] = self.baseline_2024['total_market_value'] * \
            (1 + self.growth_rates['total_market']) ** years_ahead
        
        # China market share (maintains dominance)
        projections['china_market_share'] = self.baseline_2024['china_market_share']
        
        # China domestic market
        projections['china_domestic'] = self.baseline_2024['china_domestic_market'] * \
            (1 + self.growth_rates['china_domestic']) ** years_ahead
        
        # China export value
        projections['china_export'] = self.baseline_2024['china_export_value'] * \
            (1 + self.growth_rates['china_export']) ** years_ahead
        
        # China total market value
        projections['china_total'] = projections['china_domestic'] + projections['china_export']
        
        # Units sold
        projections['global_units'] = self.baseline_2024['global_units_sold'] * \
            (1 + self.growth_rates['units_sold']) ** years_ahead
        
        # China manufacturing capacity
        projections['china_capacity'] = self.baseline_2024['china_manufacturing_capacity'] * \
            (1 + 0.15) ** years_ahead  # 15% capacity growth
        
        # Segment projections
        projections['commercial'] = self.baseline_2024['commercial_market'] * \
            (1 + self.growth_rates['commercial']) ** years_ahead
        
        projections['consumer'] = self.baseline_2024['consumer_market'] * \
            (1 + self.growth_rates['consumer']) ** years_ahead
        
        projections['agriculture'] = self.baseline_2024['agriculture_market'] * \
            (1 + self.growth_rates['agriculture']) ** years_ahead
        
        # Market dominance metrics
        projections['china_dominance_score'] = self._calculate_dominance_score(projections)
        
        return projections
    
    def _calculate_dominance_score(self, projections: Dict) -> float:
        """Calculate China's market dominance score (0-100)."""
        factors = [
            projections['china_market_share'] * 40,  # Market share weight
            min(projections['china_capacity'] / projections['global_units'], 1.0) * 30,  # Capacity
            min(projections['china_export'] / projections['global_market'], 1.0) * 30,  # Export dominance
        ]
        return sum(factors)
    
    def simulate_scenarios(self, target_year: int = 2026) -> Dict:
        """
        Simulate different market scenarios.
        
        Returns:
            Dictionary with optimistic, baseline, and pessimistic scenarios
        """
        scenarios = {}
        
        # Baseline scenario
        scenarios['baseline'] = self.project_market(target_year)
        
        # Optimistic scenario (faster growth, stronger China position)
        original_rates = self.growth_rates.copy()
        self.growth_rates['total_market'] *= 1.2
        self.growth_rates['china_export'] *= 1.15
        scenarios['optimistic'] = self.project_market(target_year)
        self.growth_rates = original_rates
        
        # Pessimistic scenario (slower growth, increased competition)
        self.growth_rates['total_market'] *= 0.85
        self.growth_rates['china_market_share'] = 0.65  # Reduced share
        scenarios['pessimistic'] = self.project_market(target_year)
        self.growth_rates['total_market'] = original_rates['total_market']
        self.growth_rates['china_market_share'] = 0.70
        
        return scenarios
    
    def generate_forecast_dataframe(self, start_year: int = 2024, end_year: int = 2026) -> pd.DataFrame:
        """
        Generate forecast data for multiple years.
        
        Args:
            start_year: Starting year
            end_year: Ending year
            
        Returns:
            DataFrame with year-by-year projections
        """
        data = []
        
        for year in range(start_year, end_year + 1):
            proj = self.project_market(year)
            data.append({
                'Year': year,
                'Global_Market_Billions': proj['global_market'],
                'China_Domestic_Billions': proj['china_domestic'],
                'China_Export_Billions': proj['china_export'],
                'China_Total_Billions': proj['china_total'],
                'China_Market_Share_Pct': proj['china_market_share'] * 100,
                'Global_Units_Millions': proj['global_units'],
                'China_Capacity_Millions': proj['china_capacity'],
                'Commercial_Market_Billions': proj['commercial'],
                'Consumer_Market_Billions': proj['consumer'],
                'Agriculture_Market_Billions': proj['agriculture'],
                'Dominance_Score': proj['china_dominance_score'],
            })
        
        return pd.DataFrame(data)
    
    def plot_projections(self, save_path: str = 'china_uav_projections.png'):
        """Plot market projections."""
        df = self.generate_forecast_dataframe(2024, 2026)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Market value projection
        ax1 = axes[0, 0]
        ax1.plot(df['Year'], df['Global_Market_Billions'], 'b-o', label='Global Market', linewidth=2)
        ax1.plot(df['Year'], df['China_Total_Billions'], 'r-s', label='China Total', linewidth=2)
        ax1.plot(df['Year'], df['China_Domestic_Billions'], 'g-^', label='China Domestic', linewidth=2)
        ax1.plot(df['Year'], df['China_Export_Billions'], 'm-d', label='China Export', linewidth=2)
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Market Value (Billion USD)')
        ax1.set_title('UAV Market Projections 2024-2026')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Market share
        ax2 = axes[0, 1]
        ax2.bar(df['Year'], df['China_Market_Share_Pct'], color='red', alpha=0.7)
        ax2.set_xlabel('Year')
        ax2.set_ylabel('China Market Share (%)')
        ax2.set_title('China Market Share')
        ax2.set_ylim([0, 100])
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Units and capacity
        ax3 = axes[1, 0]
        ax3.plot(df['Year'], df['Global_Units_Millions'], 'b-o', label='Global Units Sold', linewidth=2)
        ax3.plot(df['Year'], df['China_Capacity_Millions'], 'r-s', label='China Capacity', linewidth=2)
        ax3.set_xlabel('Year')
        ax3.set_ylabel('Units (Millions)')
        ax3.set_title('Manufacturing Capacity vs Demand')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Market segments
        ax4 = axes[1, 1]
        segments = ['Commercial', 'Consumer', 'Agriculture']
        values_2024 = [
            df.loc[df['Year'] == 2024, 'Commercial_Market_Billions'].values[0],
            df.loc[df['Year'] == 2024, 'Consumer_Market_Billions'].values[0],
            df.loc[df['Year'] == 2024, 'Agriculture_Market_Billions'].values[0],
        ]
        values_2026 = [
            df.loc[df['Year'] == 2026, 'Commercial_Market_Billions'].values[0],
            df.loc[df['Year'] == 2026, 'Consumer_Market_Billions'].values[0],
            df.loc[df['Year'] == 2026, 'Agriculture_Market_Billions'].values[0],
        ]
        
        x = np.arange(len(segments))
        width = 0.35
        ax4.bar(x - width/2, values_2024, width, label='2024', alpha=0.7)
        ax4.bar(x + width/2, values_2026, width, label='2026', alpha=0.7)
        ax4.set_xlabel('Market Segment')
        ax4.set_ylabel('Market Value (Billion USD)')
        ax4.set_title('Market Segment Growth')
        ax4.set_xticks(x)
        ax4.set_xticklabels(segments)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Projections plot saved to {save_path}")
        
        return fig


def main():
    """Run China UAV market analysis."""
    print("=" * 60)
    print("China UAV Market Analysis & 2026 Projections")
    print("=" * 60)
    
    simulator = ChinaUAVMarketSimulator()
    
    # Generate 2026 projections
    print("\n2026 Market Projections:")
    print("-" * 60)
    projections_2026 = simulator.project_market(2026)
    
    print(f"Global Market Value: ${projections_2026['global_market']:.2f} billion")
    print(f"China Total Market: ${projections_2026['china_total']:.2f} billion")
    print(f"  - Domestic: ${projections_2026['china_domestic']:.2f} billion")
    print(f"  - Export: ${projections_2026['china_export']:.2f} billion")
    print(f"China Market Share: {projections_2026['china_market_share']*100:.1f}%")
    print(f"Global Units Sold: {projections_2026['global_units']:.1f} million")
    print(f"China Manufacturing Capacity: {projections_2026['china_capacity']:.1f} million units")
    print(f"Dominance Score: {projections_2026['china_dominance_score']:.1f}/100")
    
    # Market segments
    print("\nMarket Segments (2026):")
    print("-" * 60)
    print(f"Commercial: ${projections_2026['commercial']:.2f} billion")
    print(f"Consumer: ${projections_2026['consumer']:.2f} billion")
    print(f"Agriculture: ${projections_2026['agriculture']:.2f} billion")
    
    # Scenarios
    print("\nMarket Scenarios (2026):")
    print("-" * 60)
    scenarios = simulator.simulate_scenarios(2026)
    
    for scenario_name, scenario_data in scenarios.items():
        print(f"\n{scenario_name.upper()} Scenario:")
        print(f"  Global Market: ${scenario_data['global_market']:.2f} billion")
        print(f"  China Total: ${scenario_data['china_total']:.2f} billion")
        print(f"  China Share: {scenario_data['china_market_share']*100:.1f}%")
    
    # Generate forecast dataframe
    print("\nGenerating Forecast Data...")
    df = simulator.generate_forecast_dataframe(2024, 2026)
    print("\nForecast Data:")
    print(df.to_string(index=False))
    
    # Save to CSV
    csv_path = 'china_uav_forecast_2024_2026.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nForecast data saved to {csv_path}")
    
    # Plot projections
    print("\nGenerating visualization...")
    simulator.plot_projections('china_uav_projections.png')
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

