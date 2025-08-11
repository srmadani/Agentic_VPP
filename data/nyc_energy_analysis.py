#!/usr/bin/env python3
"""
NYC Energy Market Analysis with Solar Generation
Complete toolkit for NYISO NYC pricing data and solar PV generation analysis

This script provides:
1. Data loading and verification
2. Solar generation calculations for 1kWp systems
3. Energy market analysis and visualizations
4. Economic analysis for VPP applications

Data: July 1-8, 2025, 5-minute intervals, NYC zone only
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class NYCEnergyAnalyzer:
    """Complete analyzer for NYC energy market and solar generation data"""
    
    def __init__(self, data_file="rates_pv_1kwp.parquet"):
        """Initialize with data file"""
        self.data_file = data_file
        self.df = None
        
    def load_data(self):
        """Load the NYC pricing and solar data"""
        try:
            self.df = pd.read_parquet(self.data_file)
            print(f"✅ Loaded {len(self.df):,} records from {self.data_file}")
            return True
        except FileNotFoundError:
            print(f"❌ Data file {self.data_file} not found!")
            return False
    
    def verify_data(self):
        """Verify data completeness and structure"""
        if self.df is None:
            print("❌ No data loaded. Call load_data() first.")
            return
            
        print("\n🔍 DATA VERIFICATION")
        print("=" * 50)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Date range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
        print(f"Zone: {self.df['zone'].iloc[0]}")
        print(f"Frequency: 5-minute intervals")
        
        # Check for missing values
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("✅ No missing values")
        else:
            print("⚠️ Missing values found:")
            print(missing[missing > 0])
            
        # Data ranges
        print(f"\n💰 PRICING RANGES:")
        price_cols = ['lmp', 'regulation_up_price', 'regulation_down_price', 
                      'spinning_reserve_price', 'non_sync_reserve_price']
        for col in price_cols:
            min_val = self.df[col].min()
            max_val = self.df[col].max()
            mean_val = self.df[col].mean()
            print(f"   {col}: ${min_val:.2f} - ${max_val:.2f} (avg: ${mean_val:.2f})")
            
        print(f"\n🌞 SOLAR GENERATION RANGES:")
        print(f"   Solar generation (1kWp): {self.df['solar_generation_1kwp_kw'].min():.3f} - {self.df['solar_generation_1kwp_kw'].max():.3f} kW")
        print(f"   Solar irradiance: {self.df['solar_irradiance_w_m2'].min():.1f} - {self.df['solar_irradiance_w_m2'].max():.1f} W/m²")
        print(f"   Temperature: {self.df['temperature_c'].min():.1f} - {self.df['temperature_c'].max():.1f} °C")
    
    def calculate_solar_economics(self):
        """Calculate solar generation economics"""
        if self.df is None:
            print("❌ No data loaded.")
            return None
            
        # Daily analysis
        daily_data = self.df.groupby('date').agg({
            'solar_generation_1kwp_kw': 'sum',  # Total daily generation (kW*5min)
            'lmp': 'mean'  # Average daily LMP
        })
        
        # Convert to kWh (5-minute intervals -> /12 for hourly)
        daily_data['solar_kwh'] = daily_data['solar_generation_1kwp_kw'] / 12
        daily_data['energy_value_usd'] = daily_data['solar_kwh'] * daily_data['lmp'] / 1000  # $/day
        
        print(f"\n💰 SOLAR ECONOMICS ANALYSIS")
        print("=" * 50)
        print(f"Total week generation: {daily_data['solar_kwh'].sum():.2f} kWh")
        print(f"Average daily generation: {daily_data['solar_kwh'].mean():.2f} kWh/day")
        print(f"Peak instantaneous generation: {self.df['solar_generation_1kwp_kw'].max():.4f} kW")
        
        total_value = daily_data['energy_value_usd'].sum()
        avg_daily_value = daily_data['energy_value_usd'].mean()
        
        print(f"Total week energy value: ${total_value:.3f}")
        print(f"Average daily energy value: ${avg_daily_value:.3f}")
        print(f"Annual projection: ${avg_daily_value * 365:.2f}")
        
        # Capacity factor
        capacity_factor = (daily_data['solar_kwh'].mean() * 1000) / (24 * 1000) * 100
        print(f"Capacity factor: {capacity_factor:.1f}%")
        
        return daily_data
    
    def analyze_correlations(self):
        """Analyze correlations between solar and pricing"""
        if self.df is None:
            print("❌ No data loaded.")
            return
            
        # Overall correlation
        solar_lmp_corr = self.df['lmp'].corr(self.df['solar_generation_1kwp_kw'])
        
        print(f"\n📊 CORRELATION ANALYSIS")
        print("=" * 50)
        print(f"LMP-Solar generation correlation: {solar_lmp_corr:.3f}")
        
        if solar_lmp_corr < -0.1:
            print("   ↳ Solar generation tends to reduce prices (merit order effect)")
        elif solar_lmp_corr > 0.1:
            print("   ↳ Solar generation aligns with high prices (duck curve effect)")
        else:
            print("   ↳ Weak correlation between solar and prices")
            
        # Hourly patterns
        hourly_patterns = self.df.groupby('hour').agg({
            'lmp': 'mean',
            'solar_generation_1kwp_kw': 'mean'
        })
        
        peak_solar_hour = hourly_patterns['solar_generation_1kwp_kw'].idxmax()
        peak_price_hour = hourly_patterns['lmp'].idxmax()
        
        print(f"Peak solar hour: {peak_solar_hour}:00")
        print(f"Peak price hour: {peak_price_hour}:00")
        
        return hourly_patterns
    
    def create_visualizations(self, save_plots=False):
        """Create comprehensive visualizations"""
        if self.df is None:
            print("❌ No data loaded.")
            return
            
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('NYC Energy Market Analysis with Solar Generation', fontsize=16, y=0.98)
        
        # 1. Time series: LMP and Solar
        ax1 = axes[0, 0]
        ax1_twin = ax1.twinx()
        
        # Sample every 12th point for clarity (hourly)
        sample_df = self.df.iloc[::12].copy()
        
        line1 = ax1.plot(sample_df['timestamp'], sample_df['lmp'], 
                        color='blue', alpha=0.7, linewidth=1, label='LMP')
        ax1.set_ylabel('LMP ($/MWh)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        line2 = ax1_twin.plot(sample_df['timestamp'], sample_df['solar_generation_1kwp_kw'], 
                             color='orange', alpha=0.8, linewidth=1.5, label='Solar (1kWp)')
        ax1_twin.set_ylabel('Solar Generation (kW)', color='orange')
        ax1_twin.tick_params(axis='y', labelcolor='orange')
        
        ax1.set_title('LMP vs Solar Generation')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 2. Hourly patterns
        hourly_data = self.df.groupby('hour').agg({
            'lmp': 'mean',
            'solar_generation_1kwp_kw': 'mean'
        })
        
        ax2 = axes[0, 1]
        ax2_twin = ax2.twinx()
        
        bars1 = ax2.bar(hourly_data.index - 0.2, hourly_data['lmp'], 
                       width=0.4, color='skyblue', alpha=0.8, label='Avg LMP')
        bars2 = ax2_twin.bar(hourly_data.index + 0.2, hourly_data['solar_generation_1kwp_kw'], 
                            width=0.4, color='gold', alpha=0.8, label='Avg Solar')
        
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Average LMP ($/MWh)', color='blue')
        ax2_twin.set_ylabel('Average Solar (kW)', color='orange')
        ax2.set_title('Hourly Patterns')
        ax2.grid(True, alpha=0.3)
        
        # 3. Daily generation and value
        daily_data = self.df.groupby('date').agg({
            'solar_generation_1kwp_kw': 'sum',
            'lmp': 'mean'
        })
        daily_data['solar_kwh'] = daily_data['solar_generation_1kwp_kw'] / 12
        daily_data['energy_value'] = daily_data['solar_kwh'] * daily_data['lmp'] / 1000
        
        ax3 = axes[1, 0]
        ax3_twin = ax3.twinx()
        
        bars3 = ax3.bar(range(len(daily_data)), daily_data['solar_kwh'], 
                       color='green', alpha=0.7, label='Generation (kWh)')
        line3 = ax3_twin.plot(range(len(daily_data)), daily_data['energy_value'], 
                             color='red', marker='o', linewidth=2, label='Value ($)')
        
        ax3.set_xlabel('Day')
        ax3.set_ylabel('Daily Generation (kWh)', color='green')
        ax3_twin.set_ylabel('Daily Value ($)', color='red')
        ax3.set_title('Daily Solar Generation and Value')
        ax3.set_xticks(range(len(daily_data)))
        ax3.set_xticklabels([f"Jul {i+1}" for i in range(len(daily_data))], rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. Price distribution during solar periods
        ax4 = axes[1, 1]
        
        high_solar = self.df[self.df['solar_generation_1kwp_kw'] > 0.3]['lmp']
        low_solar = self.df[self.df['solar_generation_1kwp_kw'] < 0.1]['lmp']
        
        ax4.hist(low_solar, bins=20, alpha=0.6, label='Low/No Solar', color='gray')
        ax4.hist(high_solar, bins=20, alpha=0.6, label='High Solar', color='orange')
        ax4.set_xlabel('LMP ($/MWh)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('LMP Distribution by Solar Level')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('nyc_energy_analysis.png', dpi=300, bbox_inches='tight')
            print("📊 Visualization saved as 'nyc_energy_analysis.png'")
        else:
            plt.show()
    
    def export_summary(self):
        """Export summary statistics"""
        if self.df is None:
            print("❌ No data loaded.")
            return None
            
        summary = {
            'data_summary': {
                'total_records': len(self.df),
                'date_range': f"{self.df['timestamp'].min()} to {self.df['timestamp'].max()}",
                'zone': self.df['zone'].iloc[0]
            },
            'pricing_stats': {
                'avg_lmp': self.df['lmp'].mean(),
                'min_lmp': self.df['lmp'].min(),
                'max_lmp': self.df['lmp'].max(),
                'avg_reg_up': self.df['regulation_up_price'].mean(),
                'avg_reg_down': self.df['regulation_down_price'].mean()
            },
            'solar_stats': {
                'peak_generation_kw': self.df['solar_generation_1kwp_kw'].max(),
                'avg_generation_kw': self.df['solar_generation_1kwp_kw'].mean(),
                'total_week_kwh': self.df['solar_generation_1kwp_kw'].sum() / 12,
                'avg_daily_kwh': (self.df['solar_generation_1kwp_kw'].sum() / 12) / 8
            }
        }
        
        return summary
    
    def run_complete_analysis(self):
        """Run complete analysis pipeline"""
        print("🚀 STARTING NYC ENERGY MARKET ANALYSIS")
        print("=" * 60)
        
        # Load and verify data
        if not self.load_data():
            return
            
        self.verify_data()
        
        # Economic analysis
        daily_data = self.calculate_solar_economics()
        
        # Correlation analysis
        hourly_patterns = self.analyze_correlations()
        
        # Create visualizations
        self.create_visualizations()
        
        # Summary
        summary = self.export_summary()
        
        print(f"\n✅ ANALYSIS COMPLETE!")
        print(f"📊 Dataset: {len(self.df):,} records (5-min intervals)")
        print(f"📅 Period: July 1-8, 2025")
        print(f"🏢 Zone: NYC (N.Y.C.)")
        print(f"⚡ Solar: 1kWp system generation included")
        print(f"💾 Data file: {self.data_file}")
        print(f"📈 Visualization: nyc_energy_analysis.png")
        
        return self.df, summary

def main():
    """Main execution function"""
    # Initialize analyzer
    analyzer = NYCEnergyAnalyzer()
    
    # Run complete analysis
    df, summary = analyzer.run_complete_analysis()
    
    if df is not None:
        print(f"\n📋 QUICK STATS:")
        print(f"   • LMP range: ${df['lmp'].min():.2f} - ${df['lmp'].max():.2f}/MWh")
        print(f"   • Peak solar: {df['solar_generation_1kwp_kw'].max():.3f} kW")
        print(f"   • Daily avg solar: {(df['solar_generation_1kwp_kw'].sum()/12)/8:.2f} kWh/day")
        print(f"   • Week total solar: {df['solar_generation_1kwp_kw'].sum()/12:.1f} kWh")
        
        # Show sample data
        print(f"\n📊 SAMPLE DATA (first 5 records):")
        display_cols = ['timestamp', 'lmp', 'solar_generation_1kwp_kw', 
                       'regulation_up_price', 'spinning_reserve_price']
        print(df[display_cols].head().to_string(index=False))

if __name__ == "__main__":
    main()
