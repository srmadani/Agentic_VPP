#!/usr/bin/env python3
"""
VPP Profile Generator - Add New Profiles
Interactive tool for generating electricity demand/generation profiles using Gemini AI
"""

import pandas as pd
import numpy as np
import json
import random
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pickle

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from pathlib import Path

# Load .env from the repository root (one level above this script)
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(str(env_path))

# Data directory (script location)
DATA_DIR = Path(__file__).resolve().parent

class ProfileGenerator:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=self.api_key,
            temperature=0.8
        )
        
        self.solar_data = self._load_solar_data()
        # store profiles file path inside data dir
        self.profiles_file = str(DATA_DIR / "electricity_profiles.pkl")
        self.profiles = self._load_existing_profiles()
        
    def _load_solar_data(self) -> pd.DataFrame:
        try:
            parquet_path = DATA_DIR / 'nyc_pricing_with_solar_real_data.parquet'
            df = pd.read_parquet(parquet_path)
            return df[['timestamp', 'solar_generation_1kwp_kw']].copy()
        except Exception as e:
            print(f"Error loading solar data: {e}")
            return pd.DataFrame()

    def _load_existing_profiles(self) -> Dict:
        try:
            with open(self.profiles_file, 'rb') as f:
                profiles = pickle.load(f)
            print(f"📊 Found {len(profiles)} existing profiles in {self.profiles_file}")
            return profiles
        except FileNotFoundError:
            print(f"📊 No existing profiles found. Starting fresh.")
            return {}
    
    def generate_profile_specs(self) -> Dict:
        system_prompt = """You are an expert in residential energy consumption patterns. Generate realistic household characteristics for electricity demand modeling."""
        
        human_prompt = """Create a realistic household profile with these specifications in valid JSON format:

{
    "household_type": "apartment|small_house|large_house|townhouse|penthouse|studio",
    "occupants": 1-6,
    "lifestyle": "early_riser|night_owl|9to5_worker|shift_worker|retired|student",
    "work_pattern": "office_worker|remote_worker|hybrid|unemployed|retired|student",
    "energy_awareness": "low|medium|high",
    "base_daily_consumption_kwh": 12-65,
    "peak_consumption_start_hour": 6-21,
    "has_ev": true/false,
    "ev_capacity_kwh": 0 or 40-150,
    "has_solar": true/false,
    "solar_capacity_kw": 0 or 2-15,
    "has_battery": true/false,
    "battery_capacity_kwh": 0 or 5-40,
    "has_smart_appliances": true/false,
    "has_pool_spa": true/false
}"""

        try:
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
            response = self.llm.invoke(messages)
            response_text = response.content.strip()
            
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            specs = json.loads(response_text.strip())
            return specs
            
        except Exception as e:
            print(f"Error generating specs: {e}")
            return None
    
    def generate_demand_pattern(self, specs: Dict) -> np.ndarray:
        intervals_per_day = 288  # 5-minute intervals
        total_intervals = intervals_per_day * 7  # 7 days
        
        demand = np.zeros(total_intervals)
        
        for day in range(7):
            start_idx = day * intervals_per_day
            end_idx = start_idx + intervals_per_day
            
            base_consumption = specs['base_daily_consumption_kwh']
            avg_power = base_consumption / 24.0  # Average hourly consumption in kW
            
            daily_pattern = np.full(intervals_per_day, avg_power * 0.4)  # Base load
            
            # Peak consumption period (3 hours)
            peak_start = specs['peak_consumption_start_hour']
            peak_start_interval = peak_start * 12  # Convert to 5-min intervals
            peak_end_interval = peak_start_interval + 36  # 3 hours
            
            if peak_end_interval <= intervals_per_day:
                daily_pattern[peak_start_interval:peak_end_interval] *= 2.5
            
            # Add EV charging if present
            if specs['has_ev'] and specs['ev_capacity_kwh'] > 0:
                if random.random() < 0.7:  # 70% chance of charging per day
                    charge_start = random.randint(0, intervals_per_day - 24)  # 2-hour charging
                    charge_power = min(7.0, specs['ev_capacity_kwh'] / 10)  # Reasonable charging power
                    daily_pattern[charge_start:charge_start + 24] += charge_power
            
            # Add random variations
            daily_pattern *= np.random.normal(1.0, 0.1, intervals_per_day)
            daily_pattern = np.maximum(daily_pattern, avg_power * 0.2)  # Minimum load
            
            demand[start_idx:end_idx] = daily_pattern
        
        return demand
    
    def generate_solar_generation(self, specs: Dict) -> np.ndarray:
        if not specs['has_solar'] or specs['solar_capacity_kw'] == 0:
            return np.zeros(2016)
        
        solar_1kwp = self.solar_data['solar_generation_1kwp_kw'].head(2016).values
        solar_capacity = specs['solar_capacity_kw']
        
        # Add realistic variations
        variation = np.random.normal(1.0, 0.1, len(solar_1kwp))
        variation = np.clip(variation, 0.6, 1.2)
        
        generation = solar_1kwp * solar_capacity * variation
        return np.maximum(generation, 0)
    
    def calculate_willingness_score(self, specs: Dict) -> int:
        score = 5  # Base score
        
        # Energy awareness bonus
        awareness_bonus = {'low': -2, 'medium': 0, 'high': 3}
        score += awareness_bonus[specs['energy_awareness']]
        
        # Technology adoption bonus
        if specs['has_ev']:
            score += 1
        if specs['has_solar']:
            score += 1
        if specs['has_battery']:
            score += 1
        if specs['has_smart_appliances']:
            score += 1
        
        # Lifestyle adjustments
        if specs['work_pattern'] in ['remote_worker', 'retired']:
            score += 1
        if specs['base_daily_consumption_kwh'] > 40:
            score += 1
        if specs['lifestyle'] == 'shift_worker':
            score -= 1
        
        return max(1, min(10, score))
    
    def create_profile_description(self, specs: Dict) -> str:
        house_type = specs['household_type']
        occupants = specs['occupants']
        lifestyle = specs['lifestyle']
        work = specs['work_pattern']
        peak_hour = specs['peak_consumption_start_hour']
        consumption = specs['base_daily_consumption_kwh']
        awareness = specs['energy_awareness']
        
        technologies = []
        if specs['has_ev']:
            technologies.append(f"EV({specs['ev_capacity_kwh']}kWh)")
        if specs['has_solar']:
            technologies.append(f"Solar({specs['solar_capacity_kw']}kW)")
        if specs['has_battery']:
            technologies.append(f"Battery({specs['battery_capacity_kwh']}kWh)")
        if specs['has_smart_appliances']:
            technologies.append("Smart")
        if specs['has_pool_spa']:
            technologies.append("Pool/Spa")
        
        tech_str = " + ".join(technologies) if technologies else "No tech"
        
        return f"{house_type} ({occupants} occupants) | {lifestyle} {work} | peak {peak_hour}-{peak_hour+3}h | {tech_str} | {consumption}kWh/day ({awareness} awareness)"
    
    def create_profile(self, user_input: Dict = None) -> Dict:
        if user_input:
            specs = user_input
        else:
            specs = self.generate_profile_specs()
            if not specs:
                return None
        
        # Generate data
        demand = self.generate_demand_pattern(specs)
        generation = self.generate_solar_generation(specs)
        willingness = self.calculate_willingness_score(specs)
        description = self.create_profile_description(specs)
        
        # Get next profile ID
        next_id = len(self.profiles) + 1
        profile_id = f"profile_{next_id:03d}"
        
        profile = {
            'id': profile_id,
            'description': description,
            'specifications': specs,
            'demand_kw': demand.tolist(),
            'generation_kw': generation.tolist(),
            'has_EV': specs['has_ev'],
            'EV_cap': specs['ev_capacity_kwh'],
            'has_PV': specs['has_solar'],
            'PV_cap': specs['solar_capacity_kw'],
            'has_BESS': specs['has_battery'],
            'BESS_cap': specs['battery_capacity_kwh'],
            'willingness_to_participate': willingness
        }
        
        return profile
    
    def save_profiles(self):
        with open(self.profiles_file, 'wb') as f:
            pickle.dump(self.profiles, f)
        print(f"💾 Saved {len(self.profiles)} profiles to {self.profiles_file}")
    
    def show_portfolio_stats(self):
        if not self.profiles:
            print("No profiles to analyze.")
            return
        
        total = len(self.profiles)
        
        # Technology adoption
        ev_count = sum(1 for p in self.profiles.values() if p['has_EV'])
        solar_count = sum(1 for p in self.profiles.values() if p['has_PV'])
        battery_count = sum(1 for p in self.profiles.values() if p['has_BESS'])
        
        # Willingness scores
        willingness_scores = [p['willingness_to_participate'] for p in self.profiles.values()]
        
        # Capacities
        total_ev_capacity = sum(p['EV_cap'] for p in self.profiles.values() if p['has_EV'])
        total_solar_capacity = sum(p['PV_cap'] for p in self.profiles.values() if p['has_PV'])
        total_battery_capacity = sum(p['BESS_cap'] for p in self.profiles.values() if p['has_BESS'])
        
        print(f"\n📊 Portfolio Statistics ({total} profiles)")
        print("=" * 50)
        print(f"Technology Adoption:")
        print(f"  • Electric Vehicles: {ev_count}/{total} ({ev_count/total*100:.1f}%)")
        print(f"  • Solar PV: {solar_count}/{total} ({solar_count/total*100:.1f}%)")
        print(f"  • Battery Storage: {battery_count}/{total} ({battery_count/total*100:.1f}%)")
        
        print(f"\nWillingness to Participate:")
        print(f"  • Range: {min(willingness_scores)}-{max(willingness_scores)}/10")
        print(f"  • Average: {sum(willingness_scores)/len(willingness_scores):.1f}/10")
        print(f"  • High willingness (8-10): {len([s for s in willingness_scores if s >= 8])}/{total}")
        
        print(f"\nTotal Portfolio Capacity:")
        print(f"  • EV Storage: {total_ev_capacity} kWh ({ev_count} vehicles)")
        print(f"  • Solar Generation: {total_solar_capacity} kW ({solar_count} systems)")
        print(f"  • Battery Storage: {total_battery_capacity} kWh ({battery_count} systems)")
        print(f"  • Flexible Storage: {total_ev_capacity + total_battery_capacity} kWh")

def get_user_input():
    print("\n🏠 Create Custom Profile")
    print("Enter household details (press Enter for random generation):")
    
    # Validate household type
    valid_household_types = ['apartment', 'small_house', 'large_house', 'townhouse', 'penthouse', 'studio']
    household_type = input("Household type (apartment/small_house/large_house/townhouse/penthouse/studio): ").strip().lower()
    if not household_type:
        return None
    if household_type not in valid_household_types:
        print(f"Invalid household type '{household_type}'. Using 'apartment' as default.")
        household_type = 'apartment'
    
    try:
        occupants = int(input("Number of occupants (1-6): ") or "0")
        if not 1 <= occupants <= 6:
            occupants = random.randint(1, 6)
    except:
        occupants = random.randint(1, 6)
    
    lifestyle = input("Lifestyle (early_riser/night_owl/9to5_worker/shift_worker/retired/student): ").strip()
    work_pattern = input("Work pattern (office_worker/remote_worker/hybrid/unemployed/retired/student): ").strip()
    
    # Validate energy awareness input
    energy_awareness = input("Energy awareness (low/medium/high): ").strip().lower()
    if energy_awareness not in ['low', 'medium', 'high']:
        print(f"Invalid energy awareness '{energy_awareness}'. Using 'medium' as default.")
        energy_awareness = 'medium'
    
    try:
        consumption = float(input("Daily consumption in kWh (12-65): ") or "30")
        peak_hour = int(input("Peak consumption start hour (6-21): ") or "18")
    except:
        consumption = 30
        peak_hour = 18
    
    has_ev = input("Has Electric Vehicle? (y/n): ").lower().startswith('y')
    ev_capacity = 0
    if has_ev:
        try:
            ev_capacity = int(input("EV capacity in kWh (40-150): ") or "75")
        except:
            ev_capacity = 75
    
    has_solar = input("Has Solar PV? (y/n): ").lower().startswith('y')
    solar_capacity = 0
    if has_solar:
        try:
            solar_capacity = float(input("Solar capacity in kW (2-15): ") or "6")
        except:
            solar_capacity = 6
    
    has_battery = input("Has Battery Storage? (y/n): ").lower().startswith('y')
    battery_capacity = 0
    if has_battery:
        try:
            battery_capacity = int(input("Battery capacity in kWh (5-40): ") or "15")
        except:
            battery_capacity = 15
    
    has_smart = input("Has Smart Appliances? (y/n): ").lower().startswith('y')
    has_pool = input("Has Pool/Spa? (y/n): ").lower().startswith('y')
    
    return {
        'household_type': household_type or 'apartment',
        'occupants': occupants,
        'lifestyle': lifestyle or 'night_owl',
        'work_pattern': work_pattern or 'remote_worker',
        'energy_awareness': energy_awareness or 'medium',
        'base_daily_consumption_kwh': consumption,
        'peak_consumption_start_hour': peak_hour,
        'has_ev': has_ev,
        'ev_capacity_kwh': ev_capacity,
        'has_solar': has_solar,
        'solar_capacity_kw': solar_capacity,
        'has_battery': has_battery,
        'battery_capacity_kwh': battery_capacity,
        'has_smart_appliances': has_smart,
        'has_pool_spa': has_pool
    }

def main():
    print("🏠 VPP Profile Generator")
    print("=" * 30)
    
    try:
        generator = ProfileGenerator()
        
        while True:
            if generator.profiles:
                print(f"\nCurrent profiles: {len(generator.profiles)}")
            
            add_profile = input("\nAdd new profile? (Y/N): ").strip().upper()
            if add_profile != 'Y':
                break
            
            print("\nProfile generation options:")
            print("1. Random AI-generated profile")
            print("2. Custom profile with your inputs")
            
            choice = input("Choose option (1/2): ").strip()
            
            if choice == '2':
                user_input = get_user_input()
                if not user_input:
                    print("Using random generation...")
                    user_input = None
            else:
                user_input = None
            
            print("\n🤖 Generating profile with Gemini AI...")
            profile = generator.create_profile(user_input)
            
            if profile:
                generator.profiles[profile['id']] = profile
                print(f"✅ Created {profile['id']}: {profile['description']}")
                print(f"   Willingness: {profile['willingness_to_participate']}/10")
                
                generator.save_profiles()
                time.sleep(2)  # Brief pause between requests
            else:
                print("❌ Failed to generate profile")
        
        generator.show_portfolio_stats()
        
    except KeyboardInterrupt:
        print("\n\nProfile generation interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
