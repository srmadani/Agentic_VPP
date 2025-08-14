# VPP Prosumer Profile Generator

## Overview

This system generates realistic electricity consumption and generation profiles for residential customers in Virtual Power Plant (VPP) applications. It uses real NYC market data combined with AI-generated household characteristics to create diverse profiles for grid analysis and demand response modeling.

## Data Sources and Generation

### NYC Energy Market Data
The system uses real energy market data from New York City:

**Source Data Processing:**
- **Market Data**: NYISO real-time pricing and grid data
- **Solar Irradiance**: NYC weather station measurements 
- **Processing**: `nyc_energy_analysis.py` combines pricing with solar calculations
- **Output**: `nyc_pricing_with_solar_real_data.parquet` with 5-minute resolution data

**Key Data Columns:**
- `timestamp`: 5-minute intervals covering 7 days
- `lmp`: Locational Marginal Pricing ($/MWh)
- `solar_generation_1kwp_kw`: Solar output per kW installed
- `temperature_c`: Ambient temperature
- Grid service prices (regulation, reserves)

### Profile Generation Process
Household profiles are generated using Google's Gemini AI model via LangChain:

**AI Model Integration:**
```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.8  # For diversity
)
```

**Profile Components:**
1. **Household Characteristics**: Type, occupants, lifestyle, work patterns
2. **Energy Technologies**: EV, Solar PV, Battery storage, Smart appliances
3. **Consumption Patterns**: Base load, peak periods, technology-specific loads
4. **Solar Generation**: Scaled from real NYC irradiance data
5. **Willingness Score**: 1-10 scale for demand response participation

## File Structure

```
data/
├── nyc_energy_analysis.py              # Market data processing
├── nyc_pricing_with_solar_real_data.parquet  # Processed market data
├── electricity_profiles.pkl            # Generated household profiles  
├── profile_gen_add.py                  # Interactive profile generator
├── dashboard.py                        # Streamlit dashboard
└── README.md                           # This file
```

## Replication Steps

### 1. Environment Setup
```bash
# Required environment variable
GEMINI_API_KEY=your_google_ai_api_key

# Dependencies
pip install pandas numpy langchain-google-genai python-dotenv
```

### 2. Market Data Processing
```bash
python nyc_energy_analysis.py
```
This processes raw NYISO data and weather data to create the parquet file with:
- Real-time electricity prices
- Solar generation potential per kW
- Temperature and weather patterns
- Grid service pricing

### 3. Profile Generation
```bash
python profile_gen_add.py
```
Interactive tool that:
- Checks existing profiles
- Allows custom or AI-generated profiles
- Uses Gemini AI for realistic characteristics
- Saves profiles to pickle file

## Profile Structure

Each profile contains 12 essential keys:
```python
{
    'id': 'profile_001',                    # Unique identifier
    'description': 'text description',      # Human-readable summary
    'specifications': {...},                # Full AI-generated specs
    'demand_kw': [2016 values],            # 5-min load data (7 days)
    'generation_kw': [2016 values],        # 5-min solar data
    'has_EV': bool,                        # Electric vehicle ownership
    'EV_cap': int,                         # EV capacity (kWh)
    'has_PV': bool,                        # Solar PV ownership  
    'PV_cap': float,                       # Solar capacity (kW)
    'has_BESS': bool,                      # Battery storage ownership
    'BESS_cap': int,                       # Battery capacity (kWh)
    'willingness_to_participate': int      # DR willingness (1-10)
}
```

## Usage Examples

### Load Existing Profiles
```python
import pickle
import numpy as np

with open('electricity_profiles.pkl', 'rb') as f:
    profiles = pickle.load(f)

# Access profile data
profile = profiles['profile_001']
demand = np.array(profile['demand_kw'])      # 2016 data points
generation = np.array(profile['generation_kw'])
willingness = profile['willingness_to_participate']
```

### Add New Profiles
```bash
python profile_gen_add.py
# Follow interactive prompts to add profiles
```

### Analyze Portfolio
```python
# Technology adoption rates
total_profiles = len(profiles)
ev_adoption = sum(p['has_EV'] for p in profiles.values()) / total_profiles
solar_adoption = sum(p['has_PV'] for p in profiles.values()) / total_profiles

# Aggregate capacity
total_ev_storage = sum(p['EV_cap'] for p in profiles.values() if p['has_EV'])
total_solar_capacity = sum(p['PV_cap'] for p in profiles.values() if p['has_PV'])
```

## Dashboard (Streamlit)

A Streamlit dashboard is provided at `data/dashboard.py` for quick exploration of the profile pool and market data.

How to run
```bash
# from repository root (recommended virtualenv active)
streamlit run data/dashboard.py
```

Requirements
- Install Streamlit: `pip install streamlit`
- Optional: set `GEMINI_API_KEY` in the repository root `.env` to enable AI-generated profiles from the dashboard's "Add Profile" tab.

Tabs and functionality
- Overview: portfolio summary metrics (profiles, EV/PV/BESS counts and capacities), aggregated 7-day demand and generation charts, willingness distribution, and a per-profile detail view (select a profile to see its 7-day demand/generation time series).
- Add Profile: create new profiles interactively. Two modes:
    - AI-generated: uses `profile_gen_add.ProfileGenerator` (requires `GEMINI_API_KEY`); generated profile is saved to `data/electricity_profiles.pkl` and the UI shows the new profile summary and its 7-day pattern.
    - Custom: fill a form to create a profile locally (uses a simple demand/solar generator). New profiles are saved to `data/electricity_profiles.pkl`.
- Market Data: visualises `data/nyc_pricing_with_solar_real_data.parquet` (LMP time series, reserve prices, daily averages, and LMP vs solar generation).

Notes & troubleshooting
- Profiles are persisted in `data/electricity_profiles.pkl`. Back up this file before running bulk operations.
- If the Market Data tab shows an error, confirm `data/nyc_pricing_with_solar_real_data.parquet` exists and is readable.
- If AI generation fails, ensure `.env` at the repository root contains a valid `GEMINI_API_KEY` and that network/access to the model is available.
- The dashboard uses 5-minute resolution (2016 samples = 7 days) for plotting. Large portfolios may take longer to aggregate.

For full replication, install requirements listed in `requirements.txt` at the repository root. See `requirements.md` for venv and install instructions.

## Applications

**Virtual Power Plant Operations:**
- Resource assessment and aggregation
- Demand response program design
- Grid service provision (frequency regulation, reserves)
- Load forecasting and planning

**Market Analysis:**
- Customer segmentation by technology adoption
- Revenue potential estimation
- Program participation modeling

## Technical Specifications

- **Time Resolution**: 5-minute intervals
- **Duration**: 7 days (2016 data points per profile)
- **AI Model**: Google Gemini-2.5-Flash via LangChain
- **Data Format**: Python pickle files
- **Market Data**: Real NYC NYISO pricing and solar irradiance
- **Profile Diversity**: Household types, technology adoption, consumption patterns

## Generated Profile Statistics

Typical portfolio characteristics:
- **Technology Adoption**: 40-60% EV, 30-50% Solar, 20-40% Battery
- **Willingness Score**: Average 7-9/10 for demand response
- **Daily Consumption**: 15-60 kWh range with realistic patterns
- **Flexible Storage**: 50-200 kWh per 100 profiles for grid services
