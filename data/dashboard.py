import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict
import random

# Paths
DATA_DIR = Path(__file__).resolve().parent
PROFILES_FILE = DATA_DIR / "electricity_profiles.pkl"

st.set_page_config(page_title="VPP Profiles Dashboard", layout="wide")

# --- Helpers -----------------------------------------------------------------

def load_profiles() -> Dict:
    if not PROFILES_FILE.exists():
        return {}
    try:
        with open(PROFILES_FILE, 'rb') as f:
            profiles = pickle.load(f)
        return profiles
    except Exception as e:
        st.error(f"Error loading profiles: {e}")
        return {}


def save_profiles(profiles: Dict):
    with open(PROFILES_FILE, 'wb') as f:
        pickle.dump(profiles, f)


def aggregated_demand(profiles: Dict) -> np.ndarray:
    if not profiles:
        return np.zeros(2016)
    arrs = [np.array(p['demand_kw']) for p in profiles.values() if 'demand_kw' in p]
    if not arrs:
        return np.zeros(2016)
    return np.sum(arrs, axis=0)


def aggregated_generation(profiles: Dict) -> np.ndarray:
    if not profiles:
        return np.zeros(2016)
    arrs = [np.array(p.get('generation_kw', np.zeros(2016))) for p in profiles.values()]
    if not arrs:
        return np.zeros(2016)
    return np.sum(arrs, axis=0)


def compute_stats(profiles: Dict):
    n = len(profiles)
    ev_count = sum(1 for p in profiles.values() if p.get('has_EV'))
    pv_count = sum(1 for p in profiles.values() if p.get('has_PV'))
    bess_count = sum(1 for p in profiles.values() if p.get('has_BESS'))
    ev_capacity = sum(p.get('EV_cap', 0) for p in profiles.values())
    pv_capacity = sum(p.get('PV_cap', 0) for p in profiles.values())
    bess_capacity = sum(p.get('BESS_cap', 0) for p in profiles.values())
    willingness = [p.get('willingness_to_participate', 0) for p in profiles.values()]
    return {
        'n': n,
        'ev_count': ev_count,
        'pv_count': pv_count,
        'bess_count': bess_count,
        'ev_capacity': ev_capacity,
        'pv_capacity': pv_capacity,
        'bess_capacity': bess_capacity,
        'willingness': willingness
    }

# Local simple generators (fallback for custom profile creation)

def simple_generate_demand(specs: Dict) -> np.ndarray:
    intervals_per_day = 288
    total = intervals_per_day * 7
    demand = np.zeros(total)
    for day in range(7):
        start = day * intervals_per_day
        base = specs['base_daily_consumption_kwh'] / 24.0
        daily = np.full(intervals_per_day, base * 0.4)
        peak = specs['peak_consumption_start_hour']
        peak_i = peak * 12
        end_i = min(peak_i + 36, intervals_per_day)
        daily[peak_i:end_i] *= 2.5
        if specs.get('has_ev') and specs.get('ev_capacity_kwh', 0) > 0:
            if random.random() < 0.7:
                charge_start = random.randint(0, intervals_per_day - 24)
                charge_power = min(7.0, specs['ev_capacity_kwh'] / 10)
                daily[charge_start:charge_start + 24] += charge_power
        daily *= np.random.normal(1.0, 0.1, intervals_per_day)
        daily = np.maximum(daily, base * 0.2)
        demand[start:start + intervals_per_day] = daily
    return demand


def simple_generate_solar(specs: Dict, solar_profile: np.ndarray = None) -> np.ndarray:
    if not specs.get('has_solar') or specs.get('solar_capacity_kw', 0) == 0:
        return np.zeros(2016)
    # synthetic: a daytime bell across 2016 samples
    t = np.linspace(0, 2 * np.pi, 2016)
    base = 0.5 * (np.sin(t - np.pi / 2) + 1.0)
    base = np.clip(base, 0, 1)
    solar_profile = base
    return solar_profile * specs.get('solar_capacity_kw', 0)


def simple_willingness(specs: Dict) -> int:
    score = 5
    aw = specs.get('energy_awareness', 'medium')
    bonus = {'low': -2, 'medium': 0, 'high': 3}
    score += bonus.get(aw, 0)
    if specs.get('has_ev'): score += 1
    if specs.get('has_solar'): score += 1
    if specs.get('has_battery'): score += 1
    if specs.get('has_smart_appliances'): score += 1
    if specs.get('work_pattern') in ['remote_worker', 'retired']: score += 1
    if specs.get('base_daily_consumption_kwh', 0) > 40: score += 1
    if specs.get('lifestyle') == 'shift_worker': score -= 1
    return max(1, min(10, score))


# Safe rerun helper for Streamlit compatibility
def safe_rerun():
    try:
        # Newer Streamlit exposes singleton to rerun via st.session_state or st.experimental_rerun
        if hasattr(st, 'experimental_rerun'):
            st.experimental_rerun()
        else:
            # Fallback: set a flag to indicate reload requested and raise SystemExit to force reload
            st.session_state['_reload_requested'] = True
            raise SystemExit
    except Exception:
        # suppress in case rerun is not available; UI will still show created profile via session_state
        return

# --- UI ----------------------------------------------------------------------

st.title("VPP Profiles Dashboard")

tabs = st.tabs(["Overview", "Add Profile", "Market Data"])

# Overview tab
with tabs[0]:
    st.header("Portfolio Overview")
    profiles = load_profiles()
    stats = compute_stats(profiles)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Profiles", stats['n'])
    col2.metric("EVs", f"{stats['ev_count']} ({stats['ev_capacity']} kWh)")
    col3.metric("Solar PV", f"{stats['pv_count']} ({stats['pv_capacity']} kW)")
    col4.metric("BESS", f"{stats['bess_count']} ({stats['bess_capacity']} kWh)")

    st.subheader("Aggregated load (7-day, 5-min intervals)")
    agg_d = aggregated_demand(profiles)
    agg_g = aggregated_generation(profiles)

    time_index = pd.date_range(start="2025-01-01", periods=len(agg_d), freq='5min')
    df_agg = pd.DataFrame({'demand_kw': agg_d, 'generation_kw': agg_g}, index=time_index)

    st.write("**Total Portfolio Demand**")
    st.line_chart(df_agg[['demand_kw']].rename(columns={'demand_kw': 'Aggregated Demand (kW)'}))
    
    st.write("**Total Portfolio Generation**")
    st.line_chart(df_agg[['generation_kw']].rename(columns={'generation_kw': 'Aggregated Generation (kW)'}))

    st.subheader("Willingness distribution")
    if stats['willingness']:
        st.bar_chart(pd.Series(stats['willingness']).value_counts().sort_index())
    else:
        st.write("No willingness data yet.")

    st.subheader("Profile detail")
    if profiles:
        profile_ids = list(profiles.keys())
        sel = st.selectbox("Select profile", profile_ids)
        p = profiles[sel]
        st.write(p.get('description', ''))
        detail_df = pd.DataFrame({'demand_kw': p['demand_kw'], 'generation_kw': p.get('generation_kw', [0]*2016)})
        detail_time = pd.date_range(start="2025-01-01", periods=len(detail_df), freq='5min')
        detail_df.index = detail_time
        st.line_chart(detail_df)
    else:
        st.write("No profiles available.")

# Add Profile tab
with tabs[1]:
    st.header("Add a new profile")
    mode = st.radio("Creation mode", ["AI-generated (requires GEMINI API key)", "Custom (manual input)"])

    # If a new profile was just created, show it here (persist across reruns)
    last_new = st.session_state.get('last_new_profile') if 'last_new_profile' in st.session_state else None
    if last_new:
        st.success(f"Recently created profile: {last_new}")
        all_p = load_profiles()
        if last_new in all_p:
            npf = all_p[last_new]
            st.write(f"**Description:** {npf.get('description','')}")
            st.write(f"**Willingness:** {npf.get('willingness_to_participate','N/A')}/10")
            prof_df = pd.DataFrame({
                'Demand (kW)': npf['demand_kw'],
                'Generation (kW)': npf.get('generation_kw', [0]*len(npf['demand_kw']))
            })
            prof_time = pd.date_range(start="2025-01-01", periods=len(prof_df), freq='5min')
            prof_df.index = prof_time
            st.write("**New Profile Energy Pattern (7 days)**")
            st.line_chart(prof_df)
            st.write(f"**Total profiles in portfolio:** {len(all_p)}")
            if st.button("Clear recent profile highlight"):
                del st.session_state['last_new_profile']

    if mode.startswith("AI"):
        st.write("Generate a random profile using the Gemini LLM.")
        if st.button("Generate AI profile"):
            with st.spinner("Generating AI profile..."):
                try:
                    # Import ProfileGenerator from the same directory
                    import sys
                    import os
                    sys.path.insert(0, os.path.dirname(__file__))
                    from profile_gen_add import ProfileGenerator
                    gen = ProfileGenerator()
                    new_profile = gen.create_profile()
                    if new_profile:
                        profiles = load_profiles()
                        profiles[new_profile['id']] = new_profile
                        save_profiles(profiles)
                        # persist id across reruns
                        st.session_state['last_new_profile'] = new_profile['id']
                        # Show success message and profile details immediately
                        st.success(f"✅ Created {new_profile['id']}")
                        st.write(f"**Description:** {new_profile.get('description', 'No description')}")
                        st.write(f"**Willingness to participate:** {new_profile.get('willingness_to_participate', 'N/A')}/10")
                        prof_df = pd.DataFrame({
                            'Demand (kW)': new_profile['demand_kw'],
                            'Generation (kW)': new_profile.get('generation_kw', [0]*len(new_profile['demand_kw']))
                        })
                        prof_time = pd.date_range(start="2025-01-01", periods=len(prof_df), freq='5min')
                        prof_df.index = prof_time
                        st.write("**Profile Energy Pattern (7 days)**")
                        st.line_chart(prof_df)
                        updated_profiles = load_profiles()
                        updated_stats = compute_stats(updated_profiles)
                        st.write(f"**Total profiles in portfolio:** {updated_stats['n']}")
                        # rerun to refresh other tabs/controls (but we kept the recent id)
                        safe_rerun()
                    else:
                        st.error("Profile generation failed.")
                except Exception as e:
                    st.error(f"AI generation failed: {e}")
                    st.write("Make sure GEMINI_API_KEY is set in .env file")
    else:
        st.write("Create a custom profile using the form below.")
        with st.form("custom_profile_form"):
            household_type = st.selectbox("Household type", ['apartment','small_house','large_house','townhouse','penthouse','studio'])
            occupants = st.number_input("Occupants", min_value=1, max_value=6, value=2)
            lifestyle = st.selectbox("Lifestyle", ['early_riser','night_owl','9to5_worker','shift_worker','retired','student'])
            work_pattern = st.selectbox("Work pattern", ['office_worker','remote_worker','hybrid','unemployed','retired','student'])
            energy_awareness = st.selectbox("Energy awareness", ['low','medium','high'], index=1)
            base_daily = st.number_input("Daily consumption kWh", min_value=12.0, max_value=200.0, value=30.0)
            peak_hour = st.slider("Peak start hour", 6, 21, 18)
            has_ev = st.checkbox("Has EV")
            ev_capacity = st.number_input("EV capacity kWh", min_value=0.0, max_value=200.0, value=75.0)
            has_solar = st.checkbox("Has Solar PV")
            solar_capacity = st.number_input("Solar capacity kW", min_value=0.0, max_value=50.0, value=6.0)
            has_battery = st.checkbox("Has Battery Storage")
            battery_capacity = st.number_input("Battery capacity kWh", min_value=0.0, max_value=200.0, value=15.0)
            has_smart = st.checkbox("Has Smart Appliances")
            has_pool = st.checkbox("Has Pool/Spa")

            submitted = st.form_submit_button("Create profile")
            if submitted:
                new_id = f"profile_{len(profiles) + 1:03d}" if profiles else "profile_001"
                specs = {
                    'household_type': household_type,
                    'occupants': int(occupants),
                    'lifestyle': lifestyle,
                    'work_pattern': work_pattern,
                    'energy_awareness': energy_awareness,
                    'base_daily_consumption_kwh': float(base_daily),
                    'peak_consumption_start_hour': int(peak_hour),
                    'has_ev': has_ev,
                    'ev_capacity_kwh': float(ev_capacity) if has_ev else 0.0,
                    'has_solar': has_solar,
                    'solar_capacity_kw': float(solar_capacity) if has_solar else 0.0,
                    'has_battery': has_battery,
                    'battery_capacity_kwh': float(battery_capacity) if has_battery else 0.0,
                    'has_smart_appliances': has_smart,
                    'has_pool_spa': has_pool
                }
                demand = simple_generate_demand(specs)
                generation = simple_generate_solar(specs)
                willingness = simple_willingness(specs)
                description = f"{specs['household_type']} ({specs['occupants']} occupants) | {specs['lifestyle']} {specs['work_pattern']} | peak {specs['peak_consumption_start_hour']}-{specs['peak_consumption_start_hour']+3}h"
                new_profile = {
                    'id': new_id,
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
                # Save
                all_profiles = load_profiles()
                # ensure unique id
                if new_id in all_profiles:
                    # find next
                    i = len(all_profiles) + 1
                    while f"profile_{i:03d}" in all_profiles:
                        i += 1
                    new_id = f"profile_{i:03d}"
                    new_profile['id'] = new_id
                all_profiles[new_profile['id']] = new_profile
                save_profiles(all_profiles)
                st.session_state['last_new_profile'] = new_profile['id']
                st.success(f"✅ Saved {new_profile['id']}")
                st.write(f"**Description:** {description}")
                st.write(f"**Willingness to participate:** {willingness}/10")
                updated_stats = compute_stats(all_profiles)
                st.write(f"**Total profiles in portfolio:** {updated_stats['n']}")
                safe_rerun()


# Footer
st.markdown("---")
st.caption("Profiles are stored in data/electricity_profiles.pkl — backups recommended.")

# --- Market Data Tab -------------------------------------------------------
with tabs[2]:
    st.header("Market Data & Rates")
    parquet_path = DATA_DIR / 'nyc_pricing_with_solar_real_data.parquet'
    if not parquet_path.exists():
        st.error(f"Market data not found at {parquet_path}")
    else:
        try:
            mdf = pd.read_parquet(parquet_path)
            # Ensure timestamp column is datetime
            if not pd.api.types.is_datetime64_any_dtype(mdf['timestamp']):
                mdf['timestamp'] = pd.to_datetime(mdf['timestamp'])

            st.subheader("Price time series (LMP)")
            # Allow user to select date range
            min_t = mdf['timestamp'].min()
            max_t = mdf['timestamp'].max()
            # Convert to python datetime for slider
            min_dt = min_t.to_pydatetime()
            max_dt = max_t.to_pydatetime()
            default_end = min_dt + (max_dt - min_dt) / 7
            rng = st.slider("Date range (select window)",
                            min_value=min_dt, max_value=max_dt,
                            value=(min_dt, default_end))

            sel_df = mdf[(mdf['timestamp'] >= rng[0]) & (mdf['timestamp'] <= rng[1])]
            if sel_df.empty:
                st.write("No market data in selected range.")
            else:
                ts = sel_df.set_index('timestamp')
                st.line_chart(ts['lmp'].rename('LMP ($/MWh)'))

                st.subheader("Price distribution & summary")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write("**LMP Statistics**")
                    st.write(ts['lmp'].describe())
                with col_b:
                    st.write("**Reserve Price Statistics**")
                    reserve_cols = ['regulation_up_price', 'regulation_down_price', 'spinning_reserve_price', 'non_sync_reserve_price']
                    st.write(ts[reserve_cols].describe())

                st.subheader("Reserve prices time series")
                st.line_chart(ts[reserve_cols])

                st.subheader("LMP vs Solar Generation")
                solar_lmp = ts[['lmp', 'solar_generation_1kwp_kw']].copy()
                solar_lmp.columns = ['LMP ($/MWh)', 'Solar Gen (kW/kWp)']
                st.line_chart(solar_lmp)

                st.subheader("Daily average LMP")
                daily = ts['lmp'].resample('D').mean()
                # Ensure we have multiple days to show
                if len(daily) > 1:
                    st.bar_chart(daily)
                else:
                    st.write("Select a longer date range to see daily averages")
                    st.write(f"Current selection has {len(daily)} day(s)")
                    if len(daily) == 1:
                        st.metric("Average LMP", f"${daily.iloc[0]:.2f}/MWh")

        except Exception as e:
            st.error(f"Error loading market data: {e}")
