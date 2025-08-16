"""
Streamlit Dashboard for VPP Multi-Agent Simulation
Visualizes the negotiation process between Aggregator and Prosumer agents
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import pickle
import io
import sys
from contextlib import redirect_stdout, redirect_stderr
from vpp_simulation import run_vpp_simulation, calculate_dissatisfaction, calculate_aggregator_profit

st.set_page_config(
    page_title="VPP Multi-Agent Simulation",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

def save_simulation_results(results, filename="simulation_results.pkl"):
    """Save simulation results to file"""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        return True
    except Exception as e:
        st.error(f"Failed to save results: {e}")
        return False

def load_simulation_results(filename="simulation_results.pkl"):
    """Load simulation results from file"""
    try:
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                results = pickle.load(f)
            return results
        return None
    except Exception as e:
        st.error(f"Failed to load results: {e}")
        return None

class StreamCapture:
    """Custom stream capture that writes to both original stream and buffer"""
    def __init__(self, original_stream, buffer):
        self.original_stream = original_stream
        self.buffer = buffer
    
    def write(self, text):
        self.original_stream.write(text)
        self.buffer.write(text)
        self.original_stream.flush()
    
    def flush(self):
        self.original_stream.flush()
        self.buffer.flush()

def load_data():
    """Load the base data for visualization"""
    pkl_path = "data/electricity_profiles.pkl"
    parquet_path = "data/nyc_pricing_with_solar_real_data.parquet"
    
    electricity_profiles = pd.read_pickle(pkl_path)
    pricing_nyc = pd.read_parquet(parquet_path)
    
    # Filter for first day
    first_day = pricing_nyc['date'].min()
    pricing_nyc_day1 = pricing_nyc[pricing_nyc['date'] == first_day].copy()
    
    return electricity_profiles, pricing_nyc_day1

def plot_lmp_data(pricing_data):
    """Plot LMP data for the day"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=pricing_data['timestamp'],
        y=pricing_data['lmp'],
        mode='lines',
        name='LMP',
        line=dict(color='green', width=2),
        hovertemplate='<b>LMP</b><br>Time: %{x}<br>$/MWh: %{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Locational Marginal Price (LMP)",
        xaxis_title="Time",
        yaxis_title="LMP ($/MWh)",
        height=300,
        hovermode='x unified'
    )
    
    return fig

def plot_individual_prosumer_profile(profile, prosumer_id):
    """Plot individual prosumer profile with demand, generation, and net load"""
    
    time_intervals = pd.date_range('2025-07-01', periods=288, freq='5min')
    demand = profile['demand_kw'][:288]
    generation = profile['generation_kw'][:288]
    net_load = [d - g for d, g in zip(demand, generation)]
    
    fig = go.Figure()
    
    # Demand line
    fig.add_trace(go.Scatter(
        x=time_intervals,
        y=demand,
        mode='lines',
        name='Demand',
        line=dict(color='red', width=2),
        hovertemplate='<b>Demand</b><br>Time: %{x}<br>kW: %{y:.2f}<extra></extra>'
    ))
    
    # Generation line
    fig.add_trace(go.Scatter(
        x=time_intervals,
        y=generation,
        mode='lines',
        name='Generation',
        line=dict(color='orange', width=2),
        hovertemplate='<b>Generation</b><br>Time: %{x}<br>kW: %{y:.2f}<extra></extra>'
    ))
    
    # Net load as filled area
    fig.add_trace(go.Scatter(
        x=time_intervals,
        y=net_load,
        mode='lines',
        name='Net Load (Demand - Generation)',
        line=dict(color='blue', width=1),
        fill='tozeroy',
        fillcolor='rgba(0,0,255,0.2)',
        hovertemplate='<b>Net Load</b><br>Time: %{x}<br>kW: %{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Prosumer {prosumer_id} - Daily Profile",
        xaxis_title="Time",
        yaxis_title="Power (kW)",
        height=300,
        hovermode='x unified'
    )
    
    return fig

def display_negotiation_messages(simulation_log):
    """Display the negotiation messages between agents"""
    
    for log_entry in simulation_log:
        step = log_entry['step']
        round_num = log_entry['round']
        data = log_entry['data']
        timestamp = log_entry['timestamp']
        
        st.write(f"**{timestamp.strftime('%H:%M:%S')} - Round {round_num}: {step.replace('_', ' ').title()}**")
        
        if 'aggregator' in step:
            st.info("🏢 **Aggregator Message:**")
            if 'message' in str(data):
                for prosumer_id, offer in data.items():
                    if isinstance(offer, dict) and 'message' in offer:
                        st.write(f"*To {prosumer_id}:*")
                        st.write(offer['message'])
                        if 'offered_rates' in offer:
                            st.write(f"Offered rates: {[f'{r:.3f}' for r in offer['offered_rates'][:3]]}... $/kWh")
        
        elif 'prosumer' in step:
            st.success("🏠 **Prosumer Response:**")
            if 'counter_message' in data:
                st.write(data['counter_message'])
            elif 'final_message' in data:
                st.write(data['final_message'])
            
            if 'proposed_commitments' in data:
                st.write(f"Proposed commitments: {[f'{c:.2f}' for c in data['proposed_commitments'][:3]]}... kWh")
            if 'final_commitments' in data:
                st.write(f"Final commitments: {[f'{c:.2f}' for c in data['final_commitments'][:3]]}... kWh")
        
        st.write("---")

def plot_dissatisfaction_analysis(profiles, selected_prosumers, prosumer_responses):
    """Plot prosumer dissatisfaction for each time slot"""
    
    if not prosumer_responses:
        st.warning("No prosumer responses available yet.")
        return None
    
    fig = make_subplots(
        rows=len(selected_prosumers), cols=1,
        subplot_titles=[f"Dissatisfaction Analysis - {pid}" for pid in selected_prosumers],
        shared_xaxes=True,
        vertical_spacing=0.15
    )
    
    for i, prosumer_id in enumerate(selected_prosumers):
        if prosumer_id not in prosumer_responses:
            continue
            
        response = prosumer_responses[prosumer_id]
        profile = profiles[prosumer_id]
        
        if 'proposed_commitments' in response and 'net_demands' in response:
            commitments = response['proposed_commitments']
            demands = response['net_demands']
            
            dissatisfactions = []
            for commitment, demand in zip(commitments, demands):
                if demand > 0:
                    dissatisfaction = calculate_dissatisfaction.invoke({
                        "prosumer_id": prosumer_id,
                        "commitment_kw": commitment,
                        "demand_kw": demand,
                        "profiles": profiles
                    })
                    dissatisfactions.append(dissatisfaction)
                else:
                    dissatisfactions.append(0)
            
            time_slots = list(range(len(dissatisfactions)))
            
            fig.add_trace(
                go.Bar(
                    x=time_slots,
                    y=dissatisfactions,
                    name=f"{prosumer_id} Dissatisfaction",
                    hovertemplate='<b>Slot %{x}</b><br>Dissatisfaction: %{y:.3f}<extra></extra>'
                ),
                row=i+1, col=1
            )
    
    fig.update_layout(
        title="Prosumer Dissatisfaction by Time Slot",
        height=300 * len(selected_prosumers),
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Time Slot", row=len(selected_prosumers), col=1)
    fig.update_yaxes(title_text="Dissatisfaction Score")
    
    return fig

def calculate_simulation_results(final_state):
    """Calculate final results and profits"""
    
    results = {
        'aggregator_profit': 0,
        'prosumer_profits': {},
        'prosumer_dissatisfactions': {},
        'total_committed_kw': 0,
        'market_participation': {}
    }
    
    if not final_state.get('prosumer_responses'):
        return results
    
    dr_opportunities = final_state.get('dr_opportunities', [])
    
    for prosumer_id, response in final_state['prosumer_responses'].items():
        prosumer_profit = 0
        total_dissatisfaction = 0
        committed_kw = 0
        
        if 'final_commitments' in response and 'accepted_rates' in response:
            commitments = response['final_commitments']
            rates = response['accepted_rates']
            demands = response.get('net_demands', [0] * len(commitments))
            
            for i, (commitment, rate, demand) in enumerate(zip(commitments, rates, demands)):
                if i < len(dr_opportunities):
                    market_rate_mwh = dr_opportunities[i]['lmp']  # $/MWh
                    market_rate_kwh = market_rate_mwh / 1000.0     # Convert to $/kWh
                    
                    # Prosumer profit (what they earn)
                    prosumer_profit += commitment * rate
                    
                    # Aggregator profit (market rate - paid rate) - both in $/kWh now
                    aggregator_profit_slot = (market_rate_kwh - rate) * commitment
                    results['aggregator_profit'] += aggregator_profit_slot
                    
                    # Dissatisfaction - call function correctly
                    if demand > 0:
                        # Create a temporary state for dissatisfaction calculation
                        temp_state = {'prosumer_profiles': final_state['prosumer_profiles']}
                        from vpp_simulation import set_current_state, calculate_dissatisfaction
                        set_current_state(temp_state)
                        
                        dissatisfaction = calculate_dissatisfaction.invoke({
                            "prosumer_id": prosumer_id,
                            "commitment_kw": commitment,
                            "demand_kw": demand
                        })
                        total_dissatisfaction += dissatisfaction
                    
                    committed_kw += commitment
        
        results['prosumer_profits'][prosumer_id] = prosumer_profit
        results['prosumer_dissatisfactions'][prosumer_id] = total_dissatisfaction
        results['total_committed_kw'] += committed_kw
        results['market_participation'][prosumer_id] = committed_kw
    
    return results

def main():
    st.title("⚡ Virtual Power Plant Multi-Agent Simulation")
    st.markdown("---")
    
    # Check API keys
    gemini_pro_key = os.getenv("GEMINI_PRO_API_KEY")
    gemini_flash_key = os.getenv("GEMINI_FLASH_API_KEY")
    
    if not gemini_pro_key or not gemini_flash_key or "YOUR_" in gemini_pro_key or "YOUR_" in gemini_flash_key:
        st.error("🔑 Please set your actual GEMINI API keys in the .env file")
        st.info("Edit the .env file and replace 'YOUR_GEMINI_PRO_API_KEY_HERE' and 'YOUR_GEMINI_FLASH_API_KEY_HERE' with your actual API keys")
        st.stop()
    
    # Load base data
    try:
        profiles, pricing_data = load_data()
        selected_prosumers = ['profile_001', 'profile_002']
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    
    # Sidebar controls
    st.sidebar.title("Simulation Controls")
    
    # Load results option
    if st.sidebar.button("� Load Saved Results", type="secondary"):
        saved_results = load_simulation_results()
        if saved_results:
            st.session_state.simulation_result = saved_results
            st.sidebar.success("✅ Results loaded successfully!")
        else:
            st.sidebar.warning("No saved results found.")
    
    if st.sidebar.button("�🚀 Run VPP Simulation", type="primary"):
        # Create a progress container in the main area
        progress_container = st.container()

        with progress_container:
            st.subheader("🔄 Simulation Progress")
            progress_placeholder = st.empty()
            log_placeholder = st.container()

        with st.spinner("Running multi-agent simulation... This may take a few minutes."):
            try:
                progress_placeholder.info("🔄 Starting VPP simulation...")

                import threading
                import queue
                import time

                q = queue.Queue()
                original_stdout = sys.stdout
                original_stderr = sys.stderr

                class QueueWriter:
                    def __init__(self, orig, q):
                        self.orig = orig
                        self.q = q

                    def write(self, text):
                        try:
                            self.orig.write(text)
                            self.orig.flush()
                        except Exception:
                            pass
                        try:
                            # Put small chunks to preserve responsiveness
                            self.q.put(text)
                        except Exception:
                            pass

                    def flush(self):
                        try:
                            self.orig.flush()
                        except Exception:
                            pass

                sys.stdout = QueueWriter(original_stdout, q)
                sys.stderr = QueueWriter(original_stderr, q)

                result_container = {}
                exception_container = {}

                def target():
                    try:
                        res = run_vpp_simulation(first_day_only=True)
                        result_container['result'] = res
                        # Save results automatically
                        try:
                            save_simulation_results(res)
                            q.put('\n[SIMULATION] Results saved to file.\n')
                        except Exception as e:
                            q.put(f"\n[SIMULATION] Failed to save results: {e}\n")
                        q.put('\n[SIMULATION] Completed successfully.\n')
                    except Exception as e:
                        import traceback
                        q.put(f"\n[SIMULATION ERROR] {e}\n")
                        q.put(traceback.format_exc())
                        exception_container['error'] = e
                    finally:
                        # Ensure streams restored in case some prints used after
                        sys.stdout = original_stdout
                        sys.stderr = original_stderr

                thread = threading.Thread(target=target, daemon=True)
                thread.start()

                log_area = st.empty()
                log_text = ""

                # Poll queue and update dashboard while thread runs
                while thread.is_alive() or not q.empty():
                    try:
                        chunk = q.get(timeout=0.2)
                        if chunk is None:
                            continue
                        log_text += chunk
                        # Keep log area limited in size to avoid huge DOM
                        if len(log_text) > 20000:
                            log_text = log_text[-20000:]
                        log_area.text(log_text)
                    except queue.Empty:
                        # still alive, wait a bit
                        time.sleep(0.1)

                # Final drain
                while not q.empty():
                    try:
                        chunk = q.get_nowait()
                        log_text += chunk
                    except Exception:
                        break

                log_area.text(log_text)

                # Persist the runtime system log so it can be shown in a separate tab or expander
                try:
                    st.session_state.simulation_log_text = log_text
                except Exception:
                    pass

                # Clear the inline log area now that the log is persisted to the System Log tab
                try:
                    log_area.empty()
                except Exception:
                    pass

                # Check result
                if 'result' in result_container:
                    st.session_state.simulation_result = result_container['result']
                    progress_placeholder.success("✅ Simulation completed!")
                    st.sidebar.success("✅ Simulation completed!")
                else:
                    progress_placeholder.error("❌ Simulation failed. See logs above.")
                    st.sidebar.error("❌ Simulation failed.")
                    st.stop()

            except Exception as e:
                # Restore streams in case of unexpected error
                sys.stdout = original_stdout
                sys.stderr = original_stderr

                progress_placeholder.error(f"❌ Simulation failed: {str(e)}")
                st.sidebar.error(f"❌ Simulation failed: {e}")
                st.error(f"Detailed error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                st.stop()
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Initial Data", 
        "💬 Negotiation Messages", 
        "🏢 Aggregator Strategy",
        "💰 Final Results",
        "📝 System Log"
    ])
    
    # Tab 1: Initial Data
    with tab1:
        st.header("Initial Market and Prosumer Data")
        
        # LMP plot first
        st.subheader("Locational Marginal Price (LMP)")
        fig_lmp = plot_lmp_data(pricing_data)
        st.plotly_chart(fig_lmp, use_container_width=True)
        
        # Individual prosumer plots
        for prosumer_id in selected_prosumers:
            st.subheader(f"Prosumer {prosumer_id} Daily Profile")
            profile = profiles[prosumer_id]
            fig_profile = plot_individual_prosumer_profile(profile, prosumer_id)
            st.plotly_chart(fig_profile, use_container_width=True)
        
        # Show prosumer details
        st.subheader("Prosumer Details")
        col1, col2 = st.columns(2)
        
        for i, prosumer_id in enumerate(selected_prosumers):
            with col1 if i == 0 else col2:
                profile = profiles[prosumer_id]
                st.write(f"**{prosumer_id}:**")
                st.write(f"**Description:** {profile.get('description', 'N/A')}")
                st.write(f"**Willingness to Participate:** {profile['willingness_to_participate']}/10")
                st.write(f"**Has EV:** {profile['has_EV']} ({profile.get('EV_cap', 0)} kWh)")
                st.write(f"**Has Solar:** {profile['has_PV']} ({profile.get('PV_cap', 0)} kW)")
                st.write(f"**Has Battery:** {profile['has_BESS']} ({profile.get('BESS_cap', 0)} kWh)")
    
    # Check if simulation has been run
    if 'simulation_result' not in st.session_state:
        for tab in [tab2, tab3, tab4]:
            with tab:
                st.info("👆 Please run the simulation first using the button in the sidebar.")
        return
    
    final_state = st.session_state.simulation_result
    
    # Tab 2: Negotiation Messages
    with tab2:
        st.header("💬 Negotiation Flow")
        st.markdown("Complete conversation between Aggregator and Prosumer agents")
        
        if final_state.get('simulation_log'):
            display_negotiation_messages(final_state['simulation_log'])
        else:
            st.warning("No negotiation messages found.")
    
    # Tab 3: Aggregator Strategy
    with tab3:
        st.header("🏢 Aggregator Strategy")
        
        if final_state.get('negotiation_offers'):
            # For each prosumer, show their specific DR opportunities
            for prosumer_id in selected_prosumers:
                st.subheader(f"DR Opportunities for {prosumer_id}")
                
                profile = profiles[prosumer_id]
                offer = final_state['negotiation_offers'].get(prosumer_id, {})
                response = final_state.get('prosumer_responses', {}).get(prosumer_id, {})
                
                # Check if DR opportunities exist in the offer or at state level
                dr_opportunities = offer.get('dr_opportunities', final_state.get('dr_opportunities', []))
                
                if dr_opportunities:
                    # Plot load, LMP, and DR opportunities
                    fig_strategy = go.Figure()
                    
                    # Get time slots and prosumer data
                    time_intervals = pd.date_range('2025-07-01', periods=288, freq='5min')
                    demand = profile['demand_kw'][:288]
                    generation = profile['generation_kw'][:288]
                    net_load = [d - g for d, g in zip(demand, generation)]
                    
                    # Add net load line
                    fig_strategy.add_trace(go.Scatter(
                        x=time_intervals,
                        y=net_load,
                        mode='lines',
                        name='Net Load',
                        line=dict(color='blue', width=2),
                        yaxis='y',
                        hovertemplate='<b>Net Load</b><br>Time: %{x}<br>kW: %{y:.2f}<extra></extra>'
                    ))
                    
                    # Add LMP line on secondary y-axis
                    fig_strategy.add_trace(go.Scatter(
                        x=pricing_data['timestamp'],
                        y=pricing_data['lmp'],
                        mode='lines',
                        name='LMP',
                        line=dict(color='green', width=1),
                        yaxis='y2',
                        hovertemplate='<b>LMP</b><br>Time: %{x}<br>$/MWh: %{y:.2f}<extra></extra>'
                    ))
                    
                    # Fill area under the LMP curve for each DR hour
                    # Color the hour red if accepted by prosumer, blue if not
                    for i, opp in enumerate(dr_opportunities):
                        hour = opp['hour']
                        hour_start = pd.to_datetime(f'2025-07-01 {hour:02d}:00:00')
                        hour_end = hour_start + pd.Timedelta(hours=1)

                        # Create mask for the full hour (all 5-minute intervals)
                        mask = (pricing_data['timestamp'] >= hour_start) & (pricing_data['timestamp'] < hour_end)
                        
                        if not mask.any():
                            continue

                        # Get all timestamps and LMP values for this hour
                        x_seg = pricing_data.loc[mask, 'timestamp'].tolist()
                        y_seg = pricing_data.loc[mask, 'lmp'].tolist()
                        
                        # Ensure we have a complete hour by adding endpoints if needed
                        if len(x_seg) > 0:
                            # Add start point if not present
                            if x_seg[0] != hour_start:
                                x_seg.insert(0, hour_start)
                                y_seg.insert(0, y_seg[0] if y_seg else 0)
                            
                            # Add end point if not present
                            if x_seg[-1] < hour_end - pd.Timedelta(minutes=5):
                                x_seg.append(hour_end - pd.Timedelta(minutes=5))
                                y_seg.append(y_seg[-1] if y_seg else 0)

                        accepted = False
                        if response.get('final_participation') and i < len(response['final_participation']):
                            accepted = bool(response['final_participation'][i])

                        fillcolor = 'rgba(255,0,0,0.25)' if accepted else 'rgba(0,0,255,0.15)'
                        line_color = 'red' if accepted else 'blue'

                        fig_strategy.add_trace(go.Scatter(
                            x=x_seg,
                            y=y_seg,
                            mode='lines',
                            name=f"DR Hour {hour:02d}:00 {'(accepted)' if accepted else '(proposed)'}",
                            line=dict(color=line_color, width=1),
                            fill='tozeroy',
                            fillcolor=fillcolor,
                            yaxis='y2',
                            hovertemplate=f'<b>DR Hour {hour:02d}:00</b><br>Time: %{{x}}<br>LMP: %{{y:.2f}} $/MWh<extra></extra>'
                        ))

                    # Update layout with dual y-axis
                    fig_strategy.update_layout(
                        title=f"{prosumer_id} - Load, LMP, and DR Opportunities",
                        xaxis_title="Time",
                        yaxis=dict(title="Net Load (kW)", side="left"),
                        yaxis2=dict(title="LMP ($/MWh)", side="right", overlaying="y"),
                        height=400,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_strategy, use_container_width=True)
                    
                    # Debug info
                    with st.expander(f"Debug info for {prosumer_id}"):
                        st.write(f"DR opportunities found: {len(dr_opportunities)}")
                        st.write(f"First few opportunities: {dr_opportunities[:3]}")
                        if response.get('final_participation'):
                            st.write(f"Final participation: {response['final_participation'][:5]}")
                        else:
                            st.write("No final participation data")
                else:
                    st.warning(f"No DR opportunities data found for {prosumer_id}")
                    # Debug info
                    with st.expander(f"Debug info for {prosumer_id}"):
                        st.write(f"Offer keys: {list(offer.keys()) if offer else 'No offer'}")
                        st.write(f"State level DR opportunities: {len(final_state.get('dr_opportunities', []))}")
                        st.write(f"Sample state keys: {list(final_state.keys())}")
        
            st.subheader("Negotiation Offers Summary")
            for prosumer_id, offers in final_state['negotiation_offers'].items():
                st.write(f"**Offers to {prosumer_id}:**")
                if isinstance(offers, dict) and 'offered_rates' in offers:
                    avg_rate = np.mean(offers['offered_rates'])
                    st.write(f"Average offered rate: ${avg_rate:.3f}/kWh")
        else:
            st.warning("No negotiation offers found. Please run the simulation first.")
    
    # Tab 4: Final Results (includes prosumer responses)
    with tab4:
        st.header("💰 Final Results")
        
        results = calculate_simulation_results(final_state)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Aggregator Profit", 
                f"${results['aggregator_profit']:.2f}",
                help="Total profit from all DR transactions"
            )
        
        with col2:
            st.metric(
                "Total Committed Power", 
                f"{results['total_committed_kw']:.2f} kWh",
                help="Total committed demand response energy"
            )
        
        with col3:
            total_prosumer_profit = sum(results['prosumer_profits'].values())
            st.metric(
                "Total Prosumer Earnings", 
                f"${total_prosumer_profit:.2f}",
                help="Combined earnings of all prosumers"
            )
        
        with col4:
            total_dissatisfaction = sum(results['prosumer_dissatisfactions'].values())
            st.metric(
                "Total Dissatisfaction", 
                f"{total_dissatisfaction:.3f}",
                help="Combined dissatisfaction of all prosumers"
            )
        
        # Prosumer Response Details
        if final_state.get('prosumer_responses'):
            st.subheader("🏠 Prosumer Response Details")
            for prosumer_id, response in final_state['prosumer_responses'].items():
                st.write(f"**Prosumer {prosumer_id}**")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    if 'proposed_commitments' in response:
                        total_commitment = sum(response['proposed_commitments'])
                        st.metric("Total Proposed Commitment", f"{total_commitment:.2f} kWh")
                    
                    if 'final_commitments' in response:
                        final_commitment = sum(response['final_commitments'])
                        st.metric("Final Commitment", f"{final_commitment:.2f} kWh")
                
                with col_b:
                    if 'proposed_rates' in response:
                        avg_proposed_rate = np.mean(response['proposed_rates'])
                        st.metric("Average Proposed Rate", f"${avg_proposed_rate:.3f}/kWh")
                    
                    if 'accepted_rates' in response:
                        avg_accepted_rate = np.mean(response['accepted_rates'])
                        st.metric("Average Accepted Rate", f"${avg_accepted_rate:.3f}/kWh")
                
                st.markdown("---")
        
        # Financial breakdown by prosumer
        st.subheader("💸 Detailed Financial Analysis")
        
        for prosumer_id in selected_prosumers:
            if prosumer_id in results['prosumer_profits']:
                with st.expander(f"Prosumer {prosumer_id} Financial Analysis"):
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        profit = results['prosumer_profits'].get(prosumer_id, 0)
                        st.metric("Total Earnings", f"${profit:.2f}")
                        
                        commitment = results['market_participation'].get(prosumer_id, 0)
                        st.metric("Total Commitment", f"{commitment:.2f} kWh")
                    
                    with col_b:
                        dissatisfaction = results['prosumer_dissatisfactions'].get(prosumer_id, 0)
                        st.metric("Dissatisfaction Score", f"{dissatisfaction:.3f}")
                        
                        # Show dissatisfaction in monetary terms (smaller penalty)
                        dissatisfaction_cost = dissatisfaction * 1  # $1 per dissatisfaction unit
                        st.metric("Dissatisfaction Cost", f"${dissatisfaction_cost:.2f}")
                    
                    with col_c:
                        # Net benefit calculation with smaller penalty
                        net_benefit = profit - dissatisfaction_cost
                        st.metric("Net Benefit", f"${net_benefit:.2f}", 
                                 delta=f"{'Positive' if net_benefit > 0 else 'Negative'}")
                        
                        # Show profit margin
                        if profit > 0:
                            profit_margin = (net_benefit / profit) * 100
                            st.metric("Profit Margin", f"{profit_margin:.1f}%")
        
        # Market participation visualization
        if results['market_participation']:
            st.subheader("📊 Market Participation")
            
            participation_df = pd.DataFrame([
                {'Prosumer': k, 'Committed Power (kWh)': v} 
                for k, v in results['market_participation'].items()
            ])
            
            fig_participation = px.bar(
                participation_df, 
                x='Prosumer', 
                y='Committed Power (kWh)',
                title="Committed Energy by Prosumer"
            )
            
            st.plotly_chart(fig_participation, use_container_width=True)
    
    # Tab 5: System Log (collapsible to save space)
    with tab5:
        st.header("📝 System Log")
        log_text = st.session_state.get('simulation_log_text', None)
        if not log_text:
            st.info("System log will appear here after running the simulation. Use the 'Run VPP Simulation' button in the sidebar.")
        else:
            with st.expander("Show system log (toggle)", expanded=False):
                st.text_area("System Log", value=log_text, height=400)

if __name__ == "__main__":
    main()
