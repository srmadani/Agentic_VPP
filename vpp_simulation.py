"""
Multi-Agent Virtual Power Plant (VPP) Simulation
Using LangGraph to orchestrate negotiations between Aggregator and Prosumer agents
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import TypedDict, List, Dict, Any, Annotated
import random
import json
import time
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# Load environment variables
load_dotenv()

# Verify API keys exist
if not os.getenv("GEMINI_PRO_API_KEY") or not os.getenv("GEMINI_FLASH_API_KEY"):
    raise ValueError("Missing GEMINI API keys in .env file")

# Initialize LLMs 
aggregator_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=os.getenv("GEMINI_PRO_API_KEY"),
    temperature=0.2,
    timeout=30,
    max_retries=2
)

prosumer_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_FLASH_API_KEY"),
    temperature=0.3,
    timeout=30,
    max_retries=2
)

# Graph State Definition
class VPPState(TypedDict):
    market_data: pd.DataFrame
    prosumer_profiles: Dict[str, Any]
    negotiation_offers: Dict[str, Dict]
    prosumer_responses: Dict[str, Dict]
    round_number: int
    selected_prosumers: List[str]
    dr_opportunities: List[Dict]
    simulation_log: List[Dict]
    messages: Annotated[list, add_messages]

# Global variables to store state (needed for tool access)
_current_state = None

def set_current_state(state):
    global _current_state
    _current_state = state

@tool
def find_best_dr_opportunities(top_n: int = 3, prosumer_id: str = None) -> List[Dict]:
    """Select top 3 hours based on aggregated DR scores per hour."""
    if _current_state is None:
        return []

    pricing_data = _current_state["market_data"].reset_index(drop=True).copy()
    n_slots = len(pricing_data)

    # Prepare net demand vector
    if prosumer_id and prosumer_id in _current_state["prosumer_profiles"]:
        profile = _current_state["prosumer_profiles"][prosumer_id]
        demand = np.array(profile.get("demand_kw", [0] * n_slots), dtype=float)
        generation = np.array(profile.get("generation_kw", [0] * n_slots), dtype=float)
        net_demand = np.maximum(0.0, demand[:n_slots] - generation[:n_slots])
        # If profile lists shorter than pricing, pad with zeros
        if len(net_demand) < n_slots:
            net_demand = np.pad(net_demand, (0, n_slots - len(net_demand)), 'constant')
    else:
        # No prosumer-specific demand available: use zeros (so score driven by price only)
        net_demand = np.zeros(n_slots, dtype=float)

    # Extract LMP as numpy array
    lmp = pricing_data['lmp'].astype(float).to_numpy()

    # Min-max normalization helper (returns array of same shape)
    def minmax_norm(arr):
        amin = arr.min() if len(arr) > 0 else 0.0
        amax = arr.max() if len(arr) > 0 else 0.0
        if amax > amin:
            return (arr - amin) / (amax - amin)
        # fallback: if constant array, return zeros
        return np.zeros_like(arr, dtype=float)

    lmp_norm = minmax_norm(lmp)
    demand_norm = minmax_norm(net_demand)

    # Calculate DR score for each 5-minute slot
    dr_score = lmp_norm + demand_norm
    
    # Add columns to pricing_data
    pricing_data['net_demand'] = net_demand
    pricing_data['dr_score'] = dr_score
    
    # Extract hour from timestamp and group by hour
    pricing_data['hour'] = pd.to_datetime(pricing_data['timestamp']).dt.hour
    
    # Aggregate DR scores by hour (sum all 5-minute slots within each hour)
    hourly_scores = pricing_data.groupby('hour').agg({
        'dr_score': 'sum',
        'lmp': 'mean',
        'net_demand': 'mean',
        'timestamp': 'first'  # Take first timestamp of each hour
    }).reset_index()
    
    # Sort by aggregated DR score and take top 3 hours
    top_n = max(1, min(int(top_n), len(hourly_scores)))
    best_hours = hourly_scores.nlargest(top_n, 'dr_score')

    # Create result records for the top hours
    best_records = []
    for _, row in best_hours.iterrows():
        best_records.append({
            'timestamp': str(row['timestamp']),
            'hour': int(row['hour']),
            'lmp': float(row['lmp']),
            'net_demand': float(row['net_demand']),
            'combined_score': float(row['dr_score'])
        })

    return best_records

@tool
def calculate_dissatisfaction(prosumer_id: str, commitment_kw: float, demand_kw: float) -> float:
    """Calculate prosumer dissatisfaction based on their profile characteristics"""
    
    if _current_state is None:
        return 0.0
    
    profiles = _current_state["prosumer_profiles"]
    profile = profiles[prosumer_id]
    willingness = profile['willingness_to_participate']
    
    # Get energy awareness from description or default to medium
    description = profile.get('description', '').lower()
    if 'high awareness' in description:
        energy_awareness = 'high'
    elif 'low awareness' in description:
        energy_awareness = 'low'
    else:
        energy_awareness = 'medium'
    
    # Convert energy_awareness to numeric
    awareness_map = {'low': 0.3, 'medium': 0.6, 'high': 0.9}
    awareness_score = awareness_map.get(energy_awareness, 0.6)
    
    # Calculate dynamic alpha based on profile
    # Higher willingness and awareness = lower alpha (less dissatisfaction)
    base_alpha = 1.0
    alpha = base_alpha * (1 - (willingness / 10) * 0.5) * (1 - awareness_score * 0.3)
    
    # Add some randomness to make it more realistic
    alpha *= random.uniform(0.8, 1.2)
    
    # Calculate dissatisfaction
    if demand_kw == 0:
        return 0.0
    
    dissatisfaction = alpha * (commitment_kw / demand_kw)
    return max(0.0, float(dissatisfaction))

@tool
def calculate_aggregator_profit(market_rate: float, prosumer_rate: float, committed_kw: float) -> float:
    """Calculate aggregator profit for a single transaction"""
    return float((market_rate - prosumer_rate) * committed_kw)

# Bind tools to LLMs
aggregator_llm_with_tools = aggregator_llm.bind_tools([
    find_best_dr_opportunities,
    calculate_aggregator_profit
])

prosumer_llm_with_tools = prosumer_llm.bind_tools([
    calculate_dissatisfaction
])

# Agent Node Functions
def aggregator_proposer(state: VPPState) -> VPPState:
    """Aggregator analyzes market and makes initial offers to prosumers"""
    
    print("🏢 Aggregator: Starting initial proposal phase...")
    set_current_state(state)
    
    # Create offers for each selected prosumer
    offers = {}
    
    for prosumer_id in state["selected_prosumers"]:
        print(f"🏢 Aggregator: Creating offer for {prosumer_id}...")
        profile = state["prosumer_profiles"][prosumer_id]
        
        # Find best DR opportunities for this specific prosumer
        print(f"🏢 Aggregator: Finding best DR opportunities for {prosumer_id}...")
        dr_opportunities = find_best_dr_opportunities.invoke({"top_n": 3, "prosumer_id": prosumer_id})
        print(f"🏢 Aggregator: Found {len(dr_opportunities)} opportunities for {prosumer_id}")
        
        # Create personalized prompt based on prosumer profile
        profile_desc = profile.get("description", "")
        willingness = profile["willingness_to_participate"]
        
        # Simplified prompt to reduce token usage
        prompt = f"""You are a VPP aggregator. Create a brief offer for prosumer {prosumer_id}.

Profile: {profile_desc[:100]}...
Willingness: {willingness}/10

Top 3 market opportunities:
{json.dumps(dr_opportunities[:3], indent=2)}

Write a concise, persuasive message offering 95% of market rate for demand reduction during these high-price periods."""
        
        try:
            print(f"🏢 Aggregator: Calling Gemini API for {prosumer_id}...")
            # Add delay to respect rate limits
            time.sleep(2)
            response = aggregator_llm.invoke([HumanMessage(content=prompt)])
            print(f"🏢 Aggregator: Got response for {prosumer_id}")
            
            # Create offer structure
            offer_data = {
                "time_slots": [opp["timestamp"] for opp in dr_opportunities],
                # Convert LMP from $/MWh to $/kWh (divide by 1000) and apply 5% discount
                "offered_rates": [float(opp["lmp"]) / 1000.0 * 0.95 for opp in dr_opportunities],  # 5% discount, now $/kWh
                "message": response.content,
                "dr_opportunities": dr_opportunities
            }
            
            offers[prosumer_id] = offer_data
            
        except Exception as e:
            print(f"🏢 Aggregator: Error with LLM for {prosumer_id}: {e}")
            raise  # Don't use fallback, raise the error
    
    print("🏢 Aggregator: Completed initial proposals")
    # Update state
    new_state = state.copy()
    new_state["negotiation_offers"] = offers
    new_state["dr_opportunities"] = dr_opportunities
    new_state["round_number"] = 1
    new_state["simulation_log"] = state.get("simulation_log", []) + [{
        "step": "aggregator_initial_offer",
        "round": 1,
        "data": offers,
        "timestamp": datetime.now()
    }]
    
    return new_state

def prosumer_responder(state: VPPState, prosumer_id: str) -> VPPState:
    """Prosumer evaluates offer and makes counter-offer"""
    
    print(f"🏠 {prosumer_id}: Starting response evaluation...")
    set_current_state(state)
    
    offer = state["negotiation_offers"][prosumer_id]
    profile = state["prosumer_profiles"][prosumer_id]
    
    # Get prosumer's demand for the DR opportunity hours
    demand_data = profile["demand_kw"]
    generation_data = profile["generation_kw"]
    dr_opportunities = offer.get("dr_opportunities", [])
    
    # Calculate net demand for each DR hour (energy in kWh for the hour)
    net_demands = []
    for opp in dr_opportunities[:3]:  # Top 3 DR opportunities
        hour = opp['hour']
        # Sum demand for all 5-minute intervals in this hour (12 intervals per hour)
        hour_start_idx = hour * 12  # 12 five-minute intervals per hour
        hour_end_idx = min(hour_start_idx + 12, len(demand_data))
        
        # Convert kW readings to kWh: sum kW values and multiply by 5 minutes / 60 minutes per hour
        hour_demand_kwh = sum(demand_data[hour_start_idx:hour_end_idx]) * (5/60)  # kWh
        hour_generation_kwh = sum(generation_data[hour_start_idx:hour_end_idx]) * (5/60)  # kWh
        net_demand_hour = max(0, hour_demand_kwh - hour_generation_kwh)  # Total kWh for the hour
        net_demands.append(net_demand_hour)
    
    # Simplified prompt to reduce token usage
    prompt = f"""You are prosumer {prosumer_id}. Evaluate this demand response offer:

Profile: {profile.get('description', '')[:100]}...
Willingness: {profile['willingness_to_participate']}/10

Offer: {offer['message'][:200]}...

Top 3 DR hours: {[opp['hour'] for opp in dr_opportunities[:3]]}
Rates: {[f"${rate:.3f}/kWh" for rate in offer['offered_rates'][:3]]}
Your hourly net demand: {[f"{nd:.1f}kWh" for nd in net_demands[:3]]}

Respond briefly: Will you participate? What rates do you want?"""

    try:
        print(f"🏠 {prosumer_id}: Calling Gemini API for response...")
        # Add delay to respect rate limits
        time.sleep(2)
        response = prosumer_llm.invoke([HumanMessage(content=prompt)])
        print(f"🏠 {prosumer_id}: Got response from Gemini")
        
        # Create response structure with reasonable commitments (kWh for the hour)
        # Commit to reducing 40-60% of net demand for each hour
        commitment_percentages = [0.5, 0.45, 0.55]  # Higher percentages for more realistic values
        prosumer_response = {
            "participating_slots": [True] * min(3, len(offer["time_slots"])),
            "proposed_commitments": [max(nd * pct, 0.5) for nd, pct in zip(net_demands, commitment_percentages)],  # kWh commitments, minimum 0.5 kWh
            "proposed_rates": [rate * 1.1 for rate in offer["offered_rates"][:3]],  # 10% markup
            "counter_message": response.content,
            "net_demands": net_demands
        }
        
    except Exception as e:
        print(f"🏠 {prosumer_id}: Error with LLM: {e}")
        raise  # Don't use fallback, raise the error
    
    # Update state
    new_state = state.copy()
    if "prosumer_responses" not in new_state:
        new_state["prosumer_responses"] = {}
    
    new_state["prosumer_responses"][prosumer_id] = prosumer_response
    new_state["simulation_log"] = state.get("simulation_log", []) + [{
        "step": f"prosumer_response_{prosumer_id}",
        "round": 1,
        "data": prosumer_response,
        "timestamp": datetime.now()
    }]
    
    return new_state

def prosumer_responder_1(state: VPPState) -> VPPState:
    """Prosumer 1 response wrapper"""
    return prosumer_responder(state, state["selected_prosumers"][0])

def prosumer_responder_2(state: VPPState) -> VPPState:
    """Prosumer 2 response wrapper"""
    return prosumer_responder(state, state["selected_prosumers"][1])

def aggregator_reviser(state: VPPState) -> VPPState:
    """Aggregator revises offers based on prosumer responses"""
    
    print("🏢 Aggregator: Revising offers based on responses...")
    set_current_state(state)
    
    responses = state["prosumer_responses"]
    
    # Simplified prompt
    prompt = f"""VPP Aggregator Round 2: Revise offers based on prosumer feedback.

Prosumer responses:
{json.dumps({k: v.get('counter_message', '')[:100] + '...' for k, v in responses.items()}, indent=2)}

Create brief revised offers. Increase rates slightly for better acceptance."""

    try:
        print("🏢 Aggregator: Getting revised strategy...")
        time.sleep(2)  # Rate limiting
        response = aggregator_llm.invoke([HumanMessage(content=prompt)])
        
        # Create revised offers
        revised_offers = {}
        for prosumer_id in state["selected_prosumers"]:
            original_offer = state["negotiation_offers"][prosumer_id]
            prosumer_resp = responses.get(prosumer_id, {})
            
            # Simple revision logic: meet halfway on rates
            revised_rates = []
            original_rates = original_offer["offered_rates"][:3]
            proposed_rates = prosumer_resp.get("proposed_rates", original_rates)
            
            for orig_rate, proposed_rate in zip(original_rates, proposed_rates):
                revised_rate = (orig_rate + proposed_rate) / 2
                revised_rates.append(revised_rate)
            
            revised_offers[prosumer_id] = {
                "time_slots": original_offer["time_slots"][:3],
                "offered_rates": revised_rates,
                "message": response.content,
                "round": 2
            }
            
    except Exception as e:
        print(f"🏢 Aggregator: Error in revision: {e}")
        raise  # Don't use fallback, raise the error
    
    new_state = state.copy()
    new_state["negotiation_offers"] = revised_offers
    new_state["round_number"] = 2
    new_state["simulation_log"] = state.get("simulation_log", []) + [{
        "step": "aggregator_revised_offer",
        "round": 2,
        "data": revised_offers,
        "timestamp": datetime.now()
    }]
    
    return new_state

def prosumer_finalizer(state: VPPState, prosumer_id: str) -> VPPState:
    """Prosumer makes final decision"""
    
    print(f"🏠 {prosumer_id}: Making final decision...")
    set_current_state(state)
    
    revised_offer = state["negotiation_offers"][prosumer_id]
    original_response = state["prosumer_responses"][prosumer_id]
    profile = state["prosumer_profiles"][prosumer_id]
    
    # Simplified prompt
    prompt = f"""Final decision for prosumer {prosumer_id}:

Willingness: {profile['willingness_to_participate']}/10
Revised rates: {[f"${rate:.3f}" for rate in revised_offer['offered_rates'][:3]]}

Accept or reject? Brief response."""

    try:
        print(f"🏠 {prosumer_id}: Getting final decision...")
        time.sleep(2)  # Rate limiting
        response = prosumer_llm.invoke([HumanMessage(content=prompt)])
        
        # Create final response
        final_response = {
            "final_participation": [True] * len(revised_offer["time_slots"]),
            "final_commitments": original_response.get("proposed_commitments", [2.0] * len(revised_offer["time_slots"])),
            "accepted_rates": revised_offer["offered_rates"],
            "final_message": response.content
        }
        
    except Exception as e:
        print(f"🏠 {prosumer_id}: Error in final decision: {e}")
        # Fallback final decision
        final_response = {
            "final_participation": [True] * len(revised_offer["time_slots"]),
            "final_commitments": original_response.get("proposed_commitments", [0.5] * len(revised_offer["time_slots"])),
            "accepted_rates": revised_offer["offered_rates"],
            "final_message": "I accept your revised offer."
        }
    
    # Update prosumer response with final decision
    new_state = state.copy()
    new_state["prosumer_responses"][prosumer_id].update(final_response)
    new_state["simulation_log"] = state.get("simulation_log", []) + [{
        "step": f"prosumer_final_{prosumer_id}",
        "round": 2,
        "data": final_response,
        "timestamp": datetime.now()
    }]
    
    return new_state

def prosumer_finalizer_1(state: VPPState) -> VPPState:
    """Prosumer 1 finalizer wrapper"""
    return prosumer_finalizer(state, state["selected_prosumers"][0])

def prosumer_finalizer_2(state: VPPState) -> VPPState:
    """Prosumer 2 finalizer wrapper"""
    return prosumer_finalizer(state, state["selected_prosumers"][1])

def create_vpp_graph():
    """Create and configure the LangGraph workflow"""
    
    # Create the graph
    workflow = StateGraph(VPPState)
    
    # Add nodes
    workflow.add_node("aggregator_proposer", aggregator_proposer)
    workflow.add_node("prosumer_responder_1", prosumer_responder_1)
    workflow.add_node("prosumer_responder_2", prosumer_responder_2)
    workflow.add_node("aggregator_reviser", aggregator_reviser)
    workflow.add_node("prosumer_finalizer_1", prosumer_finalizer_1)
    workflow.add_node("prosumer_finalizer_2", prosumer_finalizer_2)
    
    # Set entry point
    workflow.set_entry_point("aggregator_proposer")
    
    # Add edges - sequential flow to ensure proper state passing
    workflow.add_edge("aggregator_proposer", "prosumer_responder_1")
    workflow.add_edge("prosumer_responder_1", "prosumer_responder_2")
    workflow.add_edge("prosumer_responder_2", "aggregator_reviser")
    workflow.add_edge("aggregator_reviser", "prosumer_finalizer_1")
    workflow.add_edge("prosumer_finalizer_1", "prosumer_finalizer_2")
    workflow.add_edge("prosumer_finalizer_2", END)
    
    # Try different compilation approaches for compatibility
    try:
        # Try with explicit empty config
        return workflow.compile(checkpointer=None, store=None)
    except (TypeError, AttributeError):
        try:
            # Try basic compile
            return workflow.compile()
        except Exception:
            # Final fallback - just return the workflow itself
            return workflow

def run_vpp_simulation(first_day_only=True):
    """Run the VPP simulation for the first day"""
    
    print("Loading data...")
    
    # API keys assumed valid (pre-checked)
    
    # Load data
    pkl_path = "data/electricity_profiles.pkl"
    parquet_path = "data/nyc_pricing_with_solar_real_data.parquet"
    
    try:
        electricity_profiles = pd.read_pickle(pkl_path)
        pricing_nyc = pd.read_parquet(parquet_path)
    except Exception as e:
        raise Exception(f"Error loading data: {e}")
    
    # Filter for first day only
    if first_day_only:
        first_day = pricing_nyc['date'].min()
        pricing_nyc = pricing_nyc[pricing_nyc['date'] == first_day].copy()
    
    print(f"Using data from {pricing_nyc['date'].iloc[0]} with {len(pricing_nyc)} time slots")
    
    # Select two prosumers for the simulation
    selected_prosumers = ['profile_001', 'profile_002']
    
    print(f"Selected prosumers: {selected_prosumers}")
    
    # Create initial state
    initial_state = VPPState(
        market_data=pricing_nyc,
        prosumer_profiles=electricity_profiles,
        negotiation_offers={},
        prosumer_responses={},
        round_number=0,
        selected_prosumers=selected_prosumers,
        dr_opportunities=[],
        simulation_log=[],
        messages=[]
    )
    
    print("Creating LangGraph workflow...")
    
    # Create and run the graph
    try:
        graph = create_vpp_graph()
        print("Starting simulation...")
        
        # Use stream instead of invoke to avoid checkpointing issues
        final_state = None
        for chunk in graph.stream(initial_state):
            final_state = chunk
        
        # Extract the final state from the last chunk
        if final_state and isinstance(final_state, dict):
            # Get the last node's output
            final_state = list(final_state.values())[-1]
        
        print("Simulation completed successfully!")
        
        return final_state
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    # Test the simulation
    try:
        result = run_vpp_simulation()
        print(f"Simulation completed with {len(result['simulation_log'])} log entries")
        
        # Print summary
        print("\n=== SIMULATION SUMMARY ===")
        for log_entry in result['simulation_log']:
            print(f"{log_entry['timestamp'].strftime('%H:%M:%S')} - {log_entry['step']} (Round {log_entry['round']})")
        
        print(f"\nFinal prosumer responses: {list(result['prosumer_responses'].keys())}")
        
    except Exception as e:
        print(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()
