# Agentic_VPP — Virtual Power Plant Multi‑Agent Simulation

Brief, opinionated README to quickly understand and reproduce the repository.

## What this repo contains
- `vpp_simulation.py` — Core multi‑agent VPP simulation. Orchestrates an Aggregator and multiple Prosumer agents using a state graph (LangGraph) and LLMs (Google Gemini via LangChain). Implements tools for selecting DR opportunities, computing prosumer dissatisfaction, and calculating aggregator profit.
- `streamlit_dashboard.py` — Streamlit UI that visualizes market LMP, prosumer profiles, negotiation messages, aggregator offers, and final results. Provides buttons to run or load a saved simulation and shows interactive charts (Plotly).
- `data/` — Market and profile data. Full data collection and preprocessing documentation is in `data/README.md` (produced by the data pipeline). Key files used by the repo: `data/electricity_profiles.pkl` and `data/nyc_pricing_with_solar_real_data.parquet`.

## High level design (quick)

- Input data: 5‑minute market pricing (`lmp`, timestamps) and prosumer profiles (per‑profile 5‑min demand & generation arrays, metadata such as willingness to participate).
- Simulation flow: `run_vpp_simulation()` loads the first day of market data and two prosumers, builds an initial typed state, creates a LangGraph workflow and executes it. Nodes:
  - `aggregator_proposer` — finds DR hours (uses `find_best_dr_opportunities`) and issues initial offers.
  - `prosumer_responder` (per prosumer) — evaluates offers, computes net demand, generates counter‑offers (uses `calculate_dissatisfaction`).
  - `aggregator_reviser` — revises offers based on prosumer feedback.
  - `prosumer_finalizer` (per prosumer) — final accept/reject and final commitments.
- Tools: small helper functions registered as tools for the LLMs: `find_best_dr_opportunities`, `calculate_dissatisfaction`, and `calculate_aggregator_profit`. The simulation stores a `VPPState` typed dict that passes through nodes and accumulates `simulation_log`, `negotiation_offers`, `prosumer_responses`, and `dr_opportunities`.

## File summaries

- `vpp_simulation.py`
  - Purpose: full negotiation simulation glue between market data, prosumer profiles, and LLM‑based agents.
  - Key responsibilities: load data, build initial `VPPState`, define LangGraph nodes and edges, register/help LLM tools, run the workflow, and return a final state with logs and results.
  - Important variables / types: `VPPState` TypedDict, `_current_state` (global helper used by tools), `aggregator_llm`, `prosumer_llm`.
  - External dependencies: LangGraph, LangChain Google Generative AI adapter, pandas, python‑dotenv.

- `streamlit_dashboard.py`
  - Purpose: quick interactive UI to visualize market data and the negotiation flow. Plots LMP, individual prosumer daily profiles, dissatisfaction analysis, and a step‑by‑step message log.
  - How it interacts with simulation: calls `run_vpp_simulation()` from `vpp_simulation.py` to produce `simulation_result`, then renders the results across multiple tabs.
  - Notes: expects GEMINI API keys in environment to run the LLM driven simulation. It also includes save/load helpers for `simulation_results.pkl`.

## Quick start — reproduce locally

1. Install dependencies (recommended: venv).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Add API keys to a `.env` in the repo root (required for LLM calls):

```
GEMINI_PRO_API_KEY=your_gemini_pro_key
GEMINI_FLASH_API_KEY=your_gemini_flash_key
```

3. Verify data files exist (see `data/README.md`). The simulation and dashboard expect:
- `data/electricity_profiles.pkl`
- `data/nyc_pricing_with_solar_real_data.parquet`

4. Run the Streamlit dashboard (recommended for exploration):

```bash
streamlit run streamlit_dashboard.py
```

5. Or run the simulation directly (for batch/testing) from a Python REPL or small runner that imports and calls:

```python
from vpp_simulation import run_vpp_simulation
final_state = run_vpp_simulation(first_day_only=True)
```

## Contract / data shapes
- Inputs: `market_data` pandas DataFrame with columns `timestamp` (datetime-like) and `lmp` (numeric), and `prosumer_profiles` dict keyed by profile id with lists/arrays for `demand_kw` and `generation_kw` (5‑minute resolution). Profiles include `willingness_to_participate` and optional `description` metadata.
- Outputs: final `VPPState`‑like dict containing `negotiation_offers`, `prosumer_responses`, `dr_opportunities`, `simulation_log`, and other aggregated result counters. `streamlit_dashboard.py` saves a pickle `simulation_results.pkl` for later reloading.
- Error modes: missing API keys, missing data files, or model/timeouts when talking to Gemini. The code performs basic checks and raises/aborts on missing keys or unreadable files.

## Edge cases & testing notes
- Empty or missing profiles: code expects at least two profiles (`profile_001`, `profile_002` by default) — ensure your `electricity_profiles.pkl` contains them or adjust `selected_prosumers`.
- Zero demand slots: tools guard against divide‑by‑zero but watch profiling functions that compute ratios of commitment/demand.
- LLM failures (timeouts, quota): simulation is resilient only to a point; for deterministic testing, stub the LLM calls or mock the tool outputs.

## Where to find full data & preprocessing docs
See `data/README.md` for detailed notes on data sources, profile generation, and the market data preprocessing pipeline.

## Next steps (suggested)
- Add a small unit test for `find_best_dr_opportunities` and `calculate_dissatisfaction` that runs quickly without LLM calls.
- Add a minimal runner script `run_simulation.py` that wraps `run_vpp_simulation()` and writes out `simulation_results.pkl` for CI and non‑interactive use.