#!/bin/bash

# Virtual Power Plant (VPP) Multi-Agent Simulation Dashboard
# Launch script for Streamlit application

echo "🚀 Starting VPP Multi-Agent Simulation Dashboard..."
echo "📝 Make sure to set your GEMINI API keys in the .env file"

# Activate virtual environment and run Streamlit
source .venv/bin/activate
streamlit run streamlit_dashboard.py --server.port 8501 --server.address localhost

echo "✅ Dashboard closed"
