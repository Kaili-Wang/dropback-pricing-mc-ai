# Pricing of Drop-Back Certificates: MC Simulation & MLP Surrogate

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dropback-pricing-mc-ai.streamlit.app/)

> 🏆 **Accepted for publication in the 2026 2nd International Conference on Financial Innovation and Marketing Management (FIMM).**

This repository provides a high-performance framework for pricing USD Drop-Back Certificates. It implements a comparative analysis between traditional numerical methods (Monte-Carlo), a highly optimized vectorized computing engine, and a machine learning-based surrogate model (Multi-Layer Perceptron).

## 🔑 Core Methodology
- **Stochastic Engine:** 50,000-path Monte-Carlo simulation under Geometric Brownian Motion (GBM) to establish pricing benchmarks.
- **Vectorized HPC Optimization:** Implementation of a purely tensor-based NumPy execution engine, completely eliminating native Python loops. This architectural upgrade delivers a **13x computational speedup**, dropping data generation time from 26 seconds to 2 seconds.
- **Surrogate Modeling:** A Multi-Layer Perceptron (MLP) trained on the simulation results to learn the complex mapping between market parameters and instrument value.
- **Computational Efficiency:** The AI surrogate achieves microsecond-level inference latency, providing a massive speedup over traditional simulations while maintaining numerical convergence.

## 📂 Project Structure
- `app.py`: The interactive Streamlit dashboard for real-time pricing and performance comparison.
- `data/`: Contains the datasets used for model training (`train.csv`, `train_numpy.csv`) and out-of-sample evaluation. *(Note: Large CSV files are git-ignored to keep the repository lightweight).*
- `models/`: Pre-trained MLP model weights (`.pkl`) serialized for production use.
- `scripts/`:
  - `step1_mc_base.py`: The original pure-Python Monte Carlo engine.
  - `step1b_mc_numpy.py`: The high-performance vectorized NumPy engine (for massive data synthesis).
  - `step2_ai_trainer.py`: The training pipeline for the neural surrogate model.
- `requirements.txt`: List of Python dependencies for environment reproduction.
- `Drop_Back_Pricing_MC_and_AI_Comparison.pdf`: The full research paper detailing the mathematical framework and empirical results.

## 🚀 Deployment
### Web Interface
Access the live interactive dashboard here:  
👉 [https://dropback-pricing-mc-ai.streamlit.app/](https://dropback-pricing-mc-ai.streamlit.app/)

### Local Execution
1. Install the required libraries:  
   `pip install -r requirements.txt`
2. Launch the application:  
   `streamlit run app.py`

## 💻 Project Creator
- **Kaili Wang** (New York University)
