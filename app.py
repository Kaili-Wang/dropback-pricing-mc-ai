import streamlit as st
import joblib
import time
import os
import math
import random
import pandas as pd

# App configuration
st.set_page_config(page_title="Drop-Back Pricing AI", layout="wide")

st.title("⚡ Drop-Back Option Pricing")
st.markdown("### Comparison: Traditional Monte-Carlo vs Neural Surrogate Model")
st.markdown("---")

# Load pre-trained surrogate model
@st.cache_resource
def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'models', 'mlp_surrogate_model.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

ai_model = load_model()

# Traditional Monte Carlo engine with UI progress hooks
def run_traditional_mc_with_progress(sigma, progress_bar, status_text):
    r, q, T = 0.02, 0.018, 3.0
    steps_per_year = 252
    N = int(T * steps_per_year)
    dt = 1.0 / steps_per_year
    S0 = 3790.38
    initial_investment, additional_investment = 550.0, 150.0
    initial_cash, cash_rate = 450.0, 0.0985
    trigger_levels = [0.90 * S0, 0.85 * S0, 0.80 * S0]
    num_paths = 50000
    fixed_seed = 2026
    discount_factor = math.exp(-r * T)
    
    path_values = []
    drift_part = (r - q - 0.5 * sigma ** 2) * dt
    diffusion_coef = sigma * math.sqrt(dt)

    for p in range(num_paths):
        # Update UI progress periodically
        if p % 5000 == 0 and p > 0:
            progress_bar.progress(p / num_paths)
            status_text.info(f"⏳ CPU is computing (approx. 20-60s depending on CPU)... Path {p:,} / 50,000")

        local_rng = random.Random(fixed_seed + p)
        s_t = S0
        cash = initial_cash
        invested_amts = [initial_investment]
        entry_lvls = [S0]
        accrued_interest = 0.0
        triggers_hit = 0

        for _ in range(N):
            z = local_rng.gauss(0.0, 1.0)
            exponent = max(min(drift_part + diffusion_coef * z, 50), -50)
            s_t *= math.exp(exponent)
            accrued_interest += cash * cash_rate * dt

            while triggers_hit < 3 and s_t <= trigger_levels[triggers_hit]:
                cash -= additional_investment
                invested_amts.append(additional_investment)
                entry_lvls.append(s_t)
                triggers_hit += 1

        equity_part = sum(amt * (s_t / lvl) for amt, lvl in zip(invested_amts, entry_lvls))
        path_values.append((equity_part + cash + accrued_interest) * discount_factor)

    progress_bar.progress(1.0)
    return sum(path_values) / len(path_values)

# Sidebar parameters
with st.sidebar:
    st.header("📋 Set Parameters")
    user_sigma = st.slider("Volatility (Sigma)", min_value=0.1500, max_value=0.4500, value=0.2000, step=0.0050, format="%.4f")
    
    st.markdown("---")
    st.write("📖 **Paper Authors:**")
    st.write("Haocheng Fu, Zhourui Shen, Kaili Wang, Jingheng Xu")
    st.write("💻 **Web App Developer:**")
    st.write("**Kaili Wang**")

st.markdown("<br>", unsafe_allow_html=True)
start_race = st.button("🏁 Start Racing (Compare Models)", type="primary", use_container_width=True)
st.markdown("<br>", unsafe_allow_html=True)

# Layout setup
col1, col2 = st.columns(2)

with col1:
    st.subheader("⚙️ Traditional Monte-Carlo")
    st.write("Runs 50,000 paths utilizing CPU.")
    mc_container = st.empty()

with col2:
    st.subheader("🧠 Neural Surrogate Model")
    st.write("Inference based on trained MLP architecture.")
    ai_container = st.empty()

# Execution logic
if start_race:
    if ai_model is None:
        st.error("Model not found! Please check the models folder.")
    else:
        # Neural Surrogate Inference
        ai_start_time = time.perf_counter()
        ai_price = ai_model.predict(pd.DataFrame({'sigma': [user_sigma]}).values)[0]
        ai_time = time.perf_counter() - ai_start_time
        
        with ai_container.container():
            st.success("✨ Instant Inference Completed!")
            st.metric(label="Expected PV (USD)", value=f"${ai_price:.4f}")
            st.metric(label="Computing Time", value=f"{ai_time:.8f} Seconds")
        
        # Traditional MC Simulation
        with col1:
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.info("⏳ CPU is computing (approx. 20-60s depending on CPU)...")
            
            mc_start_time = time.time()
            mc_price = run_traditional_mc_with_progress(user_sigma, progress_bar, status_text)
            mc_time = time.time() - mc_start_time
            
            progress_bar.empty()
            status_text.empty()
            
        with mc_container.container():
            st.success("✅ Calculation Completed!")
            st.metric(label="Expected PV (USD)", value=f"${mc_price:.4f}")
            st.metric(label="Computing Time", value=f"{mc_time:.4f} Seconds")

        # Performance Report
        st.markdown("---")
        st.markdown("### 🏆 Performance Battle Report")
        
        speedup_multiplier = int(mc_time / ai_time) if ai_time > 0 else 0
        abs_error = abs(mc_price - ai_price)
        pct_error = (abs_error / mc_price) * 100 if mc_price != 0 else 0
        
        rep_col1, rep_col2 = st.columns(2)
        with rep_col1:
            st.metric(label="🚀 AI Speedup Multiplier", value=f"{speedup_multiplier:,}x Faster", help="How many times faster the AI is compared to Monte-Carlo.")
        with rep_col2:
            st.metric(label="🎯 Relative Pricing Error", value=f"{pct_error:.4f}%", delta=f"Abs Diff: ${abs_error:.4f}", delta_color="off", help="The percentage deviation between the AI model and the traditional Monte-Carlo calculation.")