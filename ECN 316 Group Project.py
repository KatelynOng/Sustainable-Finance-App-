import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# =========================================================
# 1. Page Configuration & UI Styling
# =========================================================
st.set_page_config(page_title="ESG Portfolio Optimiser", layout="wide")

# Custom CSS for a professional "FinTech" look
st.markdown("""
    <style>
    .stApp { background-color: #fcfcfc; }
    [data-testid="stMetricContainer"] {
        background-color: #ffffff;
        border: 1px solid #e6e9ef;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] { background-color: #007bff !important; color: white !important; }
    </style>
    """, unsafe_allow_html=True)

# =========================================================
# 2. Finance & ESG Calculation Logic
# =========================================================

def var_covar(sigmas: np.ndarray, rho: float) -> np.ndarray:
    return np.array([
        [sigmas[0] ** 2, rho * sigmas[0] * sigmas[1]],
        [rho * sigmas[0] * sigmas[1], sigmas[1] ** 2]
    ])

def build_portfolio_grid(mu, sigma, rho, rf, esg_scores, gamma, lambda_esg, num_points):
    cov = var_covar(sigma, rho)
    weights = np.linspace(0, 1, num_points)
    rows = []
    for w1 in weights:
        w = np.array([w1, 1 - w1])
        exp_return = float(np.dot(mu, w))
        variance = float(np.dot(w, np.dot(cov, w)))
        std_dev = float(np.sqrt(max(variance, 0.0)))
        esg_score = float(np.dot(esg_scores, w))
        sharpe = (exp_return - rf) / std_dev if std_dev > 0 else 0
        utility = exp_return - 0.5 * gamma * variance + lambda_esg * esg_score
        rows.append({
            "Weight Asset 1": w1, "Weight Asset 2": 1 - w1,
            "Expected Return": exp_return, "Std Dev": std_dev,
            "ESG Score": esg_score, "Sharpe Ratio": sharpe, "Utility": utility
        })
    return pd.DataFrame(rows)

def invert_covariance(cov: np.ndarray) -> np.ndarray:
    cov = (cov + cov.T) / 2.0 + np.eye(cov.shape[0]) * 1e-10
    return np.linalg.pinv(cov)

def build_analytic_frontier(mu, cov, esg, rf, num_points=100):
    mu, esg, cov = np.asarray(mu), np.asarray(esg), np.asarray(cov)
    inv_cov = invert_covariance(cov)
    ones = np.ones(len(mu))
    
    A = float(ones @ inv_cov @ ones)
    B = float(ones @ inv_cov @ mu)
    C = float(mu @ inv_cov @ mu)
    D = max(A * C - B**2, 1e-12)
    
    # Min Var Portfolio
    w_gmv = (inv_cov @ ones) / A
    gmv_ret = float(w_gmv @ mu)
    gmv_std = float(np.sqrt(1/A))
    
    # Tangency
    excess = mu - rf
    w_tan = (inv_cov @ excess) / (ones @ inv_cov @ excess)
    tan_ret = float(w_tan @ mu)
    tan_std = float(np.sqrt(w_tan @ cov @ w_tan))
    
    # Frontier Curve
    target_rets = np.linspace(gmv_ret, max(mu.max(), tan_ret), num_points)
    frontier_stds = [np.sqrt((A * r**2 - 2*B*r + C) / D) for r in target_rets]
    
    return pd.DataFrame({"Expected Return": target_rets, "Std Dev": frontier_stds}), \
           {"Return": gmv_ret, "Std Dev": gmv_std}, \
           {"Return": tan_ret, "Std Dev": tan_std, "Weights": w_tan}

# =========================================================
# 3. Data Loading
# =========================================================

@st.cache_data
def load_firm_universe():
    # Attempt to load from current or /mnt path as per your original logic
    try:
        merged = pd.read_excel("CRSP_merged_overlap_only.xlsx", sheet_name="Merged Data")
        matrix_path = "CRSP_correlation_covariance_2010_2024.xlsx"
        corr = pd.read_excel(matrix_path, sheet_name="Correlation", index_col=0)
        
        # Filter and clean
        overlap = sorted(set(merged["ticker"].astype(str)) & set(corr.index.astype(str)))
        firms = merged[merged["ticker"].astype(str).isin(overlap)].groupby("ticker").first().reset_index()
        
        # Build Covariance: Vol * Corr * Vol
        vols = firms["annualized_volatility_std_dev"].values
        annual_cov = np.outer(vols, vols) * corr.loc[firms["ticker"], firms["ticker"]].values
        
        return firms, annual_cov
    except Exception as e:
        st.error(f"Error loading Excel files: {e}")
        return None, None

# =========================================================
# 4. Sidebar Inputs
# =========================================================

st.sidebar.title("🛠️ Optimiser Controls")

with st.sidebar.expander("Investor Preferences", expanded=True):
    lambda_esg = st.slider("ESG Preference (λ)", 0.0, 1.0, 0.30)
    gamma = st.slider("Risk Aversion (γ)", 0.1, 10.0, 3.0)
    rf_rate = st.number_input("Risk-free Rate (%)", 0.0, 10.0, 2.0) / 100

with st.sidebar.expander("2-Asset Scenario"):
    mu1 = st.number_input("Asset 1 Return (%)", 0.0, 20.0, 5.0) / 100
    esg1 = st.number_input("Asset 1 ESG Score", 0.0, 100.0, 35.0) / 100
    mu2 = st.number_input("Asset 2 Return (%)", 0.0, 20.0, 12.0) / 100
    esg2 = st.number_input("Asset 2 ESG Score", 0.0, 100.0, 80.0) / 100
    rho = st.slider("Correlation", -1.0, 1.0, -0.20)
    sig1, sig2 = 0.09, 0.20 # Volatilities fixed or also inputs

# =========================================================
# 5. Main Dashboard View
# =========================================================

st.title("📈 ESG Portfolio Optimisation Dashboard")

tab1, tab2 = st.tabs(["2-Asset Teaching Model", "Real-Stock Universe (Firm Level)"])

# --- TAB 1: 2-Asset Model ---
with tab1:
    df_all = build_portfolio_grid(np.array([mu1, mu2]), np.array([sig1, sig2]), rho, rf_rate, np.array([esg1, esg2]), gamma, lambda_esg, 500)
    
    # Identify key portfolios
    tan_idx = df_all["Sharpe Ratio"].idxmax()
    tan_pt = df_all.loc[tan_idx]
    
    # KPI Row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tangency Return", f"{tan_pt['Expected Return']:.2%}")
    c2.metric("Tangency Risk", f"{tan_pt['Std Dev']:.2%}")
    c3.metric("Sharpe Ratio", f"{tan_pt['Sharpe Ratio']:.3f}")
    c4.metric("Portfolio ESG", f"{tan_pt['ESG Score']*100:.1f}/100")

    # Interactive Plotly Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_all["Std Dev"]*100, y=df_all["Expected Return"]*100, mode='lines', name='Efficient Frontier', line=dict(color='#007bff', width=3)))
    fig.add_trace(go.Scatter(x=[tan_pt["Std Dev"]*100], y=[tan_pt["Expected Return"]*100], mode='markers', marker=dict(size=15, color='gold', symbol='star'), name='Tangency Portfolio'))
    
    # CML Line
    max_x = df_all["Std Dev"].max() * 1.2 * 100
    fig.add_trace(go.Scatter(x=[0, max_x], y=[rf_rate*100, rf_rate*100 + tan_pt['Sharpe Ratio']*max_x], mode='lines', name='CML', line=dict(dash='dash', color='gray')))
    
    fig.update_layout(title="2-Asset Efficient Frontier & CML", xaxis_title="Volatility (%)", yaxis_title="Return (%)", template="plotly_white", height=500)
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: Firm-Level Model ---
with tab2:
    firms, cov_matrix = load_firm_universe()
    if firms is not None:
        st.subheader(f"Analysing Universe of {len(firms)} Stocks")
        
        # ESG Screening
        min_esg = st.slider("Minimum Company ESG Score Filter", 0.0, 100.0, 50.0) / 100
        filtered_firms = firms[firms["valuescore"] >= min_esg]
        st.info(f"Firms passing screen: {len(filtered_firms)}")
        
        if len(filtered_firms) > 2:
            indices = filtered_firms.index
            sub_mu = filtered_firms["return_2024"].values
            sub_cov = cov_matrix[np.ix_(indices, indices)]
            sub_esg = filtered_firms["valuescore"].values
            
            frontier_df, gmv, tan = build_analytic_frontier(sub_mu, sub_cov, sub_esg, rf_rate)
            
            # KPI Row for Firms
            f1, f2, f3 = st.columns(3)
            f1.metric("ESG-Screened Tangency Return", f"{tan['Return']:.2%}")
            f2.metric("ESG-Screened Volatility", f"{tan['Std Dev']:.2%}")
            f3.metric("Sharpe", f"{(tan['Return']-rf_rate)/tan['Std Dev']:.3f}")

            # Plotly Frontier
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(x=frontier_df["Std Dev"]*100, y=frontier_df["Expected Return"]*100, mode='lines', name='Screened Frontier', line=dict(color='green')))
            fig_f.add_trace(go.Scatter(x=[tan["Std Dev"]*100], y=[tan["Return"]*100], mode='markers', marker=dict(size=12, color='darkgreen'), name='Tangency'))
            fig_f.update_layout(title="Firm-Level Efficient Frontier (Unconstrained)", xaxis_title="Risk (%)", yaxis_title="Return (%)", template="plotly_white")
            st.plotly_chart(fig_f, use_container_width=True)
            
            # Weights Table
            with st.expander("View Portfolio Weights (Top 10)"):
                weights_df = pd.DataFrame({"Ticker": filtered_firms["ticker"], "Weight": tan["Weights"]})
                st.dataframe(weights_df.sort_values("Weight", ascending=False).head(10).style.format({"Weight": "{:.2%}"}))
        else:
            st.warning("Not enough firms pass the ESG filter to build a frontier.")
    else:
        st.warning("Please ensure 'CRSP_merged_overlap_only.xlsx' and 'CRSP_correlation_covariance_2010_2024.xlsx' are in the directory.")
