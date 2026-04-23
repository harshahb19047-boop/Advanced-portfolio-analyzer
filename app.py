import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import yfinance as yf

st.set_page_config(page_title="Portfolio Analyzer Pro", layout="wide")

st.title("📊 Portfolio Optimization Dashboard (Hybrid Pro)")

RF_RATE = 0.065
TRADING_DAYS = 252

# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.header("⚙️ Configuration")

mode = st.sidebar.radio("Select Data Source", ["Dataset", "Live Market"])

budget = st.sidebar.number_input("💰 Investment Amount (₹)", 1000, 10000000, 100000)

risk_choice = st.sidebar.radio(
    "Risk Profile",
    ["Low Risk", "Medium Risk", "High Risk", "Auto (Best)"]
)

# -----------------------------
# LOAD DATA
# -----------------------------

if mode == "Dataset":
    prices = pd.read_csv("cleaned_stock_prices.csv", index_col="Date", parse_dates=True)
    st.caption("📌 Dataset model based on past 1-year historical performance")

else:
    stock_list = st.sidebar.multiselect(
        "Select NSE Stocks",
        ["TITAN", "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "LT", "SBIN", "ITC", "ASIANPAINT"]
    )

    if len(stock_list) == 0:
        st.warning("Please select stocks")
        st.stop()

    tickers = [s + ".NS" for s in stock_list]

    data = yf.download(tickers, period="1y")["Close"]
    prices = data.dropna()

# -----------------------------
# RETURNS
# -----------------------------

returns = prices.pct_change().dropna()
expected_returns = returns.mean() * TRADING_DAYS
cov_matrix = returns.cov() * TRADING_DAYS

# -----------------------------
# RESET SESSION STATE (IMPORTANT FIX)
# -----------------------------

if "num_assets" not in st.session_state:
    st.session_state.num_assets = len(prices.columns)

if st.session_state.num_assets != len(prices.columns):
    st.session_state.opt_weights = np.ones(len(prices.columns)) / len(prices.columns)
    st.session_state.num_assets = len(prices.columns)

# -----------------------------
# WEIGHTS INPUT
# -----------------------------

st.sidebar.subheader("📌 Weights")

weights = []
for stock in prices.columns:
    w = st.sidebar.slider(stock, 0.0, 1.0, 1.0/len(prices.columns))
    weights.append(w)

weights = np.array(weights)
weights = weights / np.sum(weights)

# -----------------------------
# PORTFOLIO FUNCTION
# -----------------------------

def portfolio_performance(weights):
    ret = np.sum(expected_returns.values * weights)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))
    sharpe = (ret - RF_RATE) / vol
    return ret, vol, sharpe

port_return, port_risk, sharpe = portfolio_performance(weights)

# -----------------------------
# OPTIMIZATION (FIXED)
# -----------------------------

def optimize_portfolio(risk_type):

    def objective(w):
        ret, vol, sharpe = portfolio_performance(w)

        if risk_type == "Low Risk":
            return vol
        elif risk_type == "High Risk":
            return -ret
        elif risk_type == "Medium Risk":
            return vol - ret
        else:
            return -sharpe

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0.05, 0.4) for _ in range(len(weights)))

    result = minimize(objective, weights, method='SLSQP',
                      bounds=bounds, constraints=constraints)

    # 🔥 STRICT ENFORCEMENT FIX
    w = result.x
    w = np.clip(w, 0.05, 0.4)
    w = w / np.sum(w)

    return w

# -----------------------------
# SESSION STATE
# -----------------------------

if "opt_weights" not in st.session_state:
    st.session_state.opt_weights = weights

# -----------------------------
# OPTIMIZE BUTTON
# -----------------------------

if st.button("🚀 Optimize My Portfolio"):
    st.session_state.opt_weights = optimize_portfolio(risk_choice)
    st.success("Optimal weights applied!")

final_weights = st.session_state.opt_weights

opt_return, opt_risk, opt_sharpe = portfolio_performance(final_weights)

# -----------------------------
# DISPLAY
# -----------------------------

st.subheader("📊 Your Portfolio")

c1, c2, c3 = st.columns(3)
c1.metric("Return", f"{port_return:.2%}")
c2.metric("Risk", f"{port_risk:.2%}")
c3.metric("Sharpe", f"{sharpe:.2f}")

st.subheader("🏆 Optimized Portfolio")

c1, c2, c3 = st.columns(3)
c1.metric("Return", f"{opt_return:.2%}")
c2.metric("Risk", f"{opt_risk:.2%}")
c3.metric("Sharpe", f"{opt_sharpe:.2f}")

# -----------------------------
# ALLOCATION
# -----------------------------

allocation = final_weights * budget

df_alloc = pd.DataFrame({
    "Stock": prices.columns,
    "Weight (%)": (final_weights * 100).round(2),
    "Investment (₹)": allocation.round(2)
})

st.subheader("💰 Optimal Allocation")
st.dataframe(df_alloc)

st.caption("⚠️ Diversification constraints applied (5%–40% per stock)")

# -----------------------------
# EFFICIENT FRONTIER (REAL CURVE)
# -----------------------------

target_returns = np.linspace(expected_returns.min(), expected_returns.max(), 50)
efficient_risks = []

for target in target_returns:

    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'eq', 'fun': lambda x: np.sum(expected_returns.values * x) - target}
    )

    bounds = tuple((0.05, 0.4) for _ in range(len(weights)))

    result = minimize(
        lambda w: np.sqrt(np.dot(w.T, np.dot(cov_matrix.values, w))),
        weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if result.success:
        efficient_risks.append(result.fun)
    else:
        efficient_risks.append(np.nan)

fig, ax = plt.subplots()

ax.plot(efficient_risks, target_returns, color='orange', label="Efficient Frontier")

# 🔴 Original
ax.scatter(port_risk, port_return, color='red', label="Your Portfolio")

# 🟢 Optimized
ax.scatter(opt_risk, opt_return, color='green', label="Optimized")

ax.set_xlabel("Risk")
ax.set_ylabel("Return")
ax.legend()

st.subheader("📈 Efficient Frontier")
st.pyplot(fig)

# -----------------------------
# AI SUGGESTION
# -----------------------------

st.subheader("🤖 AI Suggestion")

if sharpe < opt_sharpe:
    st.warning("Portfolio not optimal. Click 'Optimize My Portfolio' for better allocation.")
else:
    st.success("Your portfolio is already efficient.")