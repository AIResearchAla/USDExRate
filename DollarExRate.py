# BDT Exchange Rate Intelligence Dashboard (Streamlit Version)

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import plotly.express as px
from prophet import Prophet

# --- Page Configuration ---
st.set_page_config(page_title="BDT Exchange Rate Dashboard", layout="wide")

# --- Sidebar Filters ---
st.sidebar.title("Filters")
start_date = st.sidebar.date_input("Start Date", datetime.today() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", datetime.today())

# --- Data Sources (Simulated + API Placeholder) ---
def get_exchange_data():
    dates = pd.date_range(start=start_date, end=end_date)
    np.random.seed(42)
    rates = 105 + np.cumsum(np.random.randn(len(dates))) * 0.2
    df = pd.DataFrame({"ds": dates, "y": rates})
    return df

# --- Forecasting Function ---
def forecast_exchange_rate(df):
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return forecast

# --- Load Data ---
st.info("Loading exchange rate data...")
df = get_exchange_data()
st.success("Data loaded successfully!")

# --- Show Raw Data ---
with st.expander("📄 Show Raw Data"):
    st.write(df.tail())

# --- Visualization ---
st.subheader("📈 BDT to USD Exchange Rate Over Time")
fig1 = px.line(df, x='ds', y='y', title='Exchange Rate Trend', labels={'ds': 'Date', 'y': 'Rate (BDT/USD)'})
st.plotly_chart(fig1, use_container_width=True)

# --- Forecast Section ---
st.subheader("🔮 30-Day Forecast Using Prophet")
forecast = forecast_exchange_rate(df)
fig2 = px.line(forecast, x='ds', y='yhat', title='Forecasted Exchange Rate', labels={'ds': 'Date', 'yhat': 'Forecasted Rate'})
st.plotly_chart(fig2, use_container_width=True)

# --- Scenario Simulation ---
st.subheader("🧪 Simulate Exchange Rate Under Economic Scenarios")
remittance = st.slider("Remittance Increase (%)", -20, 50, 0)
import_growth = st.slider("Import Growth (%)", -20, 50, 0)
inflation = st.slider("Inflation Rate (%)", 0, 20, 6)

# Simulation logic
base_rate = df['y'].iloc[-1]
impact = -remittance * 0.05 + import_growth * 0.07 + inflation * 0.03
simulated_rate = base_rate + impact
st.metric(label="💱 Simulated USD/BDT Rate", value=round(simulated_rate, 2))

# --- Footer ---
st.markdown("---")
st.markdown("✅ *Made for educational and business intelligence purposes.*")
