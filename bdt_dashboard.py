# BDT Exchange Rate Intelligence Dashboard (Streamlit Version)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- Page Configuration ---
st.set_page_config(
    page_title="BDT Exchange Rate Dashboard",
    page_icon="💱",
    layout="wide"
)

# --- App Header ---
st.title("💱 BDT Exchange Rate Intelligence Dashboard")
st.markdown("An interactive tool for analyzing and forecasting USD to BDT exchange rates.")

# --- Sidebar Configuration ---
st.sidebar.header("Dashboard Controls")
st.sidebar.markdown("Configure the settings below to customize the dashboard.")

# --- Date Range Selection ---
st.sidebar.markdown("### Select Date Range")
default_start = datetime.today() - timedelta(days=365)
default_end = datetime.today()
start_date = st.sidebar.date_input("Start Date", default_start)
end_date = st.sidebar.date_input("End Date", default_end)

if start_date > end_date:
    st.sidebar.error("Error: End date must be after start date.")
    st.stop()

# --- Data Source Selection ---
data_source = st.sidebar.radio(
    "Select Data Source",
    ["Simulated Data", "Upload CSV"]
)

# --- Generate Simulated Data ---
def get_simulated_data():
    """Generate simulated exchange rate data."""
    dates = pd.date_range(start=start_date, end=end_date)
    np.random.seed(42)
    rates = 105 + np.cumsum(np.random.normal(0, 0.5, len(dates)))
    df = pd.DataFrame({"Date": dates, "Exchange Rate": rates})
    return df

# --- Load Data Based on User Selection ---
if data_source == "Simulated Data":
    st.sidebar.info("Using simulated data for this session.")
    data = get_simulated_data()
elif data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        data["Date"] = pd.to_datetime(data["Date"])
    else:
        st.warning("Please upload a CSV file to proceed.")
        st.stop()

# --- Data Exploration ---
st.subheader("📊 Data Exploration")
col1, col2, col3 = st.columns(3)
col1.metric("Start Date", data["Date"].min().strftime("%Y-%m-%d"))
col2.metric("End Date", data["Date"].max().strftime("%Y-%m-%d"))
col3.metric("Average Rate", f"{data['Exchange Rate'].mean():.2f} BDT/USD")

st.dataframe(data)

# --- Data Visualization ---
st.subheader("📈 Exchange Rate Visualization")
fig = px.line(data, x="Date", y="Exchange Rate", title="Exchange Rate Over Time", labels={"Exchange Rate": "BDT/USD"})
st.plotly_chart(fig, use_container_width=True)