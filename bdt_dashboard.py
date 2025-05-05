# BDT Exchange Rate Intelligence Dashboard (Streamlit Version)
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import base64
from io import BytesIO
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="BDT Exchange Rate Dashboard",
    page_icon="💱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #0D47A1;
        margin-top: 2rem;
    }
    .dashboard-container {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# --- App Header ---
st.markdown("<h1 class='main-header'>BDT Exchange Rate Intelligence Dashboard</h1>", unsafe_allow_html=True)
st.markdown("An interactive tool for analyzing and forecasting USD to BDT exchange rates, simulating economic scenarios, and supporting decision-making.")

# --- Session State Management ---
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

if 'forecast_generated' not in st.session_state:
    st.session_state.forecast_generated = False

if 'simulation_run' not in st.session_state:
    st.session_state.simulation_run = False

# --- Sidebar Configuration ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f9/Flag_of_Bangladesh.svg/320px-Flag_of_Bangladesh.svg.png", width=150)
st.sidebar.title("Dashboard Controls")

# --- Data Source Selection ---
data_source = st.sidebar.radio(
    "Select Data Source",
    ["Simulated Data", "Upload CSV"]
)

# --- Date Range Selection ---
st.sidebar.markdown("### Date Range")
default_start = datetime.today() - timedelta(days=365)
default_end = datetime.today()

start_date = st.sidebar.date_input("Start Date", default_start)
end_date = st.sidebar.date_input("End Date", default_end)

if start_date > end_date:
    st.sidebar.error("Error: End date must be after start date.")
    st.stop()

# --- Advanced Options ---
st.sidebar.markdown("### Advanced Options")

# Conditionally check if Prophet is available
try:
    from prophet import Prophet
    has_prophet = True
    forecast_periods = st.sidebar.slider("Forecast Horizon (Days)", 7, 90, 30)
except ImportError:
    has_prophet = False
    st.sidebar.warning("⚠️ Prophet library not found. Forecasting disabled.")

# --- Export Options ---
st.sidebar.markdown("### Export Options")
export_format = st.sidebar.selectbox("Export Format", ["CSV", "Excel", "JSON"])

# --- Data Functions ---
def get_simulated_data():
    """Generate simulated exchange rate data"""
    dates = pd.date_range(start=start_date, end=end_date)
    np.random.seed(42)  # For reproducibility
    
    # Base rate with trend and seasonal components
    base_trend = np.linspace(0, 2, len(dates))
    seasonal = 0.5 * np.sin(np.linspace(0, 4*np.pi, len(dates)))
    noise = np.random.normal(0, 0.2, len(dates))
    
    rates = 105 + base_trend + seasonal + np.cumsum(noise) * 0.1
    
    df = pd.DataFrame({
        "ds": dates,
        "y": rates,
        "Volume": np.random.randint(50000, 200000, size=len(dates))
    })
    
    return df

def process_uploaded_data(uploaded_file):
    """Process uploaded CSV file"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Check for required columns
        if 'ds' not in df.columns or 'y' not in df.columns:
            # Check if there are date and value columns to rename
            date_cols = [col for col in df.columns if 'date' in col.lower()]
            value_cols = [col for col in df.columns if 'rate' in col.lower() or 'price' in col.lower() or 'value' in col.lower()]
            
            if date_cols and value_cols:
                df = df.rename(columns={date_cols[0]: 'ds', value_cols[0]: 'y'})
            else:
                # Assume first column is date and second is value
                df = df.rename(columns={df.columns[0]: 'ds', df.columns[1]: 'y'})
                
        # Ensure date column is datetime
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Ensure y is numeric
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        
        # Filter by date range
        df = df[(df['ds'].dt.date >= start_date) & (df['ds'].dt.date <= end_date)]
        
        return df
    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")
        return None

# --- Data Loading Section ---
st.markdown("<h2 class='sub-header'>Data Source</h2>", unsafe_allow_html=True)

if data_source == "Simulated Data":
    if st.button("Load Simulated Data"):
        with st.spinner("Generating simulated exchange rate data..."):
            df = get_simulated_data()
            st.session_state.df = df
            st.session_state.data_loaded = True
            time.sleep(1)  # Small delay for UX
        st.success("✅ Simulated data generated successfully!")

elif data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload Exchange Rate CSV", type=['csv'])
    if uploaded_file is not None:
        with st.spinner("Processing uploaded data..."):
            df = process_uploaded_data(uploaded_file)
            if df is not None:
                st.session_state.df = df
                st.session_state.data_loaded = True
                time.sleep(1)
                st.success("✅ Data uploaded and processed successfully!")
            else:
                st.error("❌ Failed to process the uploaded file.")

# --- Data Exploration Section ---
if st.session_state.data_loaded:
    st.markdown("<h2 class='sub-header'>Data Exploration</h2>", unsafe_allow_html=True)
    
    # Summary Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Average Rate", f"{st.session_state.df['y'].mean():.2f}")
    
    with col2:
        st.metric("Min Rate", f"{st.session_state.df['y'].min():.2f}")
    
    with col3:
        st.metric("Max Rate", f"{st.session_state.df['y'].max():.2f}")
    
    with col4:
        start_val = st.session_state.df['y'].iloc[0]
        end_val = st.session_state.df['y'].iloc[-1]
        pct_change = ((end_val - start_val) / start_val) * 100
        st.metric("Overall Change", f"{pct_change:.2f}%", delta=f"{end_val - start_val:.2f}")
    
    # Raw Data Table
    with st.expander("📊 View Raw Data"):
        st.dataframe(st.session_state.df)
        
        # Export functionality
        if st.button("Export Data"):
            if export_format == "CSV":
                csv = st.session_state.df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="exchange_rate_data.csv">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)
            elif export_format == "Excel":
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    st.session_state.df.to_excel(writer, sheet_name='ExchangeRates', index=False)
                b64 = base64.b64encode(output.getvalue()).decode()
                href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="exchange_rate_data.xlsx">Download Excel File</a>'
                st.markdown(href, unsafe_allow_html=True)
            elif export_format == "JSON":
                json_str = st.session_state.df.to_json(orient='records', date_format='iso')
                b64 = base64.b64encode(json_str.encode()).decode()
                href = f'<a href="data:file/json;base64,{b64}" download="exchange_rate_data.json">Download JSON File</a>'
                st.markdown(href, unsafe_allow_html=True)

# You can add more sections for visualization, forecasting, and simulation below this block.
