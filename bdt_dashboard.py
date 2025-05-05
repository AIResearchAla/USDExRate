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
        color: #1. **{sorted_factors[0]}** - The most significant factor, contributing {abs(sorted_impacts[0]):.2f} points to the exchange rate ({("weakening" if sorted_impacts[0] > 0 else "strengthening")} the BDT)
        
        2. **{sorted_factors[1]}** - The second most impactful factor, contributing {abs(sorted_impacts[1]):.2f} points ({("weakening" if sorted_impacts[1] > 0 else "strengthening")} the BDT)
        
        The combined effect of all factors suggests a {"bearish" if delta > 0 else "bullish"} outlook for the BDT in this scenario.
        """
        
        st.markdown(analysis_text)
        
        # Strategy recommendations
        st.subheader("Strategic Recommendations")
        
        if delta > 0:  # BDT weakening
            st.markdown("""
            **For Importers:**
            - Consider hedging currency risk through forward contracts
            - Accelerate USD payments while rates are favorable
            - Diversify supplier base to reduce USD dependency
            
            **For Exporters:**
            - Delay USD conversions where possible
            - Negotiate contracts in USD where feasible
            - Focus on markets with stronger currencies
            
            **For Investors:**
            - Consider USD-denominated investments
            - Evaluate BDT-denominated assets for potential depreciation impact
            - Monitor central bank interventions closely
            """)
        else:  # BDT strengthening
            st.markdown("""
            **For Importers:**
            - Delay USD purchases where possible
            - Negotiate longer payment terms with suppliers
            - Consider increasing inventory while exchange rates are favorable
            
            **For Exporters:**
            - Hedge future receivables
            - Consider pricing strategies to maintain competitiveness
            - Accelerate repatriation of foreign earnings
            
            **For Investors:**
            - Consider increasing exposure to BDT-denominated assets
            - Evaluate USD-denominated liabilities for potential conversion
            - Monitor for potential central bank actions to limit appreciation
            """)

# --- Correlation Analysis Section ---
if st.session_state.data_loaded:
    st.markdown("<h2 class='sub-header'>Economic Indicator Correlation</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    This section examines correlations between exchange rates and key economic indicators.
    For demonstration purposes, we're using simulated data for these indicators.
    """)
    
    # Generate simulated economic indicators data
    def generate_indicators():
        dates = st.session_state.df['ds'].tolist()
        n = len(dates)
        
        # Base exchange rate
        exchange_rate = st.session_state.df['y'].values
        
        # Generate correlated indicators
        np.random.seed(42)
        
        # Inflation - moderate positive correlation
        inflation = 5 + 0.3 * exchange_rate + np.random.normal(0, 1, n)
        
        # Interest rate - weak positive correlation
        interest_rate = 3 + 0.15 * exchange_rate + np.random.normal(0, 0.5, n)
        
        # GDP growth - weak negative correlation
        gdp_growth = 6 - 0.1 * exchange_rate + np.random.normal(0, 0.8, n)
        
        # Trade deficit - strong positive correlation
        trade_deficit = -2 + 0.5 * exchange_rate + np.random.normal(0, 1.2, n)
        
        # Create dataframe
        indicators_df = pd.DataFrame({
            'Date': dates,
            'Exchange_Rate': exchange_rate,
            'Inflation': inflation,
            'Interest_Rate': interest_rate, 
            'GDP_Growth': gdp_growth,
            'Trade_Deficit': trade_deficit
        })
        
        return indicators_df
    
    # Generate indicators if not already in session state
    if 'indicators_df' not in st.session_state:
        st.session_state.indicators_df = generate_indicators()
    
    # Display correlation matrix
    indicators_df = st.session_state.indicators_df
    
    # Calculate correlations
    corr_matrix = indicators_df[['Exchange_Rate', 'Inflation', 'Interest_Rate', 'GDP_Growth', 'Trade_Deficit']].corr()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Correlation Matrix")
        # Format the correlation matrix for display
        st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'))
        
        st.markdown("""
        **Interpretation:**
        - Values close to 1 indicate strong positive correlation
        - Values close to -1 indicate strong negative correlation
        - Values close to 0 indicate little to no correlation
        """)
    
    with col2:
        st.markdown("### Correlation Heatmap")
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title="Economic Indicators Correlation Heatmap"
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Scatter plots
    st.subheader("Exchange Rate vs. Economic Indicators")
    
    # Select indicator for plot
    selected_indicator = st.selectbox(
        "Select Economic Indicator",
        ['Inflation', 'Interest_Rate', 'GDP_Growth', 'Trade_Deficit']
    )
    
    # Create scatter plot
    fig = px.scatter(
        indicators_df,
        x='Exchange_Rate',
        y=selected_indicator,
        trendline="ols",
        labels={
            'Exchange_Rate': 'Exchange Rate (BDT/USD)',
            selected_indicator: selected_indicator.replace('_', ' ')
        },
        title=f"Exchange Rate vs. {selected_indicator.replace('_', ' ')}",
        template='plotly_white'
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    correlation = corr_matrix.loc['Exchange_Rate', selected_indicator]
    
    if abs(correlation) > 0.7:
        strength = "strong"
    elif abs(correlation) > 0.3:
        strength = "moderate"
    else:
        strength = "weak"
    
    direction = "positive" if correlation > 0 else "negative"
    
    st.markdown(f"""
    ### Analysis of {selected_indicator.replace('_', ' ')} Correlation
    
    The data shows a **{strength} {direction} correlation** ({correlation:.2f}) between the USD/BDT exchange rate and {selected_indicator.replace('_', ' ')}.
    
    **What this means:**
    """)
    
    # Customized analysis based on selected indicator
    if selected_indicator == 'Inflation':
        st.markdown("""
        - Higher inflation in Bangladesh tends to weaken the BDT against the USD
        - This aligns with economic theory: higher inflation erodes a currency's purchasing power
        - Central bank policy aimed at controlling inflation could help stabilize the exchange rate
        """)
    elif selected_indicator == 'Interest_Rate':
        st.markdown("""
        - Higher interest rates in Bangladesh appear to correlate with a weaker BDT
        - This may indicate that interest rates are being raised reactively to combat currency weakness
        - The central bank may need to consider more aggressive interest rate policies to effectively strengthen the currency
        """)
    elif selected_indicator == 'GDP_Growth':
        st.markdown("""
        - Stronger GDP growth corresponds with a stronger BDT against the USD
        - This suggests economic growth supports currency strength
        - Policies that promote sustainable economic growth could help stabilize the exchange rate
        """)
    elif selected_indicator == 'Trade_Deficit':
        st.markdown("""
        - Larger trade deficits correlate with a weaker BDT
        - This aligns with economic theory: countries importing more than they export often experience currency depreciation
        - Export promotion and import substitution policies could help reduce pressure on the exchange rate
        """)

# --- Policy Impact Section ---
if st.session_state.data_loaded:
    st.markdown("<h2 class='sub-header'>Policy Impact Analysis</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    This section simulates the impact of various policy decisions on the BDT/USD exchange rate.
    Adjust the parameters to see how different policy scenarios might affect the currency.
    """)
    
    policy_col1, policy_col2 = st.columns(2)
    
    with policy_col1:
        st.markdown("### Monetary Policy")
        interest_rate_change = st.slider("Interest Rate Change (percentage points)", -3.0, 5.0, 0.0, 0.25)
        forex_intervention = st.slider("Foreign Exchange Market Intervention ($B)", -5.0, 5.0, 0.0, 0.5)
        reserve_requirement = st.slider("Reserve Requirement Change (%)", -5.0, 5.0, 0.0, 0.5)
    
    with policy_col2:
        st.markdown("### Fiscal & Trade Policy")
        govt_spending = st.slider("Government Spending Change (% of GDP)", -3.0, 3.0, 0.0, 0.5)
        import_duties = st.slider("Import Duties/Tariffs Change (%)", -10.0, 10.0, 0.0, 1.0)
        export_incentives = st.slider("Export Incentives (% increase)", 0.0, 10.0, 0.0, 1.0)
    
    # Calculate policy impact
    if st.button("Calculate Policy Impact"):
        with st.spinner("Analyzing policy impacts..."):
            # Base rate
            current_rate = st.session_state.df['y'].iloc[-1]
            
            # Impact coefficients (these would ideally be based on econometric models)
            interest_impact = -0.8 * interest_rate_change  # Higher interest rates strengthen BDT
            intervention_impact = -0.3 * forex_intervention  # Positive intervention (buying USD) weakens BDT
            reserve_impact = -0.2 * reserve_requirement  # Higher reserve requirements strengthen BDT
            spending_impact = 0.4 * govt_spending  # Higher govt spending weakens BDT
            duties_impact = -0.15 * import_duties  # Higher import duties strengthen BDT
            export_impact = -0.25 * export_incentives  # Higher export incentives strengthen BDT
            
            # Total impact
            total_policy_impact = (
                interest_impact +
                intervention_impact +
                reserve_impact +
                spending_impact +
                duties_impact +
                export_impact
            )
            
            # Calculate new rate
            policy_rate = current_rate + total_policy_impact
            
            # Store in session state
            st.session_state.policy_impacts = {
                "Interest Rate": interest_impact,
                "Forex Intervention": intervention_impact,
                "Reserve Requirement": reserve_impact,
                "Government Spending": spending_impact,
                "Import Duties": duties_impact,
                "Export Incentives": export_impact
            }
            
            st.session_state.current_rate = current_rate
            st.session_state.policy_rate = policy_rate
            st.session_state.policy_calculated = True
            
            time.sleep(1)  # Small delay for UX
        
        st.success("✅ Policy impact analysis completed!")
    
    # Display policy impact results
    if 'policy_calculated' in st.session_state and st.session_state.policy_calculated:
        st.subheader("Policy Impact Results")
        
        # Results metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Current USD/BDT Rate",
                value=f"{st.session_state.current_rate:.2f}"
            )
        
        with col2:
            delta = st.session_state.policy_rate - st.session_state.current_rate
            delta_pct = (delta / st.session_state.current_rate) * 100
            st.metric(
                label="Projected USD/BDT Rate",
                value=f"{st.session_state.policy_rate:.2f}",
                delta=f"{delta:.2f} ({delta_pct:.1f}%)"
            )
        
        with col3:
            if delta > 0:
                st.metric(
                    label="BDT Strength",
                    value="Weakened",
                    delta=f"{abs(delta_pct):.1f}%",
                    delta_color="inverse"
                )
            else:
                st.metric(
                    label="BDT Strength",
                    value="Strengthened",
                    delta=f"{abs(delta_pct):.1f}%",
                    delta_color="normal"
                )
        
        # Policy impact visualization
        impacts = st.session_state.policy_impacts
        policies = list(impacts.keys())
        values = list(impacts.values())
        
        # Sort by absolute impact
        sorted_indices = np.argsort(np.abs(values))[::-1]
        sorted_policies = [policies[i] for i in sorted_indices]
        sorted_values = [values[i] for i in sorted_indices]
        
        # Create waterfall chart for policy impacts
        fig = go.Figure(go.Waterfall(
            name="Policy Impact",
            orientation="v",
            measure=["relative"] * len(sorted_policies),
            x=sorted_policies,
            y=sorted_values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "red"}},
            decreasing={"marker": {"color": "green"}}
        ))
        
        fig.update_layout(
            title="Impact of Each Policy on Exchange Rate",
            showlegend=False,
            height=400,
            yaxis_title="Impact on BDT/USD Rate",
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Policy recommendations
        st.subheader("Policy Recommendations")
        
        if delta > 0:  # BDT weakening
            st.markdown("""
            **To strengthen the BDT:**
            
            1. **Consider interest rate increases** - The model suggests this would have the strongest positive impact on currency value
            
            2. **Review import duties and tariffs** - Selective increases may help reduce trade deficit and support the currency
            
            3. **Enhance export incentives** - This could improve the trade balance and increase foreign currency inflows
            
            4. **Exercise caution with government spending** - Fiscal expansion appears to put pressure on the currency
            """)
        else:  # BDT strengthening
            st.markdown("""
            **If the goal is to prevent excessive BDT appreciation:**
            
            1. **Consider measured reductions in interest rates** - This could reduce pressure on the currency while supporting economic growth
            
            2. **Re-evaluate export incentives** - Current incentives may be creating unsustainable currency appreciation
            
            3. **Consider strategic foreign exchange interventions** - Building reserves may help moderate the appreciation
            
            4. **Maintain steady government spending** - This provides economic stability while moderating currency strength
            """)
        
        # Timeline for implementation
        st.subheader("Implementation Timeline")
        
        st.markdown("""
        **Short-term (0-3 months):**
        - Adjust interest rates
        - Implement forex market interventions if needed
        
        **Medium-term (3-12 months):**
        - Revise import duties and export incentives
        - Adjust reserve requirements
        
        **Long-term (1-3 years):**
        - Implement structural economic reforms
        - Develop more robust trade policies
        """)
        
        # Risk assessment
        st.markdown("""
        ### Risk Assessment
        
        | Policy Action | Potential Risks | Mitigation Strategies |
        |--------------|-----------------|----------------------|
        | Interest Rate Changes | Economic growth impact, market volatility | Gradual implementation, clear communication |
        | Forex Intervention | Reserve depletion, market distortion | Set clear limits, diversify intervention tactics |
        | Duty/Tariff Changes | Inflation impact, WTO compliance issues | Targeted approach, stakeholder consultation |
        | Export Incentives | Fiscal cost, trade partner reactions | Sustainable funding, WTO-compliant design |
        """)

# --- Footer ---
st.markdown("---")
st.markdown("✅ *Made for educational and business intelligence purposes*")
st.markdown("Developed for BDT Exchange Rate Analysis - © 2025")
E88E5;
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
    ["Simulated Data", "API Data (Coming Soon)", "Upload CSV"]
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

data_loading_container = st.container()

with data_loading_container:
    if data_source == "Simulated Data":
        if st.button("Load Simulated Data"):
            with st.spinner("Generating simulated exchange rate data..."):
                df = get_simulated_data()
                st.session_state.df = df
                st.session_state.data_loaded = True
                time.sleep(1)  # Small delay for UX
            st.success("✅ Simulated data generated successfully!")
            
    elif data_source == "API Data (Coming Soon)":
        st.info("🔜 This feature will be available in the next update.")
        st.warning("Using simulated data for now.")
        if st.button("Load Temporary Data"):
            with st.spinner("Generating temporary data..."):
                df = get_simulated_data()
                st.session_state.df = df
                st.session_state.data_loaded = True
                time.sleep(1)
            st.success("✅ Temporary data loaded successfully!")
            
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
    
    # --- Visualization Section ---
    st.markdown("<h2 class='sub-header'>Exchange Rate Visualization</h2>", unsafe_allow_html=True)
    
    # Chart type selection
    chart_type = st.radio("Select Chart Type", ["Line Chart", "Candlestick Chart", "Interactive Chart"], horizontal=True)
    
    if chart_type == "Line Chart":
        fig = px.line(
            st.session_state.df, 
            x='ds', 
            y='y',
            title='USD to BDT Exchange Rate Trend',
            labels={'ds': 'Date', 'y': 'Exchange Rate (BDT/USD)'},
            template='plotly_white'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
    elif chart_type == "Candlestick Chart":
        # For candlestick, we need to resample to get OHLC data
        # Create a copy of the dataframe to avoid modifying the original
        ohlc_df = st.session_state.df.copy()
        
        # Generate OHLC data (Open, High, Low, Close)
        # For simplicity, we'll add some noise to the original data to create these values
        np.random.seed(42)
        noise = np.random.normal(0, 0.1, len(ohlc_df))
        
        ohlc_df['open'] = ohlc_df['y'] - noise
        ohlc_df['high'] = ohlc_df[['y', 'open']].max(axis=1) + np.abs(noise) * 0.5
        ohlc_df['low'] = ohlc_df[['y', 'open']].min(axis=1) - np.abs(noise) * 0.5
        ohlc_df['close'] = ohlc_df['y']
        
        fig = go.Figure(data=[go.Candlestick(
            x=ohlc_df['ds'],
            open=ohlc_df['open'],
            high=ohlc_df['high'],
            low=ohlc_df['low'],
            close=ohlc_df['close']
        )])
        
        fig.update_layout(
            title='USD to BDT Exchange Rate (Candlestick)',
            xaxis_title='Date',
            yaxis_title='Exchange Rate (BDT/USD)',
            height=500,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    elif chart_type == "Interactive Chart":
        # Create a figure with range slider and selectors
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=st.session_state.df['ds'],
            y=st.session_state.df['y'],
            mode='lines',
            name='Exchange Rate',
            line=dict(color='#1E88E5', width=2)
        ))
        
        # Add range slider
        fig.update_layout(
            title='Interactive USD to BDT Exchange Rate Chart',
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            ),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # --- Trend Analysis ---
    st.markdown("<h2 class='sub-header'>Trend Analysis</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Calculate moving averages
        ma_period = st.slider("Moving Average Period (Days)", 5, 60, 20)
        st.session_state.df['MA'] = st.session_state.df['y'].rolling(window=ma_period).mean()
        
        # Plot with moving average
        fig = px.line(st.session_state.df, x='ds', y=['y', 'MA'], 
                      title=f'Exchange Rate with {ma_period}-Day Moving Average',
                      labels={'ds': 'Date', 'value': 'Exchange Rate (BDT/USD)', 'variable': 'Series'},
                      color_discrete_map={'y': '#1E88E5', 'MA': '#FF5722'})
        
        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Calculate percent change
        period = st.selectbox("Select Period for Rate of Change", 
                              ["Daily", "Weekly", "Monthly"], index=1)
        
        if period == "Daily":
            st.session_state.df['pct_change'] = st.session_state.df['y'].pct_change() * 100
            title = 'Daily Percent Change in Exchange Rate'
        elif period == "Weekly":
            st.session_state.df['pct_change'] = st.session_state.df['y'].pct_change(7) * 100
            title = 'Weekly Percent Change in Exchange Rate'
        else:
            st.session_state.df['pct_change'] = st.session_state.df['y'].pct_change(30) * 100
            title = 'Monthly Percent Change in Exchange Rate'
        
        # Remove NaN values
        plot_df = st.session_state.df.dropna(subset=['pct_change'])
        
        # Plot percent change
        fig = px.bar(plot_df, x='ds', y='pct_change',
                     title=title,
                     labels={'ds': 'Date', 'pct_change': 'Percent Change (%)'},
                     color='pct_change',
                     color_continuous_scale=['red', 'lightgrey', 'green'],
                     range_color=[-plot_df['pct_change'].abs().max(), plot_df['pct_change'].abs().max()])
        
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # --- Forecasting Section (Conditional) ---
    if has_prophet:
        st.markdown("<h2 class='sub-header'>Exchange Rate Forecast</h2>", unsafe_allow_html=True)
        
        forecast_col1, forecast_col2 = st.columns([1, 3])
        
        with forecast_col1:
            st.markdown("### Forecast Settings")
            seasonality_mode = st.radio("Seasonality Mode", ["Additive", "Multiplicative"])
            include_components = st.checkbox("Show Forecast Components", value=True)
            
            if st.button("Generate Forecast"):
                with st.spinner("Generating forecast model..."):
                    # Create and fit Prophet model
                    m = Prophet(
                        seasonality_mode=seasonality_mode.lower(),
                        daily_seasonality=False,
                        weekly_seasonality=True,
                        yearly_seasonality=True
                    )
                    
                    # Add country-specific holidays if appropriate
                    # m.add_country_holidays(country_name='BD')
                    
                    m.fit(st.session_state.df[['ds', 'y']])
                    
                    # Make future dataframe
                    future = m.make_future_dataframe(periods=forecast_periods)
                    
                    # Generate forecast
                    forecast = m.predict(future)
                    
                    # Store forecast in session state
                    st.session_state.forecast = forecast
                    st.session_state.prophet_model = m
                    st.session_state.forecast_generated = True
                
                st.success("✅ Forecast generated successfully!")
        
        # Display forecast if generated
        if st.session_state.forecast_generated:
            with forecast_col2:
                # Plot forecast
                fig = go.Figure()
                
                # Add actual values
                fig.add_trace(go.Scatter(
                    x=st.session_state.df['ds'],
                    y=st.session_state.df['y'],
                    mode='markers',
                    name='Actual',
                    marker=dict(color='#1E88E5', size=4)
                ))
                
                # Add forecast line
                fig.add_trace(go.Scatter(
                    x=st.session_state.forecast['ds'],
                    y=st.session_state.forecast['yhat'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='#FF5722', width=2)
                ))
                
                # Add uncertainty intervals
                fig.add_trace(go.Scatter(
                    x=st.session_state.forecast['ds'],
                    y=st.session_state.forecast['yhat_upper'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=st.session_state.forecast['ds'],
                    y=st.session_state.forecast['yhat_lower'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(255, 87, 34, 0.2)',
                    name='95% Confidence Interval'
                ))
                
                fig.update_layout(
                    title='USD to BDT Exchange Rate Forecast',
                    xaxis_title='Date',
                    yaxis_title='Exchange Rate (BDT/USD)',
                    height=500,
                    template='plotly_white',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Show forecast components if requested
            if include_components:
                st.subheader("Forecast Components")
                
                try:
                    # Plot components
                    fig_comp = st.session_state.prophet_model.plot_components(st.session_state.forecast)
                    st.pyplot(fig_comp)
                except Exception as e:
                    st.error(f"Error plotting components: {e}")
                    st.warning("Components visualization requires matplotlib. Using alternative display.")
                    
                    # Alternative components visualization
                    components_tab1, components_tab2, components_tab3 = st.tabs(["Trend", "Seasonality", "Holidays"])
                    
                    with components_tab1:
                        fig = px.line(
                            st.session_state.forecast, 
                            x='ds', 
                            y='trend',
                            title='Trend Component',
                            labels={'ds': 'Date', 'trend': 'Trend'},
                            template='plotly_white'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with components_tab2:
                        # Plot yearly seasonality if present
                        if 'yearly' in st.session_state.forecast.columns:
                            fig = px.line(
                                st.session_state.forecast, 
                                x='ds', 
                                y='yearly',
                                title='Yearly Seasonality Component',
                                labels={'ds': 'Date', 'yearly': 'Yearly Effect'},
                                template='plotly_white'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Plot weekly seasonality if present
                        if 'weekly' in st.session_state.forecast.columns:
                            fig = px.line(
                                st.session_state.forecast, 
                                x='ds', 
                                y='weekly',
                                title='Weekly Seasonality Component',
                                labels={'ds': 'Date', 'weekly': 'Weekly Effect'},
                                template='plotly_white'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with components_tab3:
                        # Plot holidays if present
                        if 'holidays' in st.session_state.forecast.columns:
                            fig = px.line(
                                st.session_state.forecast, 
                                x='ds', 
                                y='holidays',
                                title='Holidays Component',
                                labels={'ds': 'Date', 'holidays': 'Holiday Effect'},
                                template='plotly_white'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No holiday components found in forecast.")
            
            # Forecast data table
            with st.expander("📊 View Forecast Data Table"):
                forecast_display = st.session_state.forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_periods)
                forecast_display.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
                st.dataframe(forecast_display)
                
                # Export forecast data
                if st.button("Export Forecast Data"):
                    csv = forecast_display.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="forecast_data.csv">Download Forecast CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)
    
    # --- Scenario Simulation Section ---
    st.markdown("<h2 class='sub-header'>Economic Scenario Simulation</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    This simulation allows you to model how different economic factors might affect the USD-BDT exchange rate.
    Adjust the sliders below to simulate various economic scenarios.
    """)
    
    # Factor sliders
    col1, col2 = st.columns(2)
    
    with col1:
        remittance = st.slider("Remittance Growth (%)", -30, 50, 0)
        export_growth = st.slider("Export Growth (%)", -30, 50, 0)
        foreign_reserves = st.slider("Foreign Reserves Change (%)", -30, 50, 0)
    
    with col2:
        import_growth = st.slider("Import Growth (%)", -30, 50, 0)
        inflation = st.slider("Inflation Rate (%)", 0, 20, 6)
        interest_rate_diff = st.slider("Interest Rate Differential (% points)", -5, 10, 0)
    
    # Simulate button
    if st.button("Run Simulation"):
        with st.spinner("Running economic scenario simulation..."):
            # Get base rate
            base_rate = st.session_state.df['y'].iloc[-1]
            
            # Impact calculation logic
            # These are illustrative weights - you would refine these based on economic research
            remittance_impact = -0.05 * remittance  # Negative sign because higher remittance strengthens BDT
            export_impact = -0.04 * export_growth  # Higher exports strengthen BDT
            reserves_impact = -0.07 * foreign_reserves  # Higher reserves strengthen BDT
            import_impact = 0.06 * import_growth  # Higher imports weaken BDT
            inflation_impact = 0.08 * inflation  # Higher inflation weakens BDT
            interest_rate_impact = -0.10 * interest_rate_diff  # Higher interest rates strengthen BDT
            
            # Total impact calculation
            total_impact = (
                remittance_impact + 
                export_impact + 
                reserves_impact + 
                import_impact + 
                inflation_impact + 
                interest_rate_impact
            )
            
            # Simulated rate
            simulated_rate = base_rate + total_impact
            
            # Store in session state
            st.session_state.base_rate = base_rate
            st.session_state.simulated_rate = simulated_rate
            st.session_state.factor_impacts = {
                "Remittance": remittance_impact,
                "Exports": export_impact,
                "Foreign Reserves": reserves_impact,
                "Imports": import_impact,
                "Inflation": inflation_impact,
                "Interest Rate Differential": interest_rate_impact
            }
            
            st.session_state.simulation_run = True
            
            time.sleep(1)  # Small delay for UX
        
        st.success("✅ Simulation completed!")
    
    # Display simulation results
    if st.session_state.simulation_run:
        st.subheader("Simulation Results")
        
        # Metrics display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Current USD/BDT Rate",
                value=f"{st.session_state.base_rate:.2f}"
            )
        
        with col2:
            delta = st.session_state.simulated_rate - st.session_state.base_rate
            delta_pct = (delta / st.session_state.base_rate) * 100
            st.metric(
                label="Simulated USD/BDT Rate",
                value=f"{st.session_state.simulated_rate:.2f}",
                delta=f"{delta:.2f} ({delta_pct:.1f}%)"
            )
        
        with col3:
            if delta > 0:
                st.metric(
                    label="BDT Strength",
                    value="Weakened",
                    delta=f"{abs(delta_pct):.1f}%",
                    delta_color="inverse"
                )
            else:
                st.metric(
                    label="BDT Strength",
                    value="Strengthened",
                    delta=f"{abs(delta_pct):.1f}%",
                    delta_color="normal"
                )
        
        # Factor impact visualization
        st.subheader("Factor Impact Analysis")
        
        # Prepare data for waterfall chart
        factors = list(st.session_state.factor_impacts.keys())
        impacts = list(st.session_state.factor_impacts.values())
        
        # Sort by absolute impact
        sorted_indices = np.argsort(np.abs(impacts))[::-1]
        sorted_factors = [factors[i] for i in sorted_indices]
        sorted_impacts = [impacts[i] for i in sorted_indices]
        
        # Create bar chart
        fig = go.Figure()
        
        # Add bars for each factor
        for i, (factor, impact) in enumerate(zip(sorted_factors, sorted_impacts)):
            fig.add_trace(go.Bar(
                x=[factor],
                y=[impact],
                name=factor,
                marker_color='red' if impact > 0 else 'green'
            ))
        
        fig.update_layout(
            title='Impact of Economic Factors on Exchange Rate',
            yaxis_title='Impact on BDT/USD Rate',
            barmode='relative',
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation
        st.markdown("""
        ### Interpretation Guide
        
        - **Red bars** represent factors that **weaken** the BDT against USD
        - **Green bars** represent factors that **strengthen** the BDT against USD
        - The height of each bar indicates the magnitude of impact
        
        ### Economic Analysis
        """)
        
        # Generate analysis text based on simulation results
        analysis_text = f"""
        Based on the simulated scenario, the BDT is expected to {"weaken" if delta > 0 else "strengthen"} against the USD by approximately {abs(delta_pct):.1f}%.
        
        **Key factors affecting the exchange rate:**
