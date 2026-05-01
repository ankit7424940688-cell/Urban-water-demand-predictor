import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

# Ignore statistical warnings for cleaner output
warnings.filterwarnings('ignore')

# --- 1. App Configuration ---
st.set_page_config(page_title="Urban Water Analytics", page_icon="💧", layout="wide")
st.title("💧 Smart Urban Water Demand Dashboard")
st.write("Predict future water consumption and analyze historical trends using Machine Learning.")

# --- 2. Load and Prepare the Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("urban_water_demand.csv")
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date') # Crucial for time-series
        
        # Shift dates to the present day so the dashboard always looks current
        time_difference = pd.Timestamp.now().normalize() - df['date'].max()
        df['date'] = df['date'] + time_difference
        
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Error: Dataset not found. Ensure 'urban_water_demand.csv' is in the same directory.")
    st.stop()

# --- 3. Train Models ---

# A. Random Forest for Weather-based Prediction
X = df[['temperature', 'rainfall', 'population']]
y = df['water_demand']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

# B. SARIMAX for Time-Series Forecasting with Weather Variables (Exogenous)
@st.cache_resource
def train_sarimax(data):
    ts_data = data[['date', 'water_demand']].set_index('date')
    
    # FIX: Set the index for exog_data so it aligns with ts_data
    exog_data = data[['date', 'temperature', 'rainfall']].set_index('date') 
    
    # SARIMAX with a 7-day weekly seasonal pattern
    model = SARIMAX(ts_data, exog=exog_data, order=(2, 1, 2), seasonal_order=(1, 1, 1, 7)) 
    model_fit = model.fit(disp=False)
    
    return model_fit, ts_data

sarimax_model, ts_data = train_sarimax(df)

# --- 4. Main Dashboard Layout ---
left_col, right_col = st.columns([1, 1.5], gap="large")

with left_col:
    st.header("🔮 Demand Predictor")
    st.info(f"**Random Forest Accuracy:** MAE: {rf_mae:.2f} | R² Score: {rf_r2:.2f}")
    
    tab1, tab2, tab3 = st.tabs(["🌍 Live API", "🎛️ Manual", "📅 SARIMAX Forecast"])

    # --- Live API Tab ---
    with tab1:
        st.write("Fetch real-time weather data for prediction.")
        
        # Pre-filled API Key
        api_key = st.text_input("OpenWeatherMap API Key", value="ed10afe74444d33e343b1cdf38059972", type="password")
        city_name = st.text_input("City Name", placeholder="e.g., Jaipur")
        live_population = st.number_input("Estimated Population", value=int(df['population'].mean()))
        
        if st.button("Fetch & Predict", type="primary", use_container_width=True):
            if not api_key or not city_name:
                st.warning("⚠️ Please enter an API Key and City Name.")
            else:
                with st.spinner(f"Fetching weather for {city_name}..."):
                    url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}&units=metric"
                    response = requests.get(url)
                    if response.status_code == 200:
                        data = response.json()
                        live_temp = data["main"]["temp"]
                        live_rain = data.get("rain", {}).get("1h", 0.0) 
                        st.success(f"**Current Weather:** {live_temp}°C, {live_rain}mm Rain")
                        
                        input_data = pd.DataFrame({'temperature': [live_temp], 'rainfall': [live_rain], 'population': [live_population]})
                        live_prediction = rf_model.predict(input_data)[0]
                        st.metric(label=f"Predicted Demand ({city_name.title()})", value=f"{live_prediction:.2f} Units")
                    else:
                        st.error("❌ City not found or invalid API key.")

    # --- Manual Tab ---
    with tab2:
        st.write("Simulate single-day custom weather scenarios.")
        temperature = st.slider("Temperature (°C)", float(df['temperature'].min()), float(df['temperature'].max()), float(df['temperature'].mean()))
        rainfall = st.slider("Rainfall (mm)", float(df['rainfall'].min()), float(df['rainfall'].max()), float(df['rainfall'].mean()))
        population = st.number_input("Population", value=int(df['population'].mean()))

        if st.button("Calculate Manual Demand", use_container_width=True):
            input_data = pd.DataFrame({'temperature': [temperature], 'rainfall': [rainfall], 'population': [population]})
            manual_prediction = rf_model.predict(input_data)[0] 
            st.metric(label="Predicted Daily Demand", value=f"{manual_prediction:.2f} Units")

    # --- Forecasting Tab (SARIMAX) ---
    with tab3:
        st.write("Generate a time-based forecast using SARIMAX with simulated future weather.")
        forecast_days = st.slider("Days to forecast", 7, 60, 30)
        
        st.markdown("##### 🌤️ Simulated Future Weather")
        sim_temp = st.slider("Average Future Temp (°C)", float(df['temperature'].min()), float(df['temperature'].max()), float(df['temperature'].mean()))
        sim_rain = st.slider("Average Future Rain (mm)", float(df['rainfall'].min()), float(df['rainfall'].max()), 0.0)
        
        if st.button("Generate Forecast", use_container_width=True):
            # FIX: Generate future dates FIRST
            future_dates = pd.date_range(start=ts_data.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
            
            # FIX: Apply those dates as the index to the future weather data
            future_exog = pd.DataFrame({
                'date': future_dates,
                'temperature': [sim_temp] * forecast_days, 
                'rainfall': [sim_rain] * forecast_days
            }).set_index('date')
            
            # Use get_forecast to unlock confidence intervals
            forecast_obj = sarimax_model.get_forecast(steps=forecast_days, exog=future_exog)
            forecast_values = forecast_obj.predicted_mean
            conf_int = forecast_obj.conf_int() # Upper and lower bounds
            
            st.success(f"Successfully generated a {forecast_days}-day forecast! Look at the chart.")
            
            # Store in session state for Plotly
            st.session_state['forecast_df'] = pd.DataFrame({
                'date': future_dates, 
                'forecast': forecast_values.values,
                'lower_bound': conf_int.iloc[:, 0].values,
                'upper_bound': conf_int.iloc[:, 1].values
            })

with right_col:
    st.header("📊 Historical & Future Analytics")
    
    # 1. Time Series Chart with Forecasting & Confidence Intervals
    st.subheader("Water Demand Timeline")
    fig_line = go.Figure()
    
    # Historical data
    fig_line.add_trace(go.Scatter(x=df['date'], y=df['water_demand'], mode='lines', name='Historical Demand', line=dict(color='blue')))
    
    # Forecast data & Confidence Band
    if 'forecast_df' in st.session_state:
        f_df = st.session_state['forecast_df']
        
        # Draw the shaded confidence interval
        fig_line.add_trace(go.Scatter(
            x=pd.concat([f_df['date'], f_df['date'][::-1]]),
            y=pd.concat([f_df['upper_bound'], f_df['lower_bound'][::-1]]),
            fill='toself',
            fillcolor='rgba(255, 165, 0, 0.2)', # Transparent orange
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name='95% Confidence Interval'
        ))
        
        # Draw the main forecast line
        fig_line.add_trace(go.Scatter(x=f_df['date'], y=f_df['forecast'], mode='lines', name='SARIMAX Forecast', line=dict(color='orange', dash='dash', width=2)))
        
    fig_line.update_layout(template="plotly_white", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    st.plotly_chart(fig_line, use_container_width=True)
    
    # 2. Scatter Plot: Temperature vs. Demand
    st.subheader("Weather Impact on Demand")
    fig_scatter = px.scatter(df, x='temperature', y='water_demand', color='rainfall', size='population',
                             labels={'temperature': 'Temperature (°C)', 'water_demand': 'Demand', 'rainfall': 'Rainfall (mm)'},
                             color_continuous_scale=px.colors.sequential.Blues, template="plotly_white")
    st.plotly_chart(fig_scatter, use_container_width=True)

# --- 5. Raw Data Expander ---
with st.expander("📂 View Raw Dataset & Feature Importance"):
    col_a, col_b = st.columns(2)
    with col_a:
        st.dataframe(df.head(10), use_container_width=True)
    with col_b:
        importances = rf_model.feature_importances_
        importance_df = pd.DataFrame({'Feature': ['Temperature', 'Rainfall', 'Population'], 'Importance': importances})
        fig_bar = px.bar(importance_df, x='Importance', y='Feature', orientation='h', title="Random Forest Feature Importance")
        st.plotly_chart(fig_bar, use_container_width=True)