import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# --- 1. App Header & Description ---
st.title("💧 Urban Water Demand Predictor")
st.write("""
This app predicts the urban water demand based on temperature, rainfall, and population using a Linear Regression model.
""")

# --- 2. Load the Data ---
# We use @st.cache_data so the dataset isn't reloaded on every user interaction
@st.cache_data
def load_data():
    df = pd.read_csv("urban_water_demand.csv")
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Error: 'urban_water_demand.csv' not found. Please ensure it's in the same directory as app.py.")
    st.stop()

# Option to view the raw data
if st.checkbox("Show Raw Data"):
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))

# --- 3. Train the Model ---
# Extract features and target variable
X = df[['temperature', 'rainfall', 'population']]
y = df['water_demand']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Calculate Mean Absolute Error (MAE) for context
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
st.write(f"**Model Performance (MAE):** `{mae:.2f}`")

st.divider()

# --- 4. User Inputs for Prediction ---
st.header("🔮 Make a Prediction")
st.write("Adjust the sliders below to see how changes in weather and population affect water demand.")

# Create columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    temperature = st.slider(
        "Temperature (°C)", 
        min_value=float(df['temperature'].min()), 
        max_value=float(df['temperature'].max()), 
        value=float(df['temperature'].mean())
    )

with col2:
    rainfall = st.slider(
        "Rainfall (mm)", 
        min_value=float(df['rainfall'].min()), 
        max_value=float(df['rainfall'].max()), 
        value=float(df['rainfall'].mean())
    )

with col3:
    population = st.number_input(
        "Population", 
        min_value=int(df['population'].min()), 
        max_value=int(df['population'].max() * 1.5),  # Allow predicting for slightly higher future pop
        value=int(df['population'].mean())
    )

# --- 5. Display Prediction ---
# Predict using the inputs
if st.button("Calculate Demand", type="primary"):
    input_data = pd.DataFrame({
        'temperature': [temperature],
        'rainfall': [rainfall],
        'population': [population]
    })
    
    prediction = model.predict(input_data)[0]
    
    st.success(f"### Predicted Water Demand: **{prediction:.2f} units**")