deployment link="https://urban-water-demand-predictor-bappvqngve2arw7e68owats.streamlit.app"

## ✨ Features

- **Live Weather Integration:** Fetches real-time temperature and rainfall data via the OpenWeatherMap API to predict immediate water demand.
- **Advanced Predictive Modeling:** Utilizes a **Random Forest Regressor** to determine water demand based on complex, non-linear weather and population patterns.
- **Time-Series Forecasting (SARIMAX):** Projects water demand up to 60 days into the future using a SARIMAX model that accounts for 7-day weekly seasonality and simulated future weather conditions (exogenous variables).
- **Confidence Intervals:** Displays a shaded 95% confidence interval cone around future forecasts to account for statistical volatility.
- **Interactive Analytics:** Features an enterprise-grade dashboard using Plotly to visualize historical demand timelines and multivariate weather impact (Temperature vs. Demand vs. Rainfall vs. Population).

## 🛠 Tech Stack

- **Frontend/UI:** [Streamlit](https://streamlit.io/)
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn (Random Forest)
- **Time-Series Forecasting:** Statsmodels (SARIMAX)
- **Data Visualization:** Plotly Express, Plotly Graph Objects
- **External APIs:** Requests, OpenWeatherMap API

## 🚀 Installation

To run this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/urban_water_demand.git](https://github.com/yourusername/urban_water_demand.git)
   cd urban_water_demand
Create a virtual environment (optional but recommended):

Bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
Install the required dependencies:

Bash
pip install -r requirements.txt
💻 Usage
Start the Streamlit application by running the following command in your terminal:

Bash
streamlit run web.py
The application will open automatically in your default web browser at http://localhost:8501.

⚙️ Configuration (API Keys)
To use the "Live API" prediction feature, you will need an OpenWeatherMap API key.

Create a free account at OpenWeatherMap.

Generate an API key.

For local use, you can type the key directly into the app interface.

For Cloud Deployment: If deploying to Streamlit Community Cloud, navigate to your App Settings > Secrets, and add your key:

Ini, TOML
OPENWEATHER_KEY = "your_api_key_here"
📁 Project Structure
Plaintext
urban_water_demand/
│
├── web.py                     # Main Streamlit application file
├── urban_water_demand.csv     # Historical dataset (dates, temp, rain, population, demand)
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
