import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from datetime import date, timedelta

# --- Configuration ---
MODEL_FILENAME = 'rf_air_quality_model.joblib'
METRICS_FILENAME = 'model_metrics.txt'
# Features used for training and user input (daily averages)
FEATURES = ['PT08.S1(CO)', 'C6H6(GT)', 'NOx(GT)', 'T', 'RH', 'AH']
FEATURE_DESCRIPTIONS = {
    'PT08.S1(CO)': 'PT08.S1 Sensor (CO) - TGS concentration',
    'C6H6(GT)': 'True Benzene concentration (mg/m³)',
    'NOx(GT)': 'True NOx concentration (ppb)',
    'T': 'Temperature (Avg. °C)',
    'RH': 'Relative Humidity (Avg. %)',
    'AH': 'Absolute Humidity (Avg. g/m³)'
}
# Define typical ranges for slider inputs (based on general data exploration)
FEATURE_RANGES = {
    'PT08.S1(CO)': (600.0, 2000.0),
    'C6H6(GT)': (0.0, 30.0),
    'NOx(GT)': (0.0, 1000.0),
    'T': (0.0, 40.0),
    'RH': (10.0, 80.0),
    'AH': (0.5, 2.5)
}

# --- Functions ---

@st.cache_resource
def load_model():
    """Loads the pre-trained model."""
    if not os.path.exists(MODEL_FILENAME):
        return None
    try:
        return joblib.load(MODEL_FILENAME)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_metrics():
    """Loads the model performance metrics."""
    metrics = {}
    if not os.path.exists(METRICS_FILENAME):
        return metrics
    
    try:
        with open(METRICS_FILENAME, 'r') as f:
            for line in f:
                try:
                    key, value = line.strip().split(': ')
                    metrics[key] = float(value)
                except ValueError:
                    continue
    except Exception:
        return {}
        
    return metrics

def make_prediction(model, input_data):
    """Makes a prediction using the loaded model."""
    input_df = pd.DataFrame([input_data], columns=FEATURES)
    prediction = model.predict(input_df)[0]
    return prediction

# --- Streamlit App ---

def app():
    st.set_page_config(
        page_title="Next-Day Air Quality Predictor (CO)",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Air Quality Predictor (Daily Avg.CO)")
    st.markdown("""
This application uses a **Random Forest Regressor** to predict the **Average Carbon Monoxide (CO) concentration** for the **next day**.
The prediction is based on the average air quality and weather parameters from sensors as input for a selected date.
""")
    st.markdown("---")

    model = load_model()
    metrics = load_metrics()
    
    if model is None:
        st.error(f"Model file '{MODEL_FILENAME}' not found or failed to load. Please ensure 'model.py' has been run successfully.")
        st.stop()

    # --- Sidebar for Model Metrics ---
    st.sidebar.header("Model Performance Metrics")
    if metrics:
        st.sidebar.markdown("**Random Forest Regressor**")
        st.sidebar.metric(label="Root Mean Squared Error (RMSE)", value=f"{metrics.get('RMSE', 0):.4f}")
        st.sidebar.metric(label="Mean Absolute Error (MAE)", value=f"{metrics.get('MAE', 0):.4f}")
        st.sidebar.metric(label="R-squared ($R^2$)", value=f"{metrics.get('R2', 0):.4f}")
    else:
        st.sidebar.info("Model metrics not loaded. Run 'model.py' to generate.")

    # --- User Input Section ---
    st.header("Define Day of Prediction")
    
    today = date.today()
    selected_date = st.date_input(
        "Select the date for which you are entering the **current average parameters**:",
        today,
        help="The prediction will be for the day immediately following this date."
    )
    
    predicted_date = selected_date + timedelta(days=1)
    st.markdown(f"**Predicted Date:** {predicted_date.strftime('%B %d, %Y')}")

    st.header(f"Input Average Parameters for {selected_date.strftime('%B %d, %Y')}")
    
    with st.expander("Parameter Descriptions and Input"):
        cols = st.columns(3)
        user_inputs = {}
        for i, feature in enumerate(FEATURES):
            col_idx = i % 3
            with cols[col_idx]:
                min_val, max_val = FEATURE_RANGES.get(feature, (0.0, 100.0))
                default_val = min_val + (max_val - min_val) * 0.4
                user_inputs[feature] = st.slider(
                    label=f"**{feature}**",
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    step=(max_val - min_val) / 100.0,
                    help=FEATURE_DESCRIPTIONS.get(feature, "No description available.")
                )

    # --- Prediction ---
    st.header(f"Predicted Air Quality for {predicted_date.strftime('%B %d, %Y')}")
    if st.button("Predict Next Day CO Concentration"):
        try:
            # Perform prediction
            prediction = make_prediction(model, user_inputs)
            
            # --- Visualization and Output ---
            st.subheader(f"Predicted Average CO for {predicted_date.strftime('%B %d, %Y')}:")
            
            # Color coding for air quality (simplified example)
            if prediction < 1.0:
                color = '#008000' # Green
                quality = 'Good'
            elif 1.0 <= prediction < 4.0:
                color = '#FFA500' # Orange
                quality = 'Moderate'
            else:
                color = '#FF0000' # Red
                quality = 'Poor'
            
            st.markdown(
                f"<h1 style='text-align: center; color: {color};'>{prediction:.2f} mg/m³</h1>", 
                unsafe_allow_html=True
            )
            st.markdown(
                f"<h3 style='text-align: center;'>Predicted Quality: {quality}</h3>", 
                unsafe_allow_html=True
            )
            
            # Simple bar chart
            st.markdown("---")
            st.subheader("Prediction Context")
            chart_data = pd.DataFrame({
                'Category': [f'Predicted CO (mg/m³) for {predicted_date.strftime("%Y-%m-%d")}'],
                'CO Value': [prediction]
            })
            
            st.bar_chart(chart_data.set_index('Category'), color='#800080')
            
            st.info(f"The model predicts an average CO concentration of **{prediction:.2f} mg/m³** for {predicted_date.strftime('%B %d, %Y')}.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.exception(e)


if __name__ == '__main__':

    app()

