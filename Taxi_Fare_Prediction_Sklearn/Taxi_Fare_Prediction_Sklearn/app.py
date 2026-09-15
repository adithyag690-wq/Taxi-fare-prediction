import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

# Set Page Config
st.set_page_config(
    page_title="Taxi Fare Prediction",
    page_icon="🚕",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #D97706;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .fare-card {
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(245, 158, 11, 0.3);
    }
    .fare-val {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0.3rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🚕 AI Taxi Fare & Ride Estimator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Accurately calculate estimated taxi trip fares using <b>RandomForestRegressor</b> based on distance, pickup time, passengers, and weekend surcharges.</div>', unsafe_allow_html=True)

model_path = BASE_DIR / "models" / "taxi_fare_model.pkl"
csv_path = BASE_DIR / "data" / "taxi_fares.csv"
chart_path = BASE_DIR / "outputs" / "feature_importance.png"

if not model_path.exists():
    st.error(f"Model file not found at `{model_path}`! Please run 'python train_model.py' first.")
    st.stop()

@st.cache_resource
def get_model():
    return joblib.load(str(model_path))

model = get_model()

tab1, tab2, tab3 = st.tabs(["🔮 Live Fare Estimator", "📈 Fare Predictors & Importance", "📋 Historical Trip Records"])

with tab1:
    col_input, col_result = st.columns([1.1, 0.9])
    
    with col_input:
        st.subheader("Trip Parameters")
        
        scenario = st.selectbox(
            "⚡ Quick Trip Scenario Preset",
            ["Custom Ride", "✈️ Long-Distance Airport Trip", "🌆 Evening Peak Hour Commute", "🌙 Late Night Weekend Return", "⚡ Short City Hop"]
        )
        
        if scenario == "✈️ Long-Distance Airport Trip":
            def_dist, def_pass, def_hr, def_wk = 28.5, 3, 11, 0
        elif scenario == "🌆 Evening Peak Hour Commute":
            def_dist, def_pass, def_hr, def_wk = 9.2, 1, 18, 0
        elif scenario == "🌙 Late Night Weekend Return":
            def_dist, def_pass, def_hr, def_wk = 14.0, 2, 23, 1
        elif scenario == "⚡ Short City Hop":
            def_dist, def_pass, def_hr, def_wk = 3.5, 1, 14, 0
        else:
            def_dist, def_pass, def_hr, def_wk = 8.5, 2, 19, 0
            
        with st.form("fare_form"):
            col_a, col_b = st.columns(2)
            with col_a:
                trip_distance = st.slider("Trip Distance (km)", 0.5, 50.0, float(def_dist), step=0.5)
                passenger_count = st.slider("Passenger Count", 1, 6, int(def_pass))
            with col_b:
                pickup_hour = st.slider("Pickup Hour (24h)", 0, 23, int(def_hr), format="%d:00")
                is_weekend = st.radio("Weekend Trip?", [0, 1], index=def_wk, format_func=lambda x: "Weekday (Mon-Fri)" if x == 0 else "Weekend (Sat-Sun)")
                
            submit_btn = st.form_submit_button("🚀 Calculate Estimated Fare", use_container_width=True)
            
    with col_result:
        st.subheader("Fare Breakdown")
        if submit_btn:
            trip = pd.DataFrame([{
                "trip_distance_km": trip_distance,
                "passenger_count": passenger_count,
                "pickup_hour": pickup_hour,
                "is_weekend": is_weekend
            }])
            
            fare = model.predict(trip)[0]
            per_km = fare / max(trip_distance, 0.1)
            per_person = fare / max(passenger_count, 1)
            
            st.markdown(f"""
            <div class="fare-card">
                <div style="font-size: 0.95rem; opacity: 0.9;">Estimated Trip Total</div>
                <div class="fare-val">₹{fare:,.2f}</div>
                <div style="font-size: 1.05rem; opacity: 0.95;">Effective Rate: ₹{per_km:.2f} / km</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            col_f1, col_f2 = st.columns(2)
            col_f1.metric("Cost per Rider", f"₹{per_person:.2f}")
            col_f2.metric("Distance Covered", f"{trip_distance:.1f} km")
            
            st.markdown("### 🏷️ Surcharge & Timing Insights")
            is_peak = (8 <= pickup_hour <= 10) or (17 <= pickup_hour <= 20)
            is_night = (pickup_hour >= 22) or (pickup_hour <= 5)
            
            if is_peak:
                st.warning("⏰ **Peak Commute Window (Traffic Surcharge)**: Slow average speeds during rush hour slightly inflate metered fare.")
            elif is_night:
                st.info("🌙 **Night Operating Hours**: Safe night transit pricing applies.")
            else:
                st.success("🟢 **Standard Daytime Pricing**: Optimal traffic conditions and standard rates.")
                
            if is_weekend == 1:
                st.info("🎉 **Weekend Leisure Window**: Weekend ride volume pricing applied.")
                
            with st.expander("🔍 Model Input Payload"):
                st.json(trip.to_dict(orient="records")[0])
        else:
            st.info("👈 Set trip distance and timing, then click **'Calculate Estimated Fare'**.")
            
    st.write("---")
    st.subheader("📊 Distance vs. Fare Curve (1 km to 30 km)")
    if st.button("Generate Fare Curve"):
        distances = list(range(1, 31))
        sim_data = pd.DataFrame([{
            "trip_distance_km": float(d),
            "passenger_count": passenger_count if 'passenger_count' in locals() else 2,
            "pickup_hour": pickup_hour if 'pickup_hour' in locals() else 19,
            "is_weekend": is_weekend if 'is_weekend' in locals() else 0
        } for d in distances])
        sim_preds = model.predict(sim_data)
        chart_data = pd.DataFrame({
            "Distance (km)": distances,
            "Estimated Fare (₹)": sim_preds
        }).set_index("Distance (km)")
        st.line_chart(chart_data)

with tab2:
    st.subheader("Feature Importance Breakdown")
    if chart_path.exists():
        st.image(str(chart_path), caption="Top Features Driving Taxi Fares", use_container_width=True)
    else:
        st.info("Feature importance plot will appear after running train_model.py")

with tab3:
    st.subheader("Historical Taxi Trips Dataset (taxi_fares.csv)")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Trips Logged", f"{len(df):,}")
        col2.metric("Average Fare", f"₹{df['fare_amount'].mean():.2f}")
        col3.metric("Avg Trip Distance", f"{df['trip_distance_km'].mean():.1f} km")
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("Dataset not found.")
