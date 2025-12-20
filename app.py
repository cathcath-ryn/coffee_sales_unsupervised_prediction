import streamlit as st
import pandas as pd
import joblib
import os

logo_path = "parami.jpg"

st.sidebar.markdown("Student Info")

if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=150)

st.sidebar.markdown("Name: **Ei Phyu Sin Win**")
st.sidebar.markdown("**Student ID: PIUS20230033**")
st.sidebar.markdown("Class: 2027")
st.sidebar.markdown("Intro to Machine Learning")

st.set_page_config(
    page_title="Coffee Selling Recommendation",
    page_icon="☕",
    layout="centered"
)

st.markdown(
    """
Let me recommend you about what type of coffee should be sold at what time of the day!
"""
)

with open("coffee_clustering_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

features = [
    'hour_of_day',
    'money',
    'Weekdaysort',
    'Monthsort',
    'coffee_name',
    'Time_of_Day',
    'cash_type'
]

# assign clusters
df['cluster'] = pipeline.predict(df[features])

# input
st.subheader("🔮 Choose Coffee Type")

# User selects coffee from dropdown
coffee = st.selectbox(
    "Select the coffee you want to sell:",
    sorted(df['coffee_name'].unique())
)

# Filter historical data for selected coffee
coffee_df = df[df['coffee_name'] == coffee].copy()

# Assign clusters using the trained model
coffee_df['cluster'] = pipeline.predict(coffee_df[features])

# Find dominant cluster for this coffee
dominant_cluster = coffee_df['cluster'].value_counts().idxmax()

# Infer recommendations from dominant cluster
recommended_time = (
    coffee_df[coffee_df['cluster'] == dominant_cluster]['Time_of_Day']
    .value_counts()
    .idxmax()
)

recommended_price = (
    coffee_df[coffee_df['cluster'] == dominant_cluster]['money']
    .mean()
)

# Recommended time
recommended_time = (
    df[df['cluster'] == cluster]['Time_of_Day']
    .value_counts()
    .idxmax()
)

recommended_price = (
    df[df['cluster'] == cluster]['money']
    .mean()
)

st.markdown("---")
st.subheader("✅ Recommendation")

st.success(
    f"""
**Coffee:** {coffee}  
**Best Time to Sell:** {recommended_time}  
**Recommended Price:** {recommended_price:.2f}
"""
)
