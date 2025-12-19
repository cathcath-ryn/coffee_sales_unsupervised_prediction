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
    layout="centered"
)

st.title("☕ Coffee Selling Recommendation App")

st.markdown(
    """
This app uses **unsupervised clustering** on historical coffee sales data  
to recommend **when** and **at what price** a specific coffee should be sold.
"""
)

@st.cache_data
def load_data():
    return pd.read_csv("Coffe_sales.csv")

@st.cache_resource
def load_pipeline():
    return joblib.load("coffee_clustering_pipeline.joblib")

df = load_data()
pipeline = load_pipeline()

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

coffee = st.selectbox(
    "Select the coffee you want to sell:",
    sorted(df['coffee_name'].unique())
)

# Input for clusters
# Neutral reference values
input_data = pd.DataFrame({
    'hour_of_day': [10],                 # neutral time
    'money': [df['money'].median()],     # typical price
    'Weekdaysort': [3],                  # mid-week
    'Monthsort': [6],                    # mid-year
    'coffee_name': [coffee],
    'Time_of_Day': ['Morning'],          # placeholder
    'cash_type': ['card']
})

# Predicting cluster
cluster = pipeline.predict(input_data)[0]

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
