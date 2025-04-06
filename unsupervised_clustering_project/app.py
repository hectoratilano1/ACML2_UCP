
import streamlit as st
import pandas as pd
from src.model import load_model, predict_clusters

st.title("🧠 Customer Segmentation (KMeans Clustering)")

st.write("Enter customer information to predict cluster segment:")

income = st.slider("Annual Income (k$)", 10, 150, 50)
score = st.slider("Spending Score (1-100)", 1, 100, 50)

input_df = pd.DataFrame([{
    "Annual_Income": income,
    "Spending_Score": score
}])

model = load_model()

if st.button("Predict Cluster"):
    cluster = predict_clusters(model, input_df)[0]
    st.success(f"🧾 This customer belongs to Cluster #{cluster}")
