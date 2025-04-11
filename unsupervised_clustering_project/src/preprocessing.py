import pandas as pd
import streamlit as st

def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        st.error(f"❌ File not found: {filepath}")
        st.stop()

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[["Annual_Income", "Spending_Score"]]
