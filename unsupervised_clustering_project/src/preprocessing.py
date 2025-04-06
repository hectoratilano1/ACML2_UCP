
import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[["Annual_Income", "Spending_Score"]]
