
import pandas as pd
from sklearn.cluster import KMeans
import joblib

def fit_kmeans(data: pd.DataFrame, n_clusters: int = 5):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(data)
    return model

def predict_clusters(model, data: pd.DataFrame):
    return model.predict(data)

def save_model(model, path="model.joblib"):
    joblib.dump(model, path)

def load_model(path="model.joblib"):
    return joblib.load(path)
