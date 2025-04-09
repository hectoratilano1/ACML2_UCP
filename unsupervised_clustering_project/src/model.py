import pandas as pd
from sklearn.cluster import KMeans
import joblib
import logging

# Safe fallback logging config (in case it's run standalone)
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

def fit_kmeans(data: pd.DataFrame, n_clusters: int = 5):
    logging.info(f"Fitting KMeans with {n_clusters} clusters...")
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(data)
    logging.info("KMeans model training complete.")
    return model

def predict_clusters(model, data: pd.DataFrame):
    logging.info("Predicting clusters...")
    return model.predict(data)

def save_model(model, path="model.joblib"):
    try:
        joblib.dump(model, path)
        logging.info(f"Model saved to '{path}'")
    except Exception as e:
        logging.error(f"Failed to save model: {e}", exc_info=True)

def load_model(path="model.joblib"):
    try:
        logging.info(f"Loading model from '{path}'...")
        return joblib.load(path)
    except FileNotFoundError:
        logging.error(f"Model file '{path}' not found.")
        raise
    except Exception as e:
        logging.error(f"Failed to load model: {e}", exc_info=True)
        raise
