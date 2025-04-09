import logging
from src import preprocessing, model, evaluate

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="pipeline.log",  # Remove this line to log to console instead
    filemode="w"
)

def run_pipeline():
    try:
        logging.info("Starting customer segmentation pipeline...")

        # Load and preprocess data
        logging.info("Loading dataset: mall_customers.csv")
        df = preprocessing.load_data("data/mall_customers.csv")

        logging.info("Selecting features for clustering...")
        X = preprocessing.select_features(df)

        # Fit KMeans
        logging.info("Fitting KMeans model with 5 clusters...")
        kmeans = model.fit_kmeans(X, n_clusters=5)

        # Save model
        logging.info("Saving trained KMeans model...")
        model.save_model(kmeans)

        # Predict clusters
        logging.info("Predicting clusters for all data points...")
        labels = model.predict_clusters(kmeans, X)

        # Visualize
        logging.info("Plotting clustered data...")
        # evaluate.plot_clusters(X, labels)

        logging.info("Pipeline completed successfully.")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        print("❌ Something went wrong. Check pipeline.log for details.")

if __name__ == "__main__":
    run_pipeline()
