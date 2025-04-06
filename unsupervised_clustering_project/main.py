
from src import preprocessing, model, evaluate

def run_pipeline():
    # Load and select features
    df = preprocessing.load_data("data/mall_customers.csv")
    X = preprocessing.select_features(df)

    # Fit KMeans
    kmeans = model.fit_kmeans(X, n_clusters=5)

    # Save model
    model.save_model(kmeans)

    # Predict clusters
    labels = model.predict_clusters(kmeans, X)

    # Visualize
    evaluate.plot_clusters(X, labels)

if __name__ == "__main__":
    run_pipeline()
