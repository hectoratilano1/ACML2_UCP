import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from src.model import save_model
from src import preprocessing

st.title("Customer Segmentation (KMeans Clustering)")

# ✅ Load and preprocess data (using a robust path)
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "data", "mall_customers.csv")
df = preprocessing.load_data(data_path)

X = preprocessing.select_features(df)

# Sidebar - number of clusters
st.sidebar.header("🛠️ Clustering Settings")
n_clusters = st.sidebar.slider("Select number of clusters:", 2, 10, 5)

# Fit KMeans
model = KMeans(n_clusters=n_clusters, random_state=42)
model.fit(X)
labels = model.predict(X)

# Save the trained model (optional)
save_model(model)

# Scatter plot
fig, ax = plt.subplots()
scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='viridis')
ax.set_xlabel(X.columns[0])
ax.set_ylabel(X.columns[1])
ax.set_title(f"KMeans Clustering (k={n_clusters})")
st.pyplot(fig)

# Show cluster counts
st.subheader("🔍 Cluster Distribution")
cluster_counts = pd.Series(labels).value_counts().sort_index()
st.bar_chart(cluster_counts)
