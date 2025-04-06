
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_clusters(df: pd.DataFrame, labels: list):
    df_copy = df.copy()
    df_copy["Cluster"] = labels
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_copy, x="Annual_Income", y="Spending_Score", hue="Cluster", palette="Set2", s=100)
    plt.title("Customer Segments via KMeans")
    plt.xlabel("Annual Income")
    plt.ylabel("Spending Score")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
