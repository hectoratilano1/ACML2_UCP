# Unsupervised Customer Clustering

This project uses KMeans to cluster mall customers based on income and spending score, helping to identify distinct customer segments.

## Project Structure

```
unsupervised_clustering_project/
├── app.py                        # Streamlit app
├── main.py                       # Full pipeline runner
├── notebooks/
│   └── Unsupervised_Clustering_Solution.ipynb
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── evaluate.py
│   └── utils.py
├── data/
│   └── mall_customers.csv        # <- Add the dataset here
├── requirements.txt
└── README.md
```

## How to Run

1. Install packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Add `mall_customers.csv` to the `data/` folder.

3. Run pipeline to train and visualize clusters:
   ```bash
   python main.py
   ```

4. Launch Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Output

- KMeans clustering model
- Customer segment visualization
- Streamlit UI for interactive predictions