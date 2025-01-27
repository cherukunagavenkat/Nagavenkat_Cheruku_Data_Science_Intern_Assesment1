import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt


file_path = r'D:\Job\merged_data.csv'
data = pd.read_csv(file_path)


data['TransactionDate'] = pd.to_datetime(data['TransactionDate'], errors='coerce')
data['SignupDate'] = pd.to_datetime(data['SignupDate'], errors='coerce')

# Feature Engineering for Clustering
customer_features = data.groupby('CustomerID').agg({
    'TotalValue': 'sum',
    'Quantity': 'sum',
    'ProductID': lambda x: x.nunique()
}).reset_index()

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(customer_features.drop(columns=['CustomerID']))

# Apply K-Means Clustering
optimal_clusters = 4  
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
customer_features['Cluster'] = kmeans.fit_predict(scaled_features)

# Evaluate Clustering
db_index = davies_bouldin_score(scaled_features, customer_features['Cluster'])
print(f"Davies-Bouldin Index: {db_index:.4f}")

# Visualize Clusters 
plt.figure(figsize=(10, 6))
plt.scatter(scaled_features[:, 0], scaled_features[:, 1], c=customer_features['Cluster'], cmap='viridis', s=50)
plt.title('Customer Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster')
plt.show()

# Save clustering results 
clustering_output_path = r'D:\Job\ClusteringResults.csv'
customer_features.to_csv(clustering_output_path, index=False, float_format='%.2f')
print(f"Clustering results saved to {clustering_output_path}")
