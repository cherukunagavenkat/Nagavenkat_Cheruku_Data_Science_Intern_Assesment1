# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 22:12:38 2025

@author: cheru
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load data
file_path = r"D:\Job\merged_data.csv"
data = pd.read_csv(file_path)
data.info()

# Preprocessing
data['TransactionDate'] = pd.to_datetime(data['TransactionDate'], errors='coerce')
data['SignupDate'] = pd.to_datetime(data['SignupDate'], errors='coerce')

# Feature Engineering
customer_features = data.groupby('CustomerID').agg({
    'TotalValue': 'sum',
    'Quantity': 'sum',
    'ProductID': lambda x: x.nunique(),
    'Region': 'first'
}).reset_index()

# One-hot encode categorical features
customer_features = pd.get_dummies(customer_features, columns=['Region'], drop_first=True)

# Compute similarity
def calculate_similarity(customer_id, feature_df):
    customer_idx = feature_df[feature_df['CustomerID'] == customer_id].index[0]
    features = feature_df.drop(columns=['CustomerID']).values
    similarity_matrix = cosine_similarity(features)
    similar_customers = np.argsort(-similarity_matrix[customer_idx])[:4]  # Top 3 plus itself
    return [(feature_df.iloc[idx]['CustomerID'], similarity_matrix[customer_idx][idx]) for idx in similar_customers if idx != customer_idx]

# Get top 3 lookalikes for customers C0001 - C0020
lookalike_results = {}
for cust_id in [f'C{i:04}' for i in range(1, 21)]:
    lookalike_results[cust_id] = calculate_similarity(cust_id, customer_features)

# Save results to CSV
lookalike_output_path = r'D:\Job\Lookalike.csv'
with open(lookalike_output_path, 'w') as f:
    f.write('CustomerID,Lookalike1,Score1,Lookalike2,Score2,Lookalike3,Score3\n')
    for cust_id, lookalikes in lookalike_results.items():
        line = [cust_id]
        for lookalike, score in lookalikes[:3]:
            line.extend([lookalike, score])
        f.write(','.join(map(str, line)) + '\n')

print(f"Lookalike results saved to {lookalike_output_path}")
