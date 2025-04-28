import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('Mall_Customers.csv')

X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X_scaled)

df['Cluster'] = dbscan.labels_
 
print(df.head())

plt.figure(figsize=(10,7))
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=dbscan.labels_, cmap='plasma')
plt.title('DBSCAN Clustering on Mall Customers')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.colorbar(label='Cluster Label')
plt.show()
