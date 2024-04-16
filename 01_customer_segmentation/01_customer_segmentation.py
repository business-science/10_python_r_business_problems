# BUSINESS PROBLEMS THAT CAN BE SOLVED WITH DATA SCIENCE
# PROJECT 1: CUSTOMER SEGMENTATION

# Libraries
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Sample data creation
data = {
    'Age': [25, 47, 35, 45, 22, 34, 52, 23, 40, 60],
    'Annual Income (k$)': [25, 60, 29, 55, 20, 40, 50, 15, 60, 30],
    'Spending Score (1-100)': [30, 55, 35, 50, 45, 50, 30, 25, 70, 40]
}

df = pd.DataFrame(data)

# Using KMeans for clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']])

# Plotting the clusters
plt.figure(figsize=(10, 6))
colors = ['red', 'green', 'blue']
for i in range(3):
    plt.scatter(df[df['Cluster'] == i]['Age'], 
                df[df['Cluster'] == i]['Annual Income (k$)'], 
                label=f'Cluster {i+1}',
                c=colors[i])

plt.title('Customer Segmentation')
plt.xlabel('Age')
plt.ylabel('Annual Income (k$)')
plt.legend()
plt.show()
