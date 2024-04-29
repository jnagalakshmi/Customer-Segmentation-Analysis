# Import necessary libraries
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# Load customer data
customer_data = pd.read_csv('customer_data.csv')
# Preprocess data (e.g., handle missing values, encode categorical variables)
# Select relevant features for segmentation analysis
X = customer_data[['Age', 'Income', 'Spending_Score']]
# Perform K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(X)
# Visualize clusters
plt.scatter(customer_data['Income'],customer_data['Spending_Score'], c=customer_data['Cluster'], cmap='viridis')
plt.xlabel('Income')
plt.ylabel('Spending Score')
plt.title('Customer Segmentation Analysis')
plt.show()

# Explore cluster characteristics and interpret results
cluster_centers = kmeans.cluster_centers_
cluster_labels = kmeans.labels_
