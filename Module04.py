import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

file_path = 'Final_Report_of_the_Asian_American_Quality_of_Life__AAQoL_.csv'
data = pd.read_csv(file_path)

satisfaction_mapping = {
    "Strongly disagree": 1, "Disagree": 2, "Slightly disagree": 3,
    "Neither agree or disagree": 4, "Slightly agree": 5, "Agree": 6, "Strongly agree": 7
}
mental_health_mapping = {
    "Poor": 1, "Fair": 2, "Good": 3, "Very Good": 4, "Excellent": 5
}
ethnic_identity_mapping = {
    "Not at all": 1, "Not very close": 2, "Somewhat close": 3, "Very close": 4
}

data['Life_Satisfaction'] = data['Satisfied With Life 1'].map(satisfaction_mapping)
data['Mental_Health'] = data['Present Mental Health'].map(mental_health_mapping)
data['Ethnic_Identity'] = data['Identify Ethnically'].map(ethnic_identity_mapping)

clustering_data = data[['Life_Satisfaction', 'Mental_Health', 'Ethnic_Identity', 'Discrimination ']].dropna()
clustering_data.columns = ['Life_Satisfaction', 'Mental_Health', 'Ethnic_Identity', 'Discrimination']

scaler = StandardScaler()
clustering_data_scaled = scaler.fit_transform(clustering_data)

sse = []
range_k = range(1, 11)
for k in range_k:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(clustering_data_scaled)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range_k, sse, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.title('Elbow Method for Optimal k')
plt.show()

optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(clustering_data_scaled)

clustering_data['Cluster'] = clusters

for cluster_id in range(optimal_k):
    print(f"\nCluster {cluster_id}:")
    print(clustering_data[clustering_data['Cluster'] == cluster_id].mean())

plt.figure(figsize=(10, 8))
plt.scatter(clustering_data_scaled[:, 0], clustering_data_scaled[:, 1], c=clusters, cmap='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.xlabel('Feature 1 (Standardized)')
plt.ylabel('Feature 2 (Standardized)')
plt.title('K-Means Clustering')
plt.legend()
plt.show()

