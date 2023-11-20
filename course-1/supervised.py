import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class UnsupervisedWeatherClustering:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.features = self.data.drop(['Date', 'Events'], axis=1)  # Exclude non-numeric and target columns

    def preprocess_data(self):
        # Standardize the features
        scaler = StandardScaler()
        self.features_standardized = scaler.fit_transform(self.features)

    def perform_kmeans(self, n_clusters=3):
        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.data['Cluster'] = kmeans.fit_predict(self.features_standardized)

    def visualize_clusters(self):
        # Visualize the clusters
        plt.scatter(self.data['TempAvgF'], self.data['HumidityAvgPercent'], c=self.data['Cluster'], cmap='viridis')
        plt.title('K-Means Clustering of Weather Data')
        plt.xlabel('TempAvgF')
        plt.ylabel('HumidityAvgPercent')
        plt.show()

if __name__ == "__main__":
    data_path = "your_data.csv"  # Replace with the path to your CSV file
    weather_clustering = UnsupervisedWeatherClustering(data_path)
    weather_clustering.preprocess_data()
    weather_clustering.perform_kmeans(n_clusters=3)
    weather_clustering.visualize_clusters()
