import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class ClusteringResult:
    labels: np.ndarray
    centroids: np.ndarray
    cost: float

class DataPreprocessor:
    def __init__(self, file_path: str, separator: str = ';', n_features: int = 6):
        self.file_path = file_path
        self.separator = separator
        self.n_features = n_features
        
    def load_and_clean(self) -> pd.DataFrame:
        """Load and preprocess the data."""
        data = pd.read_csv(self.file_path, sep=self.separator)
        data = data.iloc[:, :self.n_features]
        
        # Remove outliers using 3-sigma rule
        mean = data.mean()
        std = data.std()
        mask = (data >= mean - 3 * std) & (data <= mean + 3 * std)
        return data[mask].dropna()

class PCATransformer:
    def __init__(self, n_components: int = 3):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply PCA transformation and scaling."""
        data_pca = self.pca.fit_transform(data)
        data_pca_scaled = self.scaler.fit_transform(data_pca)
        return pd.DataFrame(
            data_pca_scaled, 
            columns=[f'PC{i+1}' for i in range(self.n_components)]
        )

class KMeansClusterer:
    def __init__(self, k: int, max_iters: int = 1000, n_init: int = 100):
        self.k = k
        self.max_iters = max_iters
        self.n_init = n_init
        
    def _initialize_centroids(self, data: pd.DataFrame) -> np.ndarray:
        """Initialize cluster centroids randomly."""
        return data.sample(n=self.k).values
    
    def _assign_clusters(self, data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assign points to nearest centroid."""
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, data: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Update centroid positions."""
        return np.array([data[labels == i].mean(axis=0) for i in range(self.k)])
    
    def _calculate_cost(self, data: np.ndarray, labels: np.ndarray, 
                       centroids: np.ndarray) -> float:
        """Calculate clustering cost (sum of squared distances)."""
        cost = 0
        for i in range(len(centroids)):
            cluster_points = data[labels == i]
            cost += np.sum((cluster_points - centroids[i])**2)
        return cost
    
    def _single_kmeans_run(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, float]:
        """Perform a single run of k-means clustering."""
        centroids = self._initialize_centroids(data)
        data_values = data.values
        
        for _ in range(self.max_iters):
            labels = self._assign_clusters(data_values, centroids)
            new_centroids = self._update_centroids(data_values, labels)
            
            if np.all(centroids == new_centroids):
                break
            centroids = new_centroids
            
        cost = self._calculate_cost(data_values, labels, centroids)
        return labels, centroids, cost
    
    def fit(self, data: pd.DataFrame) -> ClusteringResult:
        """Perform multiple runs of k-means and select best result."""
        best_result = None
        best_cost = float('inf')
        
        for _ in range(self.n_init):
            labels, centroids, cost = self._single_kmeans_run(data)
            if cost < best_cost:
                best_cost = cost
                best_result = ClusteringResult(labels, centroids, cost)
                
        return best_result

class ClusterVisualizer:
    @staticmethod
    def plot_3d_clusters(data: pd.DataFrame, result: ClusteringResult, 
                        figsize: Tuple[int, int] = (10, 8)):
        """Create 3D visualization of clusters."""
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot data points
        scatter = ax.scatter(
            data['PC1'], data['PC2'], data['PC3'],
            c=result.labels, cmap='viridis', marker='o'
        )
        
        # Plot centroids
        ax.scatter(
            result.centroids[:, 0], result.centroids[:, 1], result.centroids[:, 2],
            color='red', marker='x', s=100, label='Centroids'
        )
        
        ax.set_title("K-means Clustering with PCA")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.legend()
        
        return fig

def main():
    # Initialize components
    preprocessor = DataPreprocessor('/content/kmeans_dataset.csv')
    pca_transformer = PCATransformer()
    clusterer = KMeansClusterer(k=4)
    visualizer = ClusterVisualizer()
    
    # Process data
    clean_data = preprocessor.load_and_clean()
    pca_data = pca_transformer.transform(clean_data)
    
    # Perform clustering
    clustering_result = clusterer.fit(pca_data)
    print(f"Final clustering cost: {clustering_result.cost}")
    
    # Visualize results
    fig = visualizer.plot_3d_clusters(pca_data, clustering_result)
    plt.show()

if __name__ == "__main__":
    main()