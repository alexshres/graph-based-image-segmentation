import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import SpectralClustering
from utils import get_image

class SpectralSegmentation:
    def __init__(self, n_clusters=8, n_neighbors=10, random_state=42):
        """
        Using sklearn's spectral clustering for image segmentation
        
        n_clusters: number of segments
        n_neighbors: number of neighbors for knn graph
        gamma: scaling factor for rbf kernel
        random_state: for reproducibility
        """
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.random_state = random_state
    
    def segment(self, image):
        """
        Using spatial coordinates only 
        """
        height, width = image.shape[:2]
        
        pixels = image.reshape(-1, 3)
        
        coords = []
        for i in range(height):
            for j in range(width):
                coords.append([i, j])
        coords = np.array(coords)
        
        # combining RGB and normalized coordinates  
        coords_norm = coords / np.array([height, width])
        features = np.hstack([pixels, coords_norm])
        
        # spectral clustering
        clustering = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity='nearest_neighbors',
            n_neighbors=self.n_neighbors,
            random_state=self.random_state
        )
        
        labels = clustering.fit_predict(features)

        return labels.reshape(height, width)


    def visualize_segmentation(self, image):
        """Visualize segmentation results"""

        segmentation = self.segment(image)

        height, width = segmentation.shape
        n_segments = len(np.unique(segmentation))
        
        # Create colormap
        colors = plt.cm.tab10(np.linspace(0, 1, n_segments)) # type: ignore
        
        # Create colored segmentation
        colored_seg = np.zeros((height, width, 3))
        for i, color in enumerate(colors):
            mask = segmentation == i
            colored_seg[mask] = color[:3]
        
        return colored_seg
