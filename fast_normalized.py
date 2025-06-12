import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import SpectralClustering
from utils import get_image

class NormalizedCuts:
    def __init__(self, n_clusters=2, n_neighbors=10, gamma=1.0, random_state=42):
        """
        Using sklearn's normalized cuts for image segmentation since custom normalized
        cut algorithm takes ~18 mins to run on a 480x320 image
        
        n_clusters: number of segments
        n_neighbors: number of neighbors for knn graph (much smaller than our custom implementation)
        gamma: scaling factor for rbf kernel (higher = more local connections)
        random_state: for reproducibility
        """
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.gamma = gamma
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


    def segmented_image(self, image):
        """
        Segment image from file path
        
        Returns the segmented version of the image 
        """
        
        segmentation = self.segment(image)

        return segmentation
    
    def visualize_segmentation(self, segmentation):
        """Visualize segmentation results"""
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


image_path = "images/elephant.jpg"
image = get_image(image_path, blur=False)

ncut = NormalizedCuts(n_clusters=8)
segmentation = ncut.segmented_image(image)
colored_seg = ncut.visualize_segmentation(segmentation)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image)
axes[0].set_title("Original")

axes[2].imshow(colored_seg, cmap='tab10')
axes[2].set_title("Colored Segmentation")

for a in axes:
    a.axis('off')

plt.show()

