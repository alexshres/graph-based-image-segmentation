import numpy as np
import cv2
from sklearn.cluster import SpectralClustering
from sklearn.feature_extraction.image import img_to_graph
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class SklearnNormalizedCuts:
    def __init__(self, n_clusters=2, n_neighbors=10, gamma=1.0, random_state=42):
        """
        Using sklearn's normalized cuts for image segmentation since custom normalized
        cut algorithm takes ~18 mins to run on a 480x320 image
        
        Parameters:
        n_clusters: number of segments
        n_neighbors: number of neighbors for knn graph (much smaller than our custom implementation)
        gamma: scaling factor for rbf kernel (higher = more local connections)
        random_state: for reproducibility
        """
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.gamma = gamma
        self.random_state = random_state
    
    def segment_image_spatial(self, image):
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
    
    def segment_image_graph(self, image):
        """
        Using sklearn's img_to_graph for spatial connectivity
        FAST and often produces good results
        """
        height, width = image.shape[:2]
        
        # Convert to grayscale for graph construction
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Build graph based on spatial connectivity and intensity similarity
        graph = img_to_graph(gray) # , n_neighbors=self.n_neighbors)
        
        # Use spectral clustering with precomputed affinity
        clustering = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity='precomputed',
            random_state=self.random_state
        )
        
        labels = clustering.fit_predict(graph)
        return labels.reshape(height, width)
    
    def segment_image_custom_affinity(self, image, ann_idx=None):
        """
        Build custom affinity matrix similar to our original implementation
        but let sklearn handle the eigenvector computation
        """
        height, width = image.shape[:2] 
        num_pixels = height * width
        
        if ann_idx is None:
            # Fallback to simple approach if no neighbor index provided
            return self.segment_image_spatial(image)
        
        # Get RGB features
        rgb_features = image.reshape(num_pixels, 3)
        
        # Get coordinate features  
        coords = []
        for i in range(num_pixels):
            row, col = divmod(i, width)
            coords.append([row / height, col / width])
        coords = np.array(coords)
        
        # Build affinity matrix using ANNOY neighbors
        from scipy.sparse import csr_matrix
        rows, cols, data = [], [], []
        
        sigma_I, sigma_X = 0.1, 0.1  # Fixed parameters
        
        for i in range(num_pixels):
            neighbors = ann_idx.get_nns_by_item(i, self.n_neighbors + 1)[1:]  # Skip self
            
            for neighbor in neighbors:
                if neighbor < num_pixels:
                    # Feature similarity
                    feature_dist_sq = np.sum((rgb_features[i] - rgb_features[neighbor])**2)
                    feature_sim = np.exp(-feature_dist_sq / (2 * sigma_I**2))
                    
                    # Spatial similarity
                    spatial_dist_sq = np.sum((coords[i] - coords[neighbor])**2)
                    spatial_sim = np.exp(-spatial_dist_sq / (2 * sigma_X**2))
                    
                    weight = feature_sim * spatial_sim
                    
                    if weight > 1e-6:
                        rows.append(i)
                        cols.append(neighbor)
                        data.append(weight)
        
        # Create symmetric sparse matrix
        affinity = csr_matrix((data, (rows, cols)), shape=(num_pixels, num_pixels))
        affinity = (affinity + affinity.T) / 2
        
        # Use sklearn's spectral clustering with precomputed affinity
        clustering = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity='precomputed',
            random_state=self.random_state
        )
        
        labels = clustering.fit_predict(affinity)
        return labels.reshape(height, width)
    
    def segment_image_from_path(self, image_path, method='graph', downsample_factor=1):
        """
        Segment image from file path
        
        Parameters:
        image_path: path to image
        method: 'simple', 'graph', or 'custom'
        downsample_factor: factor to downsample image for speed
        
        Returns:
        image, segmentation
        """
        from utils import get_image, get_neighbors
        
        # Load image
        image = get_image(image_path)
        
        # Downsample if needed
        if downsample_factor > 1:
            h, w = image.shape[:2]
            new_h, new_w = h // downsample_factor, w // downsample_factor
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"Downsampled to {new_h}x{new_w}")
        
        print(f"Segmenting with method: {method}")
        
        if method == 'simple':
            segmentation = self.segment_image_spatial(image)
        elif method == 'graph':
            segmentation = self.segment_image_graph(image)
        elif method == 'custom':
            ann_idx = get_neighbors(image, 10)  # Fewer trees for speed
            segmentation = self.segment_image_custom_affinity(image, ann_idx)
        else:
            raise ValueError("Method must be 'simple', 'graph', or 'custom'")
        
        return image, segmentation
    
    def visualize_segmentation(self, image, segmentation):
        """Visualize segmentation results"""
        height, width = segmentation.shape
        n_segments = len(np.unique(segmentation))
        
        # Create colormap
        colors = plt.cm.tab10(np.linspace(0, 1, n_segments))
        
        # Create colored segmentation
        colored_seg = np.zeros((height, width, 3))
        for i, color in enumerate(colors):
            mask = segmentation == i
            colored_seg[mask] = color[:3]
        
        return colored_seg

# Speed comparison and usage examples
def compare_methods():
    """Compare different sklearn approaches"""
    
    # Initialize
    ncut = SklearnNormalizedCuts(n_clusters=4, n_neighbors=10)
    
    image_path = 'images/elephant.jpg'
    
    # Method 1: FASTEST (few seconds) - simple features + knn
    print("=== SIMPLE METHOD (FASTEST) ===")
    image1, seg1 = ncut.segment_image_from_path(image_path, method='simple', downsample_factor=2)
    
    # Method 2: FAST (30 seconds - 2 minutes) - spatial graph  
    print("=== GRAPH METHOD (FAST) ===")
    image2, seg2 = ncut.segment_image_from_path(image_path, method='graph', downsample_factor=2)
    
    # Method 3: MEDIUM (2-5 minutes) - custom affinity with ANNOY
    # print("=== CUSTOM METHOD (MEDIUM) ===") 
    # image3, seg3 = ncut.segment_image_from_path(image_path, method='custom', downsample_factor=2)
    
    # # Visualize all results
    # fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # # Original images
    # axes[0,0].imshow(image1)
    # axes[0,0].set_title('Original')
    # axes[0,1].imshow(image2) 
    # axes[0,1].set_title('Original')
    # axes[0,2].imshow(image3)
    # axes[0,2].set_title('Original')
    
    # # Segmentations
    # axes[1,0].imshow(seg1, cmap='tab10')
    # axes[1,0].set_title('Simple Method')
    # axes[1,1].imshow(seg2, cmap='tab10')
    # axes[1,1].set_title('Graph Method')
    # axes[1,2].imshow(seg3, cmap='tab10') 
    # axes[1,2].set_title('Custom Method')
    
    # # Colored visualizations
    # axes[0,3].imshow(ncut.visualize_segmentation(image1, seg1))
    # axes[0,3].set_title('Simple - Colored')
    # axes[1,3].imshow(ncut.visualize_segmentation(image2, seg2))
    # axes[1,3].set_title('Graph - Colored')

    colored_seg1 = ncut.visualize_segmentation(image1, seg1)
    colored_seg2 = ncut.visualize_segmentation(image2, seg2)

    fig, axes = plt.subplots(1, 5, figsize=(20, 10))

    axes[0].imshow(image1)
    axes[0].set_title("Original")

    axes[1].imshow(seg1, cmap='tab10')
    axes[1].set_title("Spatial Segmentation")

    axes[2].imshow(seg2, cmap='tab10')
    axes[2].set_title("Graph Method Segmentation")

    axes[3].imshow(colored_seg1)
    axes[3].set_title("Spatial - Colored")

    axes[4].imshow(colored_seg2)
    axes[4].set_title("Graph - Colored")
    
    plt.tight_layout()
    plt.show()


compare_methods()

"""
# FASTEST - for prototyping (5-10 seconds):
ncut_fast = SklearnNormalizedCuts(n_clusters=3, n_neighbors=5)
image, seg = ncut_fast.segment_image_from_path('image.jpg', method='simple', downsample_factor=4)

# FAST - good quality/speed tradeoff (30 seconds - 2 minutes):
ncut_balanced = SklearnNormalizedCuts(n_clusters=4, n_neighbors=10) 
image, seg = ncut_balanced.segment_image_from_path('image.jpg', method='graph', downsample_factor=2)

# MEDIUM - closer to paper implementation (2-5 minutes):
ncut_quality = SklearnNormalizedCuts(n_clusters=4, n_neighbors=20)
image, seg = ncut_quality.segment_image_from_path('image.jpg', method='custom', downsample_factor=1)
"""
