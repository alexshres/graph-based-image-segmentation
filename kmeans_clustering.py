import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

def kmeans_clustering(image_path, n_clusters=4, spatial_weight=0.5):
    """
    K-means image clustering with spatial and color features

    n_clusters: number of clusters
    spatial_weight: weight for spatial coordinates (0 = color only, 1 = spatial+color equally)

    Returns original_image, clustered_image 
    """

    img = cv2.imread(image_path)
    
    if img is None:  
      print(f"Error: Could not load image from {image_path}")
      return None, None, None 
      
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = img.astype(np.float32) / 255.0

    height, width = image.shape[:2]

    # features: [row, col, r, g, b]
    features = []
    for i in range(height):
        for j in range(width):
            # normalized spatial coordinates
            row_norm = i / height
            col_norm = j / width

            r, g, b = image[i, j]

            # Combine features with weighting
            feature_vector = [
                row_norm * spatial_weight,     # weighted spatial
                col_norm * spatial_weight,     # weighted spatial
                r, g, b                        # color features
            ]
            features.append(feature_vector)

    features = np.array(features)

    # applying k-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)

    clustered_image = labels.reshape(height, width)

    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    segmentation = np.zeros((height, width, 3))

    for i in range(n_clusters):
        mask = clustered_image == i
        segmentation[mask] = colors[i][:3]

    return image, segmentation 

