import cv2
import numpy as np

from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh


class NormalizedCut:
    def __init__(self, sigma_I=0.1, sigma_X=0.1, k_neighbors=50, n_trees=10):
        """
        Normalized Cut Segmentation algorithm by Shi and Malik 

        sigma_I: variance for feature similarity
        sigma_X: variance for spatial similarity
        k_neighbors: number of nearest neighbors for each pixel
        n_trees: number of trees for ANNOY index
        """

        self.sigma_I = sigma_I
        self.sigma_X = sigma_X
        self.k_neighbors = k_neighbors
        self.n_trees = n_trees


    def _build_affinity_matrix(self, image, ann_idx):
        """
        Building sparse affinity matrix using k-neares neighbors
        """

        height, width = image.shape[:2]
        num_pixels = height * width

        rgb_features = image.reshape(num_pixels, 3)

        coords = []
        for i in range(num_pixels):
            row, col = divmod(i, width)
            coords.append([row/height, col/width])

        coords = np.array(coords)

        rows, cols, data = [], [], []

        for i in range(num_pixels):
            # k-nearest neighbnors
            neighbors, distances = ann_idx.get_nns_by_item(i, self.k_neighbors+1, include_distances=True)
            neighbors = neighbors[1:]       # removing self from neighbors list

            for neighbor in neighbors:
                # feature similarity
                feature_dist_sq = np.sum((rgb_features[i] - rgb_features[neighbor])**2)
                feature_sim = np.exp(-feature_dist_sq/(2*self.sigma_I**2))

                # spatial similarity
                spatial_dist_sq = np.sum((coords[i] - coords[neighbor])**2)
                spatial_sim = np.exp(-spatial_dist_sq/(2*self.sigma_X**2))

                # combined similarity
                weight = feature_sim * spatial_sim

                if weight > 1e-6:
                    rows.append(i)
                    cols.append(neighbor)
                    data.append(weight)

        # creating sparse symmetric matrix
        W = csr_matrix((data, (rows, cols)), shape=(num_pixels, num_pixels))
        W = (W + W.T)/2

        return W
