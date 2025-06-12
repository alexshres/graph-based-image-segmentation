import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh

from utils import get_image, get_neighbors


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

    def _solver(self, W, k=2):
        """
        Solves the generalized eigenvalue problem: (D-W)y = aDy 

        W: sparse affinity matrix
        k: number of eigenvectors to compute (second one has the split information called Fiedler vector)

        Returns eigenvectors corresponding to the k smalles eigenvalues
        """

        # degree matrix
        degrees = np.array(W.sum(axis=1)).flatten()

        # regularization
        degrees = degrees + 1e-12

        D = diags(degrees, format='csr')


        # normalized Laplacian: L_norm = D^(-1/2)*(D-W)*D^(-1/2)
        sqrt_degrees = np.sqrt(degrees)
        D_neg_sqrt = diags(1.0 / sqrt_degrees, format='csr')

        L = D-W
        L_norm = D_neg_sqrt @ L @ D_neg_sqrt

        # Suggestion from AI
        try:
            eigenvals, eigenvecs = eigsh(L_norm, k=k, which="SM", sigma=0)
        except Exception:
            eigenvals, eigenvecs = eigsh(L_norm, k=k, which="SA")


        for i in range(k):
            eigenvecs[:, i] = eigenvecs[:, i] / sqrt_degrees

        return eigenvecs


    def _recursive_partition(self, W, pixel_indices, max_segments, current_segments):
        """
        Recursively partitioning untils desired number of segments 
        """

        if current_segments >= max_segments or len(pixel_indices) < 10:
            return [pixel_indices]

        sub_W = W[np.ix_(pixel_indices, pixel_indices)]

        try:
            fiedler_vector = self._solver(sub_W, k=2)
            fiedler_vector = fiedler_vector[:, 1]
        except Exception:
            return [pixel_indices]

        threshold = np.median(fiedler_vector)

        # splitting based on fiedler
        mask = fiedler_vector > threshold
        group1 = pixel_indices[mask]
        group2 = pixel_indices[~mask]

        if len(group1) < 5 or len(group2) < 5:
            return [pixel_indices]

        segments = []

        segments.extend(self._recursive_partition(W, group1, max_segments, current_segments+1))
        segments.extend(self._recursive_partition(W, group2, max_segments, current_segments+1))

        return segments


    def segment(self, image, ann_idx, n_segments=2):
        """
        Actual segmentation using normalized cut segmentation 

        image: image to segment
        ann_idx: ANNOY index 
        n_segments: number of segments to create

        Returns a segmentation (numpy array of shape [height, width]) with segment labels
        """

        height, width = image.shape[:2]
        num_pixels = height * width

        print(f"Building affinity matrix for {num_pixels} pixels")

        W = self._build_affinity_matrix(image, ann_idx)

        print(f"Affinity matrix built: {W.shape}, nnz: {W.nnz}")

        if n_segments == 2:
            eigenvecs = self._solver(W, k=2)
            fiedler_vector = eigenvecs[:, 1]

            threshold = np.median(fiedler_vector)
            labels = (fiedler_vector > threshold).astype(int)

        else:
            pixel_indices = np.arange(num_pixels)
            segments = self._recursive_partition(W, pixel_indices, n_segments, 1)

            labels = np.zeros(num_pixels, dtype=int)
            for i, segment in enumerate(segments):
                labels[segment] = i

            segmentation = labels.reshape(height, width)

            return segmentation

    def segment_image(self, image, n_segments=2):
        """
        Given an image segments image into n_segments 

        Returns the segmented result
        """

        ann_idx = get_neighbors(image, self.n_trees)

        segmentation = self.segment(image, ann_idx, n_segments)

        return segmentation

    
    def visualize_segmentation(self, segmentation):
        """
        Visualizing using matplotlib the segmented image and the original image 

        Returns image with color segmentation
        """

        height, width = segmentation.shape

        n_segments = len(np.unique(segmentation))

        colors = plt.cm.tab10(np.linspace(0, 1, n_segments))

        colored_seg = np.zeros((height, width, 3))
        for i, color in enumerate(colors):
            mask = segmentation == i
            colored_seg[mask] = color[:3]
        

        return colored_seg


ncut = NormalizedCut(sigma_I=0.1, sigma_X=0.1, k_neighbors=50)
image_path = "images/elephant.jpg"
image = get_image(image_path, blur=False)
segmentation = ncut.segment_image(image)
colored_seg = ncut.visualize_segmentation(segmentation)

fig, axes = plt.subplots(1, 3, figsize=(15, 12))

axes[0].imshow(image)
axes[0].set_title("Original Image")

axes[1].imshow(segmentation, cmap='tab10')
axes[1].set_title("Segmentation")

axes[2].imshow(colored_seg)
axes[2].set_title("Colored Segmentation")

for a in axes:
    a.axis('off')

plt.show()