# Segmentation using k-means clustering algorithm

import utils
import numpy as np


class KMeans:
    def __init__(self, num_clusters=3, image):
        self.num_clusters = num_clusters
        self.image = image

        h, w = self.image.shape[:2]

        # Initializing cluster centers to be randomly
        # assigned
        self.cluster_coords = [[np.random.randint(0, h), np.random.randint(0, w)] for i in range(self.num_clusters)]

        # Currently no pixels assigned to each cluster
        self.cluster_assignments = dict()
        for i in range(self.num_clusters):
            self.cluster_assignments[i] = []

    def _L1_distance(self, x, y):
        return np.sqrt(np.sum(np.abs(x-y)))

    def _L2_distance(self, x, y):
        return np.sqrt(np.sum((x-y)**2))

    def fit(self, X):
        pass
