import numpy as np
from sklearn.neighbors import NearestNeighbors


class ANN:
    def __init__(self, k=5, gamma=1.0, algorithm='kd_tree'):
        self.k = k
        self.gamma = gamma
        self.algorithm = algorithm
        self.neighbors = None
        self.data = None


    def fit(self, data):
        self.data = data
        self.neighbors = NearestNeighbors(n_neighbors=self.k, 
                                          algorithm=self.algorithm,
                                          ).fit(self.data)

    def predict(self, query):
        pass

