import numpy as np
import unionfind as uf

from utils import flattened_to_coordinates, get_neighbors

class SegmentImage:
    def __init__(self, image):
        self.image = image
        self.size = image.shape[0] * image.shape[1]
        self.S = uf.UnionFind(self.size)
        self.adj_list =  None
        self.k = 300   # number of nearest neighbors

        ann_idx = get_neighbors(self.image, n_trees=10)

        for i in range(self.size):
            nbrs = ann_idx.get_nns_by_item(i, self.k, include_distances=True)
            print(f"{nbrs=}")





