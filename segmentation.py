import numpy as np
import unionfind as uf

from utils import flattened_to_coordinates, get_neighbors, build_adj_list

class SegmentImage:
    def __init__(self, image, k):
        self.image = image
        self.size = image.shape[0] * image.shape[1]
        self.S = uf.UnionFind(self.size)
        self.k = k  # threshold parameter
        self.num_neighbors = 10 # number of nearest neighbors to grab

        ann_idx = get_neighbors(self.image, n_trees=10)

        # This adjacency list is already sorted
        self.adjacency_list = build_adj_list(ann_idx, self.size, self.num_neighbors)

    def build_segments(self):
        """
        Builds the segmentation
        """

        for i in range(len(self.adjacency_list)):
            val1 = self.adjacency_list[i][0]
            val2 = self.adjacency_list[i][1]
            weight = self.adjacency_list[i][2]

            comp1 = self.S.find(val1)
            comp2 = self.S.find(val2)

            # If components are not the same
            if comp1 != comp2:
                # Grab thresholds for each component
                thresh_C1 = self.k / self.S.component_sizes[comp1]
                thresh_C2 = self.k / self.S.component_sizes[comp2]

                # Grab internal differences for each component
                int_C1 = self.S.internal_diff[comp1]
                int_C2 = self.S.internal_diff[comp2]

                # Merge if weight is <= MInt(C1, C2)
                if weight <= min(thresh_C1+int_C1, thresh_C2+int_C2):
                    # new internal difference is max of the three possible int_diffs
                    new_int_diff = max(weight, int_C1, int_C2)
                    self.S.union(comp1, comp2, new_int_diff)
                    
