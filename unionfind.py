import numpy as np


class Node:
    def __init__(self, pixel_value):
        self.pixel_value = pixel_value

        # Initially set parent to self
        self.parent = pixel_value

    def __repr__(self):
        return f"Node(id={self.pixel_value}, parent={self.parent})"


class UnionFind:
    def __init__(self, num_pixels):
        self.elements = [Node(i) for i in range(num_pixels)]
        self.component_size = dict() 
        self.internal_diff = dict()

        for i in range(num_pixels):
            # Setting each pixel as a component with size 1
            self.component_size[i] = 1

            # Setting each component's internal difference as 0
            self.internal_diff[i] = 0


    def find(self, ele_node):
        pass

    
    def union(self, comp1, comp2):
        pass
