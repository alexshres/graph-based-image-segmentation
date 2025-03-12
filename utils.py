# FILE: utils.py
# Contains helper functions for the program

import cv2
import numpy as np
from annoy import AnnoyIndex

def get_image(path):
    """
    Grabs image from path returns it
    in RGB format normalized between 0 and 1
    """

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)
    img = img/255.0
    sigma = 0.8 # based on the paper

    blurred_img = cv2.GaussianBlur(img, (0, 0), sigma)

    return np.clip(blurred_img, 0.0, 1.0)

def flattened_to_coordinates(width, idx):
    col = idx % width
    row = idx // width

    return [row, col]



def get_neighbors(image, n_trees):
    img_to_manip = np.copy(image)

    num_pixels = image.shape[0] * image.shape[1]

    # Each pixel is now of the format (r, g, b)
    # We want to get it to (row, col, r, g, b)
    rgb_features = img_to_manip.reshape(num_pixels, 3)


    coords_to_stack = [None for i in range(num_pixels)]

    for i in range(num_pixels):
        coords_to_stack[i] = flattened_to_coordinates(image.shape[1], i)

        # Making sure coordinates are also in range 0-1
        coords_to_stack[i][0] = coords_to_stack[i][0] / float(image.shape[0]) 
        coords_to_stack[i][1] = coords_to_stack[i][1] / float(image.shape[1]) 

    features = np.hstack((np.array(coords_to_stack), rgb_features))

    # Building ANNOY - just using euclidean distance
    ann_idx = AnnoyIndex(5, 'euclidean')

    for i in range(num_pixels):
        ann_idx.add_item(i, features[i])


    ann_idx.build(n_trees)

    # Return ann_idx so we can query it in the Segmentation class
    return ann_idx


def build_adj_list(ann_idx, num_pixels, k):
    """
    ann_idx is the AnnoyIndex after building,
    num_pixels is the total number of pixels (items) in it,
    k is how many neighbors to retrieve for each pixel.
    
    Returns a NumPy array of shape (num_pixels * k, 3)
    with rows = [pixel, neighbor, distance] 
    sorted by distance
    """

    adjacency = []

    for i in range(num_pixels):
        neighbors, distances = ann_idx.get_nns_by_item(i, k+1, include_distances=True)

        # annoy can return pixel itself as nearest neighbor (hence k+1)
        if neighbors[0] == i:
            neighbors = neighbors[1:]
            distances = distances[1:]

        # Grabbing [pixel, neighbor, distance]
        for nbr, dist in zip(neighbors, distances):
            adjacency.append([i, nbr, dist])
    
    # Sorting by distance (or weight) as specified by paper
    np_adj = np.array(adjacency)
    adj_sorted = np_adj[np_adj[:, 2].argsort()]

    return adj_sorted



    







