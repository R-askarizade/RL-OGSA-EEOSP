import numpy as np
from scipy.spatial.distance import cdist


from config import Config


class Network:
    def __init__(self):
        self.num_nodes = Config.NUM_NODES
        self.coords = np.random.rand(self.num_nodes, 2)
        self.coords[:, 0] *= Config.FIELD_LENGTH
        self.coords[:, 1] *= Config.FIELD_WIDTH

        self.energies = np.full(self.num_nodes, Config.E_0)
        self.dist_matrix = cdist(self.coords, self.coords, metric='euclidean')
        self.alive = np.ones(self.num_nodes, dtype=bool)

    def get_distance(self, i, j):
        return self.dist_matrix[i, j]

    def get_distance_to_point(self, node_idx, x, y):
        return np.sqrt((self.coords[node_idx, 0] - x)**2 + (self.coords[node_idx, 1] - y)**2)
