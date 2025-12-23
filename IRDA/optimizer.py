import numpy as np
import math
from config import Config


class KNNTrafficPredictor:
    def __init__(self, k_neighbors=14, history_dim=18):
        self.K = k_neighbors
        self.R = history_dim
        self.historical_data = []
        self.historical_labels = []

    def train(self, data_stream):
        limit = 1000
        data = data_stream[-limit:] if len(
            data_stream) > limit else data_stream
        for i in range(len(data) - self.R - 1):
            vector = data[i: i + self.R]
            label = data[i + self.R]
            self.historical_data.append(np.array(vector))
            self.historical_labels.append(label)

    def predict(self, current_vector):
        if len(self.historical_data) == 0:
            return 0

        hist_matrix = np.array(self.historical_data)
        curr_vec = np.array(current_vector)
        dists = np.linalg.norm(hist_matrix - curr_vec, axis=1)
        idx = np.argpartition(dists, self.K)[:self.K]
        v_next = np.mean(np.array(self.historical_labels)[idx])
        return v_next


class IRDAOptimizer:
    def __init__(self, network, ms_x, ms_y):
        self.net = network
        self.ms_x = ms_x
        self.ms_y = ms_y
        self.dim = Config.NUM_CHS
        self.alive_indices = np.where(self.net.alive)[0]
        self.num_alive = len(self.alive_indices)
        self.lb = 0
        self.ub = self.num_alive - 1

    def map_ind_to_nodes(self, individual):
        indices = np.clip(np.array(individual).astype(int),
                          0, self.num_alive - 1)
        return self.alive_indices[indices]

    def calculate_fitness(self, population_matrix):
        fitness_values = []
        avg_e = np.mean(self.net.energies[self.net.alive])

        for ind in population_matrix:
            chs = np.unique(self.map_ind_to_nodes(ind))
            m = len(chs)
            if m == 0:
                fitness_values.append(1e5)
                continue

            e_vals = self.net.energies[chs]
            f1 = 1.0 - (np.sum(e_vals) / (m * Config.E_0))

            ch_dists = self.net.dist_matrix[chs][:, self.net.alive]
            degrees = np.sum(ch_dists <= Config.D_0, axis=1)
            avg_degree = np.mean(degrees)
            f2 = 1.0 / (avg_degree + 1e-5)

            dist_path = np.abs(self.net.coords[chs, 1] - Config.SINK_PATH_Y)
            f3 = np.sum(dist_path) / (m * Config.FIELD_WIDTH)

            fit = 0.6*f1 + 0.2*f2 + 0.2*f3

            if np.any(e_vals <= avg_e):
                fit += 1000

            fitness_values.append(fit)

        return np.array(fitness_values)

    def run(self):
        if self.num_alive < self.dim:
            return self.alive_indices.tolist()

        pop_size = Config.IRDA_POPULATION

        x_norm = np.random.rand(pop_size, self.dim)
        for i in range(pop_size):
            for j in range(self.dim):
                x_norm[i, j] = Config.DELTA * math.sin(math.pi * x_norm[i, j])

        population = x_norm * (self.ub - self.lb) + self.lb

        best_sol = None
        best_fit = float('inf')

        for t in range(Config.IRDA_MAX_ITER):
            fits = self.calculate_fitness(population)

            min_idx = np.argmin(fits)
            if fits[min_idx] < best_fit:
                best_fit = fits[min_idx]
                best_sol = population[min_idx].copy()

            sorted_idx = np.argsort(fits)
            num_com = int(pop_size * Config.GAMMA)

            commanders = population[sorted_idx[:num_com]]
            stags = population[sorted_idx[num_com:]]

            if len(stags) > 0:
                stag_indices = np.random.randint(
                    0, len(stags), size=len(commanders))
                chosen_stags = stags[stag_indices]
                mask = np.random.rand(len(commanders)) < 0.5
                commanders[mask] = (commanders[mask] + chosen_stags[mask]) / 2

            offspring = []
            if len(stags) > 0:
                for com in commanders:
                    partner = stags[np.random.randint(0, len(stags))]
                    cut = np.random.randint(0, self.dim)
                    child = np.concatenate((com[:cut], partner[cut:]))
                    offspring.append(child)

            if offspring:
                offspring = np.array(offspring)
                if len(offspring) <= len(stags):
                    stags[-len(offspring):] = offspring

            population[:len(commanders)] = commanders
            population[len(commanders):] = stags
            population = np.clip(population, self.lb, self.ub)

        return np.unique(self.map_ind_to_nodes(best_sol)).tolist()
