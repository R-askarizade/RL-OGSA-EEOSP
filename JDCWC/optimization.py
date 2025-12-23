import numpy as np
from config import CONFIG


class PUBMO:
    def __init__(self, node_positions, energies, max_iter=50, pop_size=10, teams=5):
        self.node_positions = node_positions
        self.energies = energies
        self.n = len(node_positions)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.teams = teams
        self.speed = CONFIG["speed"]

    def objective(self, ch_indices):
        if not ch_indices:
            return float('inf')
        ch_positions = [self.node_positions[i] for i in ch_indices]
        node_positions = self.node_positions
        energies = [self.energies[i] for i in ch_indices]

        Dis = compute_Dis(node_positions, ch_positions)
        En = compute_En(energies)
        D = compute_D(Dis, self.speed)

        C1, C2, C3 = Dis, 1 - En, D
        total_C = C1 + C2 + C3
        if total_C == 0:
            w1 = w2 = w3 = 1.0 / 3.0
        else:
            w1, w2, w3 = C1 / total_C, C2 / total_C, C3 / \
                total_C  # as per Eq. (6) description
        return w1 * Dis + w2 * (1 - En) + w3 * D

    def optimize(self, k):
        # Initialize population: each is list of k unique node indices
        population = []
        for _ in range(self.pop_size):
            ind = np.random.choice(self.n, size=k, replace=False).tolist()
            population.append(ind)

        best_individual = min(population, key=self.objective)
        best_fitness = self.objective(best_individual)

        rand_val = CONFIG["gauss_map_init"]

        for itr in range(1, self.max_iter + 1):
            new_pop = []
            for ind in population:
                # BMO-inspired update
                offspring = ind.copy()
                for j in range(len(offspring)):
                    if np.random.rand() < 0.5:
                        leader = best_individual[j % len(best_individual)]
                        rand_val = gauss_map(rand_val)
                        offspring[j] = leader
                        if offspring[j] >= self.n:
                            offspring[j] = np.random.randint(0, self.n)
                offspring = list(set(offspring))
                while len(offspring) < k:
                    offspring.append(np.random.randint(0, self.n))
                    offspring = list(set(offspring))
                offspring = offspring[:k]
                new_pop.append(offspring)

            # POA enhancement (Eq. 11–12)
            R = 10.0
            for i in range(len(new_pop)):
                if np.random.rand() < 0.3:
                    for j in range(len(new_pop[i])):
                        rand_val = gauss_map(rand_val)
                        if np.random.rand() < 0.5:
                            term = R * (1 - itr / self.max_iter) * \
                                (2 * rand_val - 1)
                            new_pop[i][j] = int(
                                abs(2 * new_pop[i][j] + term)) % self.n

            population = new_pop
            current_best = min(population, key=self.objective)
            current_fit = self.objective(current_best)
            if current_fit < best_fitness:
                best_individual = current_best
                best_fitness = current_fit

        return best_individual
