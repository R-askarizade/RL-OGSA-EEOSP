import numpy as np
import random as pyrand
from typing import List, Tuple, Dict

import ConfigClass.config
from ModelClasses.sensor_node import SensorNode


class GravitationalOptimizer:
    """
    # TODO -> FILL DOCUMENTATION
    """

    def __init__(
        self,
        nodes: List['SensorNode'],
        num_heads: int,
        sink_pos: Tuple[float, float],
        iterations: int = 15,
        population_size: int = 10,
        alpha: float = 0.6,
        beta: float = 0.4,
        G0: float = 100.0,
        use_obl: bool = True,
        K0: int = None,
    ):
        self.nodes = [n for n in nodes if n.is_alive()]
        self.num_heads = min(num_heads, len(self.nodes))
        self.sink_pos = sink_pos
        self.iterations = iterations
        self.pop_size = population_size
        self.alpha = alpha
        self.beta = beta
        self.G0 = G0
        self.use_obl = use_obl
        self.K0 = K0 if K0 is not None else population_size

        if len(self.nodes) == 0:
            raise ValueError("No alive nodes provided for optimization.")

        self.id_to_node = {n.id: n for n in self.nodes}
        self.all_ids = [n.id for n in self.nodes]
        self.id_set = set(self.all_ids)

    def _fitness(self, head_ids: List[int]) -> float:
        """Fitness = α * avg_distance + β * (1 - avg_normalized_energy)"""
        if not head_ids or self.num_heads == 0:
            return float("inf")

        unique_head_ids = list(set(head_ids))
        if len(unique_head_ids) != len(head_ids):
            return float("inf")

        try:
            heads = [self.id_to_node[hid]
                     for hid in unique_head_ids if hid in self.id_to_node]
        except KeyError:
            return float("inf")

        if len(heads) == 0:
            return float("inf")

        total_dist = 0.0
        for n in self.nodes:
            dmin = min(n.distance_to(h) for h in heads)
            total_dist += dmin
        avg_dist = total_dist / len(self.nodes)

        e_avg = np.mean([h.energy / h.init_energy for h in heads])
        return self.alpha * avg_dist + self.beta * (1.0 - e_avg)

    def _random_solution(self) -> List[int]:
        return pyrand.sample(self.all_ids, self.num_heads)

    def _opposite_solution(self, sol: List[int]) -> List[int]:
        opposite = []
        excluded = set(sol)
        candidates = [id_ for id_ in self.all_ids if id_ not in excluded]
        for _ in sol:
            if candidates:
                choice = pyrand.choice(candidates)
                opposite.append(choice)
                # Avoid duplicates in opposite
                candidates.remove(choice)
            else:
                # Fallback: pick from all IDs
                opposite.append(pyrand.choice(self.all_ids))
        return opposite[:self.num_heads]

    def _apply_obl(self, pop: List[List[int]]) -> List[List[int]]:
        """Apply Opposition-Based Learning to enhance initial population."""
        enhanced_pop = []
        for sol in pop:
            opp = self._opposite_solution(sol)
            # Keep the better of the two
            if self._fitness(opp) < self._fitness(sol):
                enhanced_pop.append(opp)
            else:
                enhanced_pop.append(sol)
        return enhanced_pop

    def optimize(self) -> List[int]:
        """Run discrete OGSA and return best cluster head set."""
        if len(self.nodes) <= self.num_heads:
            return [n.id for n in self.nodes]

        # 1. Initialize population
        pop = [self._random_solution() for _ in range(self.pop_size)]
        if self.use_obl:
            pop = self._apply_obl(pop)

        fitness_vals = [self._fitness(sol) for sol in pop]
        best_solution = min(pop, key=self._fitness)
        best_fitness = self._fitness(best_solution)

        for t in range(self.iterations):
            # 2. Update gravitational constant
            G = self.G0 * np.exp(-20 * t / self.iterations)

            # 3. Compute masses
            worst_f = max(fitness_vals)
            best_f = min(fitness_vals)
            if worst_f == best_f:
                masses = [1.0 / self.pop_size] * self.pop_size
            else:
                masses = [(worst_f - f) / (worst_f - best_f)
                          for f in fitness_vals]
                total_mass = sum(masses)
                if total_mass > 0:
                    masses = [m / total_mass for m in masses]
                else:
                    masses = [1.0 / self.pop_size] * self.pop_size

            # 4. Update Kbest (decreases linearly)
            Kbest = max(1, int(self.K0 * (1 - t / self.iterations)))

            # Sort agents by fitness (best first)
            sorted_indices = np.argsort(fitness_vals)
            Kbest_indices = sorted_indices[:Kbest]

            # 5. Update each agent
            for i in range(self.pop_size):
                current = pop[i].copy()
                new_sol = current.copy()

                for j in range(self.num_heads):
                    candidates_with_force = []
                    for k in Kbest_indices:
                        if k == i:
                            continue
                        candidate_id = pop[k][j]
                        if candidate_id != new_sol[j]:
                            force = masses[k] * G
                            candidates_with_force.append((candidate_id, force))

                    if candidates_with_force:
                        total_force = sum(
                            force for _, force in candidates_with_force)
                        if total_force > 0:
                            r = pyrand.random() * total_force
                            cum_force = 0.0
                            for cid, force in candidates_with_force:
                                cum_force += force
                                if r <= cum_force:
                                    if pyrand.random() < 0.8:
                                        new_sol[j] = cid
                                    break

                # Remove duplicates while preserving order
                seen = set()
                unique_new_sol = []
                for item in new_sol:
                    if item not in seen:
                        seen.add(item)
                        unique_new_sol.append(item)

                if len(unique_new_sol) < self.num_heads:
                    missing_count = self.num_heads - len(unique_new_sol)
                    available_ids = [
                        id_ for id_ in self.all_ids if id_ not in seen]
                    selected_missing = pyrand.sample(
                        available_ids, min(missing_count, len(available_ids)))
                    unique_new_sol.extend(selected_missing)
                    while len(unique_new_sol) < self.num_heads:
                        remaining_ids = [
                            id_ for id_ in self.all_ids if id_ not in unique_new_sol]
                        if remaining_ids:
                            unique_new_sol.append(pyrand.choice(remaining_ids))
                        else:
                            break

                new_sol = unique_new_sol

                # Evaluate
                f_new = self._fitness(new_sol)
                if f_new < fitness_vals[i]:
                    pop[i] = new_sol
                    fitness_vals[i] = f_new

                if f_new < best_fitness:
                    best_fitness = f_new
                    best_solution = new_sol.copy()

        return best_solution
