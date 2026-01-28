import numpy as np
import math
from typing import List, Dict, Tuple


class ClusterManager:
    """
    # TODO -> FILL DOCUMENTATION
    """

    def __init__(
        self,
        nodes: List['SensorNode'],
        area_size: Tuple[float, float],
        comm_range: float,
        # TODO: SAY EXPLANATION OF CHOOSING THESE NUMBERS
        k_min: int = 2,
        k_max: int = 10,
        # TODO: Add config for density scaling factor and optimizer params
        density_scale: float = 1e4,
        # "optimizer", "adaptive" or "random"
        head_selection_strategy: str = "optimizer",
        seed: int = 42,
        optimizer_factory=lambda nodes, k: GravitationalOptimizer(
            nodes=nodes,
            num_heads=k,
            iterations=15,
            population_size=10,
            use_obl=True
        ),
    ):

        self.nodes = [n for n in nodes if n.is_alive()
                      and n.has_known_position()]
        self.seed = seed
        self.area_size = area_size
        self.comm_range = comm_range
        self.k_min = k_min
        self.k_max = k_max
        self.head_selection_strategy = head_selection_strategy
        self.optimizer_factory = optimizer_factory
        self.optimizer_factory.seed = self.seed

        self.clusters: Dict[int, List['SensorNode']] = {}
        self.cluster_heads: List['SensorNode'] = []
        self.density_scale = density_scale

    def _adaptive_cluster_count(self) -> int:
        """Estimate optimal number of clusters based on energy and node density."""
        if not self.nodes:
            return 0

        e_avg = np.mean([n.energy for n in self.nodes])
        e_init = np.mean([n.init_energy for n in self.nodes])
        density = len(self.nodes) / (self.area_size[0] * self.area_size[1])
        f_density = min(1.5, max(0.5, density * 1e4))

        k_est = int(self.k_max * (e_avg / e_init) * f_density)
        return max(self.k_min, min(self.k_max, k_est))

    def _select_heads_by_strategy(self, k: int) -> List['SensorNode']:
        """Select cluster heads based on configured strategy."""
        np.random.seed(self.seed)  # Ensure reproducibility
        if k <= 0 or k >= len(self.nodes):
            return self.nodes  # Edge case fallback

        if self.head_selection_strategy == "optimizer" and self.optimizer_factory is not None:
            optimizer = self.optimizer_factory(self.nodes, k)
            head_ids = optimizer.optimize()
            return [n for n in self.nodes if n.id in head_ids]

        elif self.head_selection_strategy == "random":
            if len(self.nodes) <= k:
                return self.nodes
            return np.random.choice(self.nodes, size=k, replace=False).tolist()

        else:
            if len(self.nodes) <= k:
                return self.nodes
            energies = np.array([n.energy for n in self.nodes])
            probs = energies / np.sum(energies)
            indices = np.random.choice(
                len(self.nodes), size=min(k, len(list(filter(lambda x: x > 0, probs)))), replace=False, p=probs)
            return [self.nodes[i] for i in indices]

    def form_clusters(self):
        """
        Form clusters using the selected strategy.
        """
        self.clusters.clear()
        self.cluster_heads.clear()

        k = self._adaptive_cluster_count()
        if k == 0 or not self.nodes:
            return

        heads = self._select_heads_by_strategy(k)
        self.cluster_heads = [
            h for h in heads if h.is_alive() and h.has_known_position()]

        # Ensure proper flags & buffers
        for ch in heads:
            ch.become_cluster_head(cluster_id=ch.id)
            # initialize/clear buffers
            ch.buffered_packets = getattr(ch, 'buffered_packets', [])
            ch.buffered_packets.clear()
            ch.buffer_size = getattr(ch, 'buffer_size', 10)

        # Assign members to nearest valid head
        for node in self.nodes:
            if node in heads:
                self.clusters.setdefault(node.id, []).append(node)
                continue

            best_head = None
            min_dist = float("inf")
            for head in heads:
                try:
                    d = node.distance_to(head)
                    if d <= self.comm_range and d < min_dist:
                        min_dist = d
                        best_head = head
                except ValueError:
                    continue

            if best_head is not None:
                self.clusters.setdefault(best_head.id, []).append(node)
            else:
                nearest = min(heads,
                              key=lambda h: node.distance_to(h))
                self.clusters.setdefault(nearest.id, []).append(node)

    def get_clusters(self) -> Dict[int, List['SensorNode']]:
        """Return current cluster mapping: {head_id: [members]}."""
        return self.clusters.copy()

    def summary(self) -> str:
        s = f"Adaptive clusters formed: {len(self.cluster_heads)}\n"
        for head in self.cluster_heads:
            members = self.clusters.get(head.id, [])
            s += f"  Head {head.id} (E={head.energy:.4f}) -> {len(members)} members\n"
        return s
