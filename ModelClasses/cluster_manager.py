import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from collections import defaultdict

import ConfigClass.config
from ModelClasses.sensor_node import SensorNode


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
        # "optimizer", "adaptive" or "random"
        head_selection_strategy: str = "optimizer",
        optimizer_factory=lambda nodes, k, sink: GravitationalOptimizer(
            nodes=nodes,
            num_heads=k,
            sink_pos=sink,
            iterations=15,
            population_size=10,
            use_obl=True,
        ),
    ):

        self.nodes = [n for n in nodes if n.is_alive()
                      and n.has_known_position()]
        self.area_size = area_size
        self.comm_range = comm_range
        self.k_min = k_min
        self.k_max = k_max
        self.head_selection_strategy = head_selection_strategy
        self.optimizer_factory = optimizer_factory

        self.clusters: Dict[int, List['SensorNode']] = {}
        self.cluster_heads: List['SensorNode'] = []
        self.sink_pos: Tuple[float, float] = (0.0, 0.0)

    def _adaptive_cluster_count(self, sink_pos: Tuple[float, float] = (50, 50)) -> int:
        """Estimate optimal number of clusters based on energy and node density."""
        if not self.nodes:
            return 0

        e_avg = np.mean([n.energy for n in self.nodes])
        e_init = np.mean([n.init_energy for n in self.nodes])
        density = len(self.nodes) / (self.area_size[0] * self.area_size[1])
        f_density = min(1.5, max(0.5, density * 1e4))

        k_est = int(self.k_max * (e_avg / e_init) * f_density)
        return max(self.k_min, min(self.k_max, k_est))

    def _select_heads_by_strategy(self, k: int, sink_pos: Tuple[float, float]) -> List['SensorNode']:
        """Select cluster heads based on configured strategy."""
        if self.head_selection_strategy == "optimizer" and self.optimizer_factory is not None:
            optimizer = self.optimizer_factory(self.nodes, k, sink_pos)
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
                len(self.nodes), size=k, replace=False, p=probs)
            return [self.nodes[i] for i in indices]

    def form_clusters(self, sink_pos: Tuple[float, float] = (50, 50)):
        """
        Form clusters using the selected strategy.
        :param sink_pos: position of sink/base station (used by some optimizers)
        """
        self.sink_pos = sink_pos
        self.clusters.clear()
        self.cluster_heads.clear()

        k = self._adaptive_cluster_count(sink_pos)
        if k == 0 or not self.nodes:
            return

        heads = self._select_heads_by_strategy(k, sink_pos)
        self.cluster_heads = heads

        # Assign members to nearest valid head (within communication range)
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
                nearest = min(heads, key=lambda h: node.distance_to(h))
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
