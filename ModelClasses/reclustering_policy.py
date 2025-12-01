import numpy as np
from typing import List, Tuple, Optional

import ConfigClass.config
from ModelClasses.cluster_manager import ClusterManager


class ReclusteringPolicy:
    """
    # TODO -> FILL DOCUMENTATION
    """

    def __init__(
        self,
        cm: 'ClusterManager',
        recluster_period: int = 10,
        energy_threshold: float = 0.3,
        load_threshold: int = 10,
        sink_move_threshold: float = 20.0,
        enable_time: bool = True,
        enable_energy: bool = True,
        enable_load: bool = True,
        enable_mobility: bool = True,
        enable_fitness: bool = True
    ):
        self.cm = cm
        self.recluster_period = recluster_period
        self.energy_threshold = energy_threshold
        self.load_threshold = load_threshold
        self.sink_move_threshold = sink_move_threshold

        self.enable_fitness = enable_fitness
        self._last_best_fitness = None
        self._last_current_fitness = None
        self.enable_time = enable_time
        self.enable_energy = enable_energy
        self.enable_load = enable_load
        self.enable_mobility = enable_mobility

        self.last_recluster_round = 0
        self.last_sink_pos: Optional[Tuple[float, float]] = None

    def _fitness_based(self, current_round: int, check_interval: int = 10, threshold: float = 0.15) -> bool:
        """
        # TODO -> FILL DOCUMENTATION
        """
        if not self.enable_fitness or current_round % check_interval != 0:
            return False

        # Get current cluster head IDs
        current_heads = [h.id for h in self.cm.cluster_heads if h.is_alive()]
        if not current_heads:
            return True

        # Compute current fitness
        current_fit = self._compute_fitness(current_heads)

        # Run lightweight optimization to find best possible fitness NOW
        try:
            optimizer = GravitationalOptimizer(
                nodes=self.cm.nodes,
                num_heads=len(current_heads),
                sink_pos=self.cm.sink_pos if hasattr(
                    self.cm, 'sink_pos') else (0, 0),
                iterations=6,
                population_size=6,
                alpha=0.6,
                beta=0.4,
            )
            best_heads = optimizer.optimize()
            best_fit = self._compute_fitness(best_heads)
        except Exception:
            return False

        # If current solution is much worse than best possible, trigger recluster
        if current_fit > best_fit * (1 - threshold):
            self._last_best_fitness = best_fit
            self._last_current_fitness = current_fit
            return True

        return False

    def _compute_fitness(self, head_ids: List[int]) -> float:
        """Reuse the same fitness logic as in GravitationalOptimizer."""
        if not head_ids:
            return float("inf")

        try:
            heads = [next(n for n in self.cm.nodes if n.id == hid)
                     for hid in head_ids]
        except StopIteration:
            return float("inf")

        total_dist = 0.0
        for n in self.cm.nodes:
            dmin = min(n.distance_to(h) for h in heads)
            total_dist += dmin
        avg_dist = total_dist / len(self.cm.nodes)

        e_avg = np.mean([h.energy / h.init_energy for h in heads])
        return 0.6 * avg_dist + 0.4 * (1.0 - e_avg)

    def _time_based(self, current_round: int) -> bool:
        """Check if enough rounds have passed since last re-clustering."""
        if not self.enable_time:
            return False
        return (current_round - self.last_recluster_round) >= self.recluster_period

    def _energy_based(self) -> bool:
        """Check if average residual energy of alive nodes is below threshold."""
        if not self.enable_energy:
            return False
        alive_nodes = [n for n in self.cm.nodes if n.is_alive()]
        if not alive_nodes:
            return False
        avg_energy = np.mean([n.energy for n in alive_nodes])
        avg_init_energy = np.mean([n.init_energy for n in alive_nodes])
        return avg_energy < self.energy_threshold * avg_init_energy

    def _load_based(self) -> bool:
        """Check if any cluster exceeds the maximum allowed member count."""
        if not self.enable_load:
            return False
        clusters = self.cm.get_clusters()
        for members in clusters.values():
            if len(members) > self.load_threshold:
                return True
        return False

    def _mobility_based(self, current_sink_pos: Tuple[float, float]) -> bool:
        """
        Check if sink has moved significantly since last recorded position.
        Note: First move is not considered a trigger.
        """
        if not self.enable_mobility:
            return False
        if self.last_sink_pos is None:
            self.last_sink_pos = current_sink_pos
            return False
        dx = current_sink_pos[0] - self.last_sink_pos[0]
        dy = current_sink_pos[1] - self.last_sink_pos[1]
        distance = np.hypot(dx, dy)
        return distance > self.sink_move_threshold

    def should_recluster(
        self, current_round: int, current_sink_pos: Tuple[float, float]
    ) -> Tuple[bool, Optional[str]]:
        triggers = []

        if self._time_based(current_round):
            triggers.append(("time", "Time-based trigger"))
        if self._energy_based():
            triggers.append(("energy", "Energy-based trigger"))
        if self._load_based():
            triggers.append(("load", "Load-based trigger"))
        if self._mobility_based(current_sink_pos):
            triggers.append(("mobility", "Mobility-based trigger"))
        if self._fitness_based(current_round):
            triggers.append(("fitness", "Fitness-degradation trigger"))

        if triggers:
            reason = triggers[0][1]
            return True, reason

        return False, None

    def update_after_recluster(self, current_round: int, current_sink_pos: Tuple[float, float]):
        """
        Call this AFTER reclustering is actually performed.
        Updates internal state (last round and sink position).
        """
        self.last_recluster_round = current_round
        self.last_sink_pos = current_sink_pos
