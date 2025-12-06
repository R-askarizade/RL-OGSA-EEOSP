import numpy as np
from typing import List, Tuple, Optional

import ConfigClass.config
from ModelClasses.sensor_node import SensorNode


class MobileSink:
    """
    # TODO -> FILL DOCUMENTATION
    """

    def __init__(
        self,
        area_size: Tuple[float, float],
        mode: str = "eeosp",
        speed: float = 25.0,
        visit_period: int = 5,
        trajectory: Optional[List[Tuple[float, float]]] = None,
        energy_weight: float = 0.4,
        distance_weight: float = 0.6,
    ):
        if mode not in {"fixed", "random", "adaptive", "eeosp"}:
            raise ValueError(
                "Mode must be 'fixed', 'random', 'adaptive', or 'eeosp'")
        if speed < 0:
            raise ValueError("Speed must be non-negative")
        if visit_period <= 0:
            raise ValueError("Visit period must be positive")

        self.area_size = area_size
        self.mode = mode
        self.speed = float(speed)
        self.visit_period = int(visit_period)
        self.trajectory = trajectory or []
        self.current_pos = np.array([area_size[0] / 2.0, area_size[1] / 2.0])
        self.current_target: Optional[np.ndarray] = None
        self.current_index = 0
        self.energy_weight = energy_weight
        self.distance_weight = distance_weight

        # Validate trajectory for fixed mode
        if self.mode == "fixed" and self.trajectory:
            for i, (x, y) in enumerate(self.trajectory):
                if not (0 <= x <= area_size[0] and 0 <= y <= area_size[1]):
                    raise ValueError(
                        f"Trajectory point {i} ({x},{y}) is outside area {area_size}")

    def _clip_position(self, pos: np.ndarray) -> np.ndarray:
        """Ensure sink stays inside the area."""
        w, h = self.area_size
        return np.clip(pos, [0.0, 0.0], [w, h])

    def _choose_random_target(self) -> np.ndarray:
        w, h = self.area_size
        return np.array([np.random.uniform(0, w), np.random.uniform(0, h)])

    def _choose_adaptive_target(self, nodes: List['SensorNode']) -> np.ndarray:
        """Legacy adaptive: move toward low-energy regions."""
        valid_nodes = [n for n in nodes if n.is_alive()
                       and n.has_known_position()]
        if not valid_nodes:
            return self.current_pos.copy()
        positions = np.array([[n.x, n.y] for n in valid_nodes])
        energies = np.array([n.energy for n in valid_nodes])
        weights = 1.0 / (energies + 1e-6)
        weights /= np.sum(weights)
        target = np.average(positions, axis=0, weights=weights)
        return self._clip_position(target)

    def _choose_fixed_target(self) -> np.ndarray:
        if not self.trajectory:
            return self.current_pos.copy()
        target = np.array(self.trajectory[self.current_index])
        self.current_index = (self.current_index + 1) % len(self.trajectory)
        return target

    def _choose_eeosp_target(self, cluster_heads: List['SensorNode']) -> np.ndarray:
        """
        Selects sink position that minimizes cost = energy_variance + distance_cost
        """
        if not cluster_heads:
            return self.current_pos.copy()

        # Get current positions and energies of CHs
        ch_positions = np.array([[ch.x, ch.y] for ch in cluster_heads])
        ch_energies = np.array([ch.energy for ch in cluster_heads])

        # Candidate positions: sample points in the field
        n_candidates = 20
        candidates = np.column_stack([
            np.random.uniform(0, self.area_size[0], n_candidates),
            np.random.uniform(0, self.area_size[1], n_candidates)
        ])

        best_candidate = self.current_pos.copy()
        best_cost = float('inf')

        max_energy_var = np.var(
            [n.init_energy for n in cluster_heads]) if cluster_heads else 1.0
        max_distance = np.linalg.norm([self.area_size[0], self.area_size[1]])
        max_distance_cost = len(ch_positions) * max_distance

        for cand in candidates:
            # Compute distance to each CH
            distances = np.linalg.norm(ch_positions - cand, axis=1)

            # Energy variance term
            energy_variance = np.var(ch_energies)
            energy_variance_norm = energy_variance / (max_energy_var + 1e-9)

            # Distance cost: weighted sum of distances
            distance_cost = np.sum(distances)
            distance_cost_norm = distance_cost / (max_distance_cost + 1e-9)

            # Total cost: ε * Var + α * Σ(Weight * Distance)
            cost = self.energy_weight * energy_variance_norm + \
                self.distance_weight * distance_cost_norm

            if cost < best_cost:
                best_cost = cost
                best_candidate = cand

        return self._clip_position(best_candidate)

    def _move_towards_target(self, target: np.ndarray):
        direction = target - self.current_pos
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            self.current_target = None
            return

        if dist <= self.speed:
            self.current_pos = target.copy()
            self.current_target = None
        else:
            step_vector = direction * (self.speed / dist)
            self.current_pos += step_vector
            self.current_pos = self._clip_position(self.current_pos)
            self.current_target = target.copy()

    def update_position(self, round_num: int, nodes: Optional[List['SensorNode']] = None):
        if round_num % self.visit_period != 0:
            if self.current_target is not None:
                self._move_towards_target(self.current_target)
            return

        # Get cluster heads (alive and with known position)
        cluster_heads = []
        if nodes is not None:
            cluster_heads = [
                n for n in nodes
                if n.is_alive() and n.has_known_position() and n.is_cluster_head
            ]
            # If no CH flag, assume all nodes are candidates (fallback)
            if not cluster_heads:
                cluster_heads = [
                    n for n in nodes if n.is_alive() and n.has_known_position()]

        if self.mode == "random":
            self.current_target = self._choose_random_target()
        elif self.mode == "adaptive":
            if nodes is None:
                raise ValueError("Nodes must be provided in 'adaptive' mode")
            self.current_target = self._choose_adaptive_target(nodes)
        elif self.mode == "fixed":
            self.current_target = self._choose_fixed_target()
        elif self.mode == "eeosp":
            self.current_target = self._choose_eeosp_target(cluster_heads)
        else:
            return

        self._move_towards_target(self.current_target)

    def get_position(self) -> Tuple[float, float]:
        return (float(self.current_pos[0]), float(self.current_pos[1]))
