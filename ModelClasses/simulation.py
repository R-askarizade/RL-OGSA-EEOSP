from routing import RoutingManager
from mobile_sink import MobileSink
from energy_model import EnergyModel
from reclustering_policy import ReclusteringPolicy
from cluster_manager import ClusterManager
from oppositional_gravitational_search import GravitationalOptimizer
from sensor_node import SensorNode


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
from collections import defaultdict
import random
from scipy.spatial import Voronoi
from scipy.spatial.distance import pdist, squareform


class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.9, seed=42):
        self.q_table = {}  # Map (state) -> {action: q_value}
        self.actions = actions
        self.alpha = alpha   # Learning rate
        self.gamma = gamma   # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.seed = seed

    def get_state(self, node, nodes, area_size, comm_range):
        """
        Refined State Definition for Scalability.
        Uses relative density to handle 100 vs 1000 node scenarios.
        """
        # 1. Neighbor Density (Normalized)
        dists = [np.hypot(node.x - n.x, node.y - n.y)
                 for n in nodes if n.id != node.id]
        neighbor_count = sum(d < comm_range for d in dists)

        # Calculate expected density based on total nodes & area (Global knowledge assumption or pre-config)
        # expected_density = (N * pi * R^2) / Area
        # For simplicity, we can just use broader buckets or relative logic:

        if neighbor_count == 0:
            density_state = 0  # Isolated (Bad)
        elif neighbor_count <= 2:
            density_state = 1  # Sparse
        elif neighbor_count <= 6:
            density_state = 2  # Good
        elif neighbor_count <= 12:
            density_state = 3  # Dense
        else:
            density_state = 4  # Very Dense (Overlapping)

        # 2. Boundary Proximity
        w, h = area_size
        dist_bound = min(node.x, w - node.x, node.y, h - node.y)

        if dist_bound < comm_range * 0.2:
            bound_state = 0   # Critical (Too close)
        elif dist_bound < comm_range:
            bound_state = 1       # Near
        else:
            bound_state = 2                               # Safe

        return (density_state, bound_state)

    def choose_action(self, state):
        random.seed(self.seed)
        self._ensure_state(state)
        # Epsilon-greedy
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            # Exploit: return action with max Q value
            return max(self.q_table[state], key=self.q_table[state].get)

    def learn(self, state, action, reward, next_state):
        self._ensure_state(state)
        self._ensure_state(next_state)

        old_q = self.q_table[state][action]
        max_future_q = max(self.q_table[next_state].values())

        # Bellman Equation
        new_q = old_q + self.alpha * \
            (reward + self.gamma * max_future_q - old_q)
        self.q_table[state][action] = new_q

    def _ensure_state(self, state):
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.actions}

    def decay_epsilon(self, decay_rate=0.99):
        self.epsilon = max(0.01, self.epsilon * decay_rate)


class RewardCache:
    def __init__(self):
        self._last_hash = None
        self._cached_reward = None

    def get_hash(self, nodes):
        # Create a stable hash of positions and alive status
        data = []
        for n in nodes:
            data.extend([n.x, n.y, int(n.is_alive())])
        return hash(tuple(data))

    def get(self, nodes):
        h = self.get_hash(nodes)
        if self._last_hash == h:
            return self._cached_reward, True
        return None, False

    def set(self, nodes, reward):
        self._last_hash = self.get_hash(nodes)
        self._cached_reward = reward


class Simulation:
    """
    WSN simulation with mobile sink, adaptive clustering, and multi-criteria reclustering.
    TODO -> FILL DOCUMENTATION
    """

    def __init__(
        self,
        area_size: Tuple[int, int] = (100, 100),
        n_nodes: int = 200,
        rounds: int = 4000,
        init_energy: float = 1.0,
        comm_range: float = 50.0,
        sink_mode: str = "adaptive",
        routing_mode: str = "multi-hop",
        include_ack_energy: bool = False,
        seed: Optional[int] = 42,
        localization_mode: str = "DRL",
        head_selection_strategy: str = "optimizer",
        round_duration_sec: float = 50.0,  # 1 round = 50 seconds

        # Heterogeneity controls
        enable_heterogeneity: bool = False,
        hetero_mode: str = 'weak',   # 'weak' or 'two_tier'
        weak_phi: float = 0.2,       # for weak heterogeneity: U(1-phi, 1+phi)
        # fraction of super nodes, extra energy factor (super nodes have 1+alpha)
        two_tier_m: float = 0.1,
        two_tier_alpha: float = 1.0,

        # Various data packet size
        variable_packet_size: bool = False,
        pkt_mean_bits: int = 4000,
        pkt_std_bits: int = 500,

        # Clistering
        k_min: int = 8,
        k_max: int = 20,

        # Routing
        weight_distance: float = 0.5,
        weight_energy: float = 0.3,
        weight_load: float = 0.1,
        weight_trust: float = 0.1,
        sink_policy: str = 'load_aware',
        num_sinks: int = 1,

        # re-clustering
        recluster_period: int = 75,
        energy_threshold: float = 0.1,
        load_threshold: int = 10,
        sink_move_threshold: float = 20.0,

        # OGSA
        go_iterations: int = 15,
        population_size: int = 10,
        alpha: float = 0.6,
        G0: float = 50.0,
        beta: float = 0.4,

        # Mobile sink
        energy_weight: float = 0.4,
        distance_weight: float = 0.6,
        visit_period: int = 5,

        # Node placement
        edge_threshold: float = 0.4,
        tune_edge_iterations: int = 20,
        reward_coverage_weight: float = 0.35,
        reward_edge_coverage_weight: float = 0.30,
        reward_connectivity_score_weight: float = 0.20,
        reward_uniformity_weight: float = 0.15,
        local_reward_coverage_score_weight: float = 0.30,
        local_reward_connectivity_score_weight: float = 0.25,
        local_reward_boundary_score_weight: float = 0.30,
        local_reward_overlap_penalty_weight: float = 0.15,
        q_alpha: float = 0.2,
        q_gamma: float = 0.8,
        q_epsilon: float = 0.9,
    ):

        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)

        self.area_size = area_size
        self.n_nodes = n_nodes
        self.rounds = rounds
        self.init_energy = init_energy
        self.comm_range = comm_range
        self.round_duration_sec = round_duration_sec  # seconds per round
        self.sink_mode = sink_mode
        self.routing_mode = routing_mode
        self.include_ack_energy = include_ack_energy
        self.localization_mode = localization_mode
        self.head_selection_strategy = head_selection_strategy

        # Hetero config
        self.enable_heterogeneity = bool(enable_heterogeneity)
        if hetero_mode not in {'weak', 'two_tier'}:
            raise ValueError("hetero_mode must be 'weak' or 'two_tier'")
        self.hetero_mode = hetero_mode
        self.weak_phi = weak_phi
        self.two_tier_m = two_tier_m
        self.two_tier_alpha = two_tier_alpha

        # Packet size variability
        self.variable_packet_size = variable_packet_size
        self.pkt_mean_bits = pkt_mean_bits
        self.pkt_std_bits = pkt_std_bits

        # Clustering
        self.k_min = k_min
        self.k_max = k_max
        # Routing parameters
        self.weight_distance = weight_distance
        self.weight_energy = weight_energy
        self.weight_load = weight_load
        self.weight_trust = weight_trust
        self.sink_policy = sink_policy
        self.num_sinks = num_sinks
        self.sink_status_table: Dict[MobileSink, dict] = {}
        # Reclustering parameters
        self.recluster_period = recluster_period
        self.energy_threshold = energy_threshold
        self.load_threshold = load_threshold
        self.sink_move_threshold = sink_move_threshold
        # Gravitational search parameters
        self.go_iterations = go_iterations
        self.population_size = population_size
        self.alpha = alpha
        self.beta = beta
        self.G0 = G0
        # Mobile Sink parameters
        self.visit_period = visit_period
        self.energy_weight = energy_weight
        self.distance_weight = distance_weight

        # Overhead and buffer overflow metrics
        self.buffer_overflow_count = 0
        self.routing_overhead_bytes = 0
        self.packets_dropped_link_failure = 0

        # list of (trigger_round, resolved_round)
        self.reclustering_events = []
        self.last_recluster_round = 0  # last round when reclustering occurred

        self.total_data_bytes = 0
        self.total_control_bytes = 0  # already partially tracked as routing_overhead_bytes

        # Movements plots
        self.sink_trajectory = [[]
                                # sink trajectory
                                for _ in range(self.num_sinks)]
        self.previous_edge_pos = []
        self.changed_edge_pos = []

        # Reinforcement Learning chase optimizarion (memory optimization)
        self._reward_cache = RewardCache()
        self.edge_threshold = edge_threshold
        self.tune_edge_iterations = tune_edge_iterations
        self.reward_coverage_weight = reward_coverage_weight
        self.reward_edge_coverage_weight = reward_edge_coverage_weight
        self.reward_connectivity_score_weight = reward_connectivity_score_weight
        self.reward_uniformity_weight = reward_uniformity_weight
        self.local_reward_coverage_score_weight = local_reward_coverage_score_weight
        self.local_reward_connectivity_score_weight = local_reward_connectivity_score_weight
        self.local_reward_boundary_score_weight = local_reward_boundary_score_weight
        self.local_reward_overlap_penalty_weight = local_reward_overlap_penalty_weight
        self.q_alpha = q_alpha
        self.q_gamma = q_gamma
        self.q_epsilon = q_epsilon

        # Decide initial energy according to hetero policy
        init_e = self.init_energy
        self.init_energy = []
        for i in range(self.n_nodes):
            self.np_rng = np.random.default_rng(seed=self.seed + i)
            if self.enable_heterogeneity and self.hetero_mode == 'weak':
                # multiplicative jitter U(1-phi, 1+phi)
                factor = self.np_rng.uniform(
                    1.0 - self.weak_phi, 1.0 + self.weak_phi)
                self.init_energy.append(init_e * factor)
            # tow_tier
            elif self.enable_heterogeneity and self.np_rng.random() < self.two_tier_m:
                # Decide advanced nodes deterministically via RNG
                self.init_energy.append(
                    init_e * (1.0 + self.two_tier_alpha))
            else:
                self.init_energy.append(init_e)

        # per-node default packet bits
        if self.variable_packet_size:
            data_packet_size = []
            for i in range(self.n_nodes):
                self.np_rng = np.random.default_rng(seed=self.seed + i)
                # normal around mean, clipped
                bits = int(self.np_rng.normal(
                    self.pkt_mean_bits, self.pkt_std_bits))
                bits = max(64, min(bits, 10_000))  # clip to practical range
                data_packet_size.append(bits)
        else:
            data_packet_size = [4000] * self.n_nodes

        # Create sensor nodes
        self.nodes: List[SensorNode] = [
            SensorNode(
                i,
                x=float(np.random.rand() * area_size[0]),
                y=float(np.random.rand() * area_size[1]),
                init_energy=self.init_energy[i],
                data_packet_size=data_packet_size[i],
                comm_range=self.comm_range,
                area_size=self.area_size,
                seed=self.seed
            )
            for i in range(n_nodes)
        ]

        # Initialize E2E delay trackers
        self.total_e2e_delay = 0.0
        self.delivered_packet_count = 0

        # Node placement
        if localization_mode == "DRL":
            self._voronoi_repulsion_initial_placement()
            edge_ids = self._identify_edge_nodes()
            if edge_ids:
                print(f"Edge nodes: {edge_ids}")
                self.edge_node_ids = edge_ids

                self.previous_edge_pos = [
                    (node.id, (node.x, node.y)) for node in self.nodes if node.id in edge_ids]
                final_edge_ids = self._fine_tune_edge_nodes_with_drl(edge_ids)
                self.changed_edge_pos = [(node.id, (node.x, node.y))
                                         for node in self.nodes if node.id in final_edge_ids]

        # Mobile sink (velocity = self.sink.speed m/round)
        if self.num_sinks == 1:
            self.sink = MobileSink(
                area_size=self.area_size,
                mode=self.sink_mode,
                speed=25.0,  # meters per round
                visit_period=self.visit_period,
                energy_weight=self.energy_weight,
                distance_weight=self.distance_weight,
                seed=self.seed
            )
        else:
            self.sink = []
            for i in range(self.num_sinks):
                ms = MobileSink(
                    area_size=self.area_size,
                    mode=self.sink_mode,
                    speed=25.0,  # meters per round
                    visit_period=self.visit_period,
                    energy_weight=self.energy_weight,
                    distance_weight=self.distance_weight,
                    seed=self.seed)
                ms.current_pos = np.array([(i+1) * self.area_size[0] / (self.num_sinks+1),
                                           (i+1) * self.area_size[1] / (self.num_sinks+1)])
                self.sink.append(ms)
                self.sink_status_table[ms] = {"pos": ms.get_position(), "current_load": float(
                    ms.current_load), "capacity": float(getattr(ms, 'capacity', 1.0))}

        # print(f"Sink speed: {self.sink.speed} m/round = "
        #       f"{self.sink.speed / self.round_duration_sec:.2f} m/s")

        # Energy, clustering, routing
        self.energy_model = EnergyModel(
            packet_size=4000, include_ack_energy=self.include_ack_energy)
        self.cluster_manager = ClusterManager(
            nodes=self.nodes,
            area_size=self.area_size,
            comm_range=self.comm_range,
            k_min=self.k_min,
            k_max=self.k_max,
            head_selection_strategy=self.head_selection_strategy,
            optimizer_factory=lambda nodes, k: GravitationalOptimizer(
                nodes=nodes, num_heads=k,
                iterations=self.go_iterations, population_size=self.population_size,
                alpha=self.alpha, beta=self.beta, G0=self.G0, seed=self.seed
            ),
            seed=self.seed
        )

        self.reclustering_policy = ReclusteringPolicy(
            cm=self.cluster_manager,
            recluster_period=self.recluster_period,
            energy_threshold=self.energy_threshold,
            load_threshold=self.load_threshold,
            sink_move_threshold=self.sink_move_threshold,
            enable_time=True,
            enable_energy=True,
            enable_load=True,
            enable_mobility=(sink_mode in {"adaptive", "eeosp"}),
            seed=self.seed
        )

        self.routing = RoutingManager(
            nodes=self.nodes,
            energy_model=self.energy_model,
            mode=self.routing_mode,
            area_size=self.area_size,
            comm_range=self.comm_range,
            policy=self.sink_policy,
            num_sinks=self.num_sinks,
            weight_distance=self.weight_distance,
            weight_energy=self.weight_energy,
            weight_load=self.weight_load,
            weight_trust=self.weight_trust,
            seed=self.seed
        )

        # Metrics
        self.total_generated = 0
        self.total_delivered = 0
        self.min_energy_threshold = 0.05

        self.results = {"round": [], "alive": [], "avg_energy": [
        ], "generated_cum": [], "delivered_cum": []}
        self.first_node_dead_round = None
        self.half_nodes_dead_round = None
        self.last_node_dead_round = None

        # Initial setup
        self.cluster_manager.form_clusters()
        self._apply_control_packet_cost()

        # Schedule initial data generation
        for node in self.nodes:
            node.schedule_next_data_gen(current_round=0, avg_interval=3)

    def _voronoi_repulsion_initial_placement(self, iterations: int = 10):
        """
        Uses Lloyd's Algorithm, based on Voronoi tessellation, to maximize coverage.
        Which is a more accurate implementation of "Voronoi node deployment".
        """

        n = len(self.nodes)
        w, h = self.area_size

        # Add dummy points far outside the area to handle edge cells correctly
        dummy_points = np.array([
            [-w, -h], [2*w, -h], [-w, 2*h], [2*w, 2*h],  # Corners
            [w/2, -h], [w/2, 2*h], [-w, h/2], [2*w, h/2]  # Midpoints
        ])

        # Initial positions
        positions = np.array([[node.x, node.y] for node in self.nodes])

        for _ in range(iterations):
            # 1. Compute the Voronoi diagram
            # Combine real points with dummy points
            all_points = np.vstack([positions, dummy_points])
            vor = Voronoi(all_points)

            new_positions = np.zeros_like(positions)

            # 2. For each real node, find the centroid of its Voronoi cell
            for i in range(n):
                region_index = vor.point_region[i]
                region = vor.regions[region_index]

                # A region can be unbounded or empty if the point is on the convex hull.
                if not region or -1 in region:
                    # If the cell is unbounded, just use the old position or a slight random jitter.
                    new_positions[i] = positions[i]
                    continue

                polygon = vor.vertices[region]

                # 3. Move the node to the centroid of its cell
                centroid = polygon.mean(axis=0)

                # 4. Clip to the boundaries
                new_positions[i] = np.clip(centroid, [0, 0], [w, h])

            positions = new_positions

        # Update the final positions of the node objects
        for i, node in enumerate(self.nodes):
            node.x, node.y = float(positions[i, 0]), float(positions[i, 1])

    def _identify_edge_nodes(self) -> List[int]:
        """Identify edge nodes based on neighbor count."""
        if not self.nodes:
            return []
        positions = np.array([[n.x, n.y] for n in self.nodes])
        dist_matrix = squareform(pdist(positions))
        neighbor_counts = np.sum(dist_matrix <= self.comm_range, axis=1) - 1
        max_neighbors = len(self.nodes) - 1
        threshold = max(1, self.edge_threshold * max_neighbors)
        edge_ids = [self.nodes[i].id for i in np.where(
            neighbor_counts < threshold)[0]]
        return edge_ids

    def _compute_reward(self, changed_node_id: Optional[int] = None) -> float:
        """
        Compute reward based on coverage, connectivity, and edge coverage.
        If changed_node_id is provided, focus on local improvement metrics.
        """
        if not self.nodes:
            return 0.0

        # -- LIGHTWEIGHT CACHE (no structural change) --
        if not hasattr(self, '_reward_cache'):
            self._reward_cache = RewardCache()

        cached_val, is_cached = self._reward_cache.get(self.nodes)
        if is_cached:
            return cached_val

        alive_nodes = [n for n in self.nodes if n.is_alive()]
        if not alive_nodes:
            self._reward_cache.set(self.nodes, 0.0)
            return 0.0

        alive_positions = np.array([[n.x, n.y] for n in alive_nodes])
        comm_range_sq = self.comm_range ** 2

        # 1. Coverage (grid-based)
        grid_density = 0.4
        grid_size = int(max(self.area_size) * grid_density)
        if grid_size < 1:
            grid_size = 1

        gx_vals = np.linspace(0, self.area_size[0], grid_size)
        gy_vals = np.linspace(0, self.area_size[1], grid_size)
        X, Y = np.meshgrid(gx_vals, gy_vals)
        grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)  # (G, 2)

        # Compute squared distances from each grid point to all alive nodes
        diffs = grid_points[:, None, :] - \
            alive_positions[None, :, :]  # (G, N, 2)
        dists_sq = np.sum(diffs ** 2, axis=2)  # (G, N)
        covered = np.any(dists_sq <= comm_range_sq, axis=1)  # (G,)
        covered_cells = np.sum(covered)
        coverage = covered_cells / (grid_size * grid_size)

        # 2. Edge coverage — VECTORIZED
        edge_margin = self.comm_range * 0.5
        is_edge = (
            (grid_points[:, 0] < edge_margin) |
            (grid_points[:, 0] > self.area_size[0] - edge_margin) |
            (grid_points[:, 1] < edge_margin) |
            (grid_points[:, 1] > self.area_size[1] - edge_margin)
        )
        edge_points = grid_points[is_edge]
        edge_total = edge_points.shape[0]

        if edge_total > 0:
            diffs_edge = edge_points[:, None, :] - \
                alive_positions[None, :, :]  # (E, N, 2)
            dists_sq_edge = np.sum(diffs_edge ** 2, axis=2)
            edge_covered = np.any(dists_sq_edge <= comm_range_sq, axis=1)
            edge_covered_count = np.sum(edge_covered)
            edge_coverage = edge_covered_count / edge_total
        else:
            # or 0.0? but original would give 0/0 → max(1,0)=1 → 0/1=0, but safe to use 1.0 if no edge
            edge_coverage = 1.0
            # However, original uses max(1, edge_total) → so if edge_total=0, edge_coverage = 0
            edge_coverage = 0.0

        # 3. Connectivity
        if len(alive_positions) > 1:
            dist_matrix_conn = squareform(pdist(alive_positions))
            neighbor_counts = np.sum(
                dist_matrix_conn <= self.comm_range, axis=1) - 1
            avg_neighbors = np.mean(neighbor_counts)
            min_neighbors = np.min(neighbor_counts)
            connectivity_score = min(
                1.0, (min_neighbors / 3.0) * (avg_neighbors / 5.0))
        else:
            connectivity_score = 0.0

        # 4. Uniformity
        if len(alive_positions) > 1:
            distances = pdist(alive_positions)
            distance_std = np.std(distances)
            distance_mean = np.mean(distances)
            uniformity = 1.0 / \
                (1.0 + (distance_std / max(distance_mean, 1e-6)))
        else:
            uniformity = 0.0

        reward = (
            self.reward_coverage_weight * coverage +
            self.reward_edge_coverage_weight * edge_coverage +
            self.reward_connectivity_score_weight * connectivity_score +
            self.reward_uniformity_weight * uniformity
        )

        self._reward_cache.set(self.nodes, reward)
        return reward

    def _compute_local_reward(self, node_id: int) -> float:
        """
        Compute localized reward for a specific node.
        Which is more sensitive to individual node movements.
        """
        node = self.nodes[node_id]
        if not node.is_alive():
            return 0.0

        comm_range_sq = self.comm_range ** 2

        # 1. Local coverage contribution
        grid_size = 20
        search_radius = self.comm_range * 1.5

        x_min = max(0, node.x - search_radius)
        x_max = min(self.area_size[0], node.x + search_radius)
        y_min = max(0, node.y - search_radius)
        y_max = min(self.area_size[1], node.y + search_radius)

        gx_vals = np.linspace(x_min, x_max, grid_size)
        gy_vals = np.linspace(y_min, y_max, grid_size)
        X, Y = np.meshgrid(gx_vals, gy_vals)
        local_points = np.stack([X.ravel(), Y.ravel()], axis=1)  # (G, 2)

        node_pos = np.array([node.x, node.y])
        diffs = local_points - node_pos  # (G, 2)
        dists_sq = np.sum(diffs ** 2, axis=1)
        covered_local = np.sum(dists_sq <= comm_range_sq)
        local_coverage_score = covered_local / (grid_size * grid_size)

        # 2. Connectivity
        alive_positions = np.array([[n.x, n.y]
                                    for n in self.nodes if n.is_alive()])
        if len(alive_positions) == 0:
            connectivity_score = 0.0
        else:
            diffs_conn = alive_positions - np.array([node.x, node.y])
            dists_sq_conn = np.sum(diffs_conn ** 2, axis=1)
            neighbor_count = np.sum(
                dists_sq_conn <= comm_range_sq) - 1  # exclude self
            connectivity_score = min(1.0, neighbor_count / 6.0)

        # 3. Distance to nearest boundary
        dist_to_boundary = min(
            node.x,
            self.area_size[0] - node.x,
            node.y,
            self.area_size[1] - node.y
        )
        boundary_score = 1.0 - \
            min(1.0, dist_to_boundary / (self.comm_range * 2))

        # 4. Overlap penalty
        if len(alive_positions) > 1:
            diffs_overlap = alive_positions - np.array([node.x, node.y])
            dists = np.sqrt(np.sum(diffs_overlap ** 2, axis=1))
            dists = dists[dists > 1e-8]  # exclude self (numerical safety)
            if dists.size > 0:
                min_distance = np.min(dists)
            else:
                min_distance = self.comm_range
        else:
            min_distance = self.comm_range

        overlap_penalty = max(
            0.0, 1.0 - (min_distance / (self.comm_range * 0.3)))

        reward = (
            self.local_reward_coverage_score_weight * local_coverage_score +
            self.local_reward_connectivity_score_weight * connectivity_score +
            self.local_reward_boundary_score_weight * boundary_score +
            self.local_reward_overlap_penalty_weight * (1.0 - overlap_penalty)
        )

        return reward

    def _fine_tune_edge_nodes_with_drl(self, edge_node_ids: List[int]):
        """
        Refined RL approach: Uses Independent Q-Learning (IQL).
        Each edge node acts as an agent maximizing local coverage and connectivity.
        """
        # Define Actions: (dx, dy)
        step = self.comm_range * 0.2
        actions = [
            (0, step), (0, -step), (step, 0), (-step, 0),  # N, S, E, W
            (0, 0)  # Stay
        ]

        # Initialize one shared brain (or individual brains if preferred)
        # Using a shared brain accelerates convergence for homogeneous nodes
        agent = QLearningAgent(actions=range(
            len(actions)), alpha=self.q_alpha, gamma=self.q_gamma, epsilon=self.q_epsilon, seed=self.seed)

        changes = []
        print(
            f"RL Optimizing {len(edge_node_ids)} edge nodes using Q-Learning...")

        for iteration in range(self.tune_edge_iterations):

            # Shuffle order to prevent sequential bias
            random.shuffle(edge_node_ids)

            for node_id in edge_node_ids:
                node = self.nodes[node_id]

                # 1. Observe State (S)
                state = agent.get_state(
                    node, self.nodes, self.area_size, self.comm_range)

                # 2. Choose Action (A)
                action_idx = agent.choose_action(state)
                dx, dy = actions[action_idx]

                # Store old pos/metrics for reward calc
                old_x, old_y = node.x, node.y
                # Note: We use local reward to allow decentralized learning
                old_reward = self._compute_local_reward(node_id)

                # 3. Take Action
                node.x = np.clip(node.x + dx, 0, self.area_size[0])
                node.y = np.clip(node.y + dy, 0, self.area_size[1])

                # 4. Observe New State (S') and Reward (R)
                # (S' depends on new neighbors/position)
                next_state = agent.get_state(
                    node, self.nodes, self.area_size, self.comm_range)
                new_reward = self._compute_local_reward(node_id)

                # Calculate Reward Delta (Immediate Reward for the transition)
                # You can use the absolute reward, but delta often stabilizes movement
                r_immediate = new_reward - old_reward

                # Penalty for moving out of bounds (soft constraint logic handled by clip, but add penalty)
                if node.x == 0 or node.x == self.area_size[0] or node.y == 0:
                    r_immediate -= 0.1

                # 5. Learn (Update Q-Table)
                agent.learn(state, action_idx, r_immediate, next_state)

                # 6. Deployment Logic (Not pure RL, but necessary for static deployment)
                # If the move was disastrously bad, revert it physically, but KEEP the learning
                # so the agent remembers that state-action pair was bad.
                if r_immediate < -0.05:  # Tolerance threshold
                    node.x, node.y = old_x, old_y
                else:
                    if node_id not in changes:
                        changes.append(node_id)

            # Decay exploration rate
            agent.decay_epsilon(0.95)

            if iteration % 20 == 0:
                final_reward = self._compute_reward()  # Global reward for logging
                print(
                    f"  Iter {iteration}: Epsilon={agent.epsilon:.2f}, Global Reward={final_reward:.4f}")

        return changes

    def _apply_control_packet_cost(self):
        """
        # TODO -> FILL DOCUMENTATIONS
        """
        control_packet_size = 64

        control_bytes = 64 // 8  # 64 bits = 8 bytes
        overhead_bytes = len(self.nodes) * control_bytes

        self.total_control_bytes += overhead_bytes
        self.routing_overhead_bytes += len(self.nodes) * control_bytes

        for node in self.nodes:
            if node.is_alive():
                avg_distance = self.comm_range * 0.5
                etx = self.energy_model.tx_energy(
                    avg_distance, control_packet_size)
                erx = self.energy_model.rx_energy(control_packet_size)
                node.energy = max(0.0, node.energy - (etx + erx))
                if node.energy <= self.min_energy_threshold:
                    node.alive = False
                    node.energy = 0.0

    def _build_node_to_ch_map(self):
        """Build a map from node ID to cluster head ID for fast lookup."""
        node_to_ch = {}
        clusters = self.cluster_manager.get_clusters()
        for ch_id, members in clusters.items():
            for member in members:
                node_to_ch[member.id] = ch_id
        return node_to_ch

    def _find_cluster_head(self, node: 'SensorNode', node_to_ch_map: Dict[int, int]) -> Optional['SensorNode']:
        ch_id = node_to_ch_map.get(node.id)
        if ch_id is None:
            return None
        for head in self.cluster_manager.cluster_heads:
            if head.id == ch_id and head.is_alive():
                return head
        return None

    # TODO
    def _send_to_base_station(self, data_size_bits: int, sink_pos: Tuple[float, float]):
        pass

    def run(self):
        """Run the simulation with accurate E2E delay and advanced metrics tracking."""
        self.detailed_metrics = {
            "round": [], "EC": [], "avg_RE": [], "TH": [], "PDR": [], "CA": [],
            "RL": [], "EE": [], "PLR": [], "LB": [], "FI": [], "CE": [],
            "E2E_Delay_Rounds": [], "E2E_Delay_Sec": [],
            "TH_pps": [], "EE_Js": [],
            "Buffer_Overflow_Rate": [],       # (%)
            "Routing_Overhead_Bytes": [],     # cumulative bytes
            "Traffic_Load_Pct": [],           # (%)
            "Overhead_Normalized": []
        }

        # Ensure all nodes have buffer_size attribute
        for node in self.nodes:
            if not hasattr(node, 'buffer_size'):
                node.buffer_size = 10  # default max buffered + pending packets

        for r in range(1, self.rounds + 1):
            alive_nodes = [n for n in self.nodes if n.is_alive()]
            if not alive_nodes:
                self.last_node_dead_round = r - 1
                break

            # Update mobile sink, sink_trajectory update for plot MS movements
            # Sink_positions
            if self.num_sinks > 1:
                sink_positions = []
                for i, s in enumerate(self.sink):
                    s.update_position(r, self.nodes)
                    self.sink_trajectory[i].append(s.get_position())
                    sink_positions.append(s.get_position())

                    # store latest advert into a list/dict
                    advert = s.advertise_status(current_round=r)
                    self.sink_status_table[s] = advert
            else:
                self.sink.update_position(r, self.nodes)
                self.sink_trajectory[0].append(self.sink.get_position())
                sink_positions = self.sink.get_position()

            # Check for reclustering
            should_recluster, _ = self.reclustering_policy.should_recluster(
                r, sink_positions)
            if should_recluster:
                self.cluster_manager.form_clusters()
                self._apply_control_packet_cost()
                self.reclustering_policy.update_after_recluster(
                    r, sink_positions)
                # Record reconfiguration trigger
                self.current_recluster_trigger = r
                self.routing.reset_loads()  # reset loads after reclustering

            ch_ids = {
                ch.id for ch in self.cluster_manager.cluster_heads if ch.is_alive()}
            nodes_to_send = [
                n for n in alive_nodes if n.next_data_gen_round <= r and n.id not in ch_ids]

            delivered_this_round = 0

            if nodes_to_send:
                self.total_generated += len(nodes_to_send)

                # Increment Data Bytes When Packets Are Generated
                for node in nodes_to_send:
                    data_bytes_per_packet = node.data_packet_size // 8
                    self.total_data_bytes += data_bytes_per_packet

                # Generate packets BEFORE routing
                for node in nodes_to_send:
                    node.generate_packet(r)
                    node.schedule_next_data_gen(r, avg_interval=3)

                node_to_ch_map = self._build_node_to_ch_map()

                # Route CM → CH or sink
                for node in nodes_to_send:
                    ch = self._find_cluster_head(node, node_to_ch_map)
                    if ch is None:
                        # Direct to sink
                        success = self.routing.route_to_sink(node, self.sink)
                        if success and node.is_alive():
                            for gen_round in node.pending_packets:
                                self.total_e2e_delay += (r - gen_round)
                                self.delivered_packet_count += 1
                                delivered_this_round += 1
                            node.pending_packets.clear()
                        else:
                            # Retransmission will be attempted next round
                            pass
                    else:
                        # To cluster head
                        members = self.cluster_manager.get_clusters().get(ch.id, [])
                        success = self.routing.route_to_ch(node, ch, members)
                        if success and ch.is_alive():
                            # Check buffer space BEFORE adding
                            available_space = ch.buffer_size - \
                                len(ch.buffered_packets)
                            if available_space > 0:
                                to_buffer = node.pending_packets[:available_space]
                                dropped = len(
                                    node.pending_packets) - len(to_buffer)
                                ch.buffered_packets.extend(to_buffer)
                                self.buffer_overflow_count += dropped
                            else:
                                self.buffer_overflow_count += len(
                                    node.pending_packets)
                            node.pending_packets.clear()
                        # else: keep packets in pending_packets for retransmission

                # Route CH → sink (send ALL buffered packets)
                for ch in self.cluster_manager.cluster_heads:
                    if not ch.is_alive() or not ch.buffered_packets:
                        continue
                    success = self.routing.route_to_sink(ch, self.sink)
                    if success and ch.is_alive():
                        for gen_round in ch.buffered_packets:
                            self.total_e2e_delay += (r - gen_round)
                            self.delivered_packet_count += 1
                            delivered_this_round += 1
                        # Record reconfiguration resolution if needed
                        if hasattr(self, 'current_recluster_trigger'):
                            resolve_round = r
                            reconfig_time_sec = (
                                resolve_round - self.current_recluster_trigger) * self.round_duration_sec
                            self.reclustering_events.append(reconfig_time_sec)
                            delattr(self, 'current_recluster_trigger')
                        num_packets = len(ch.buffered_packets)
                        ch.buffered_packets.clear()
                        for _ in range(num_packets):
                            if ch.is_alive():
                                self.energy_model.consume_da(
                                    ch, bits=ch.data_packet_size)

                self.total_delivered += delivered_this_round

            # Mark dead nodes
            for node in self.nodes:
                if node.energy <= self.min_energy_threshold:
                    node.alive = False
                    node.energy = 0.0

            # Log per-round results
            alive_count = len([n for n in self.nodes if n.is_alive()])
            avg_energy = np.mean(
                [n.energy for n in self.nodes if n.is_alive()]) if alive_count > 0 else 0.0
            self.results["round"].append(r)
            self.results["alive"].append(alive_count)
            self.results["avg_energy"].append(avg_energy)
            self.results["generated_cum"].append(self.total_generated)
            self.results["delivered_cum"].append(self.total_delivered)

            # Track FND/HND/LND
            dead_count = self.n_nodes - alive_count
            if self.first_node_dead_round is None and dead_count >= 1:
                self.first_node_dead_round = r
            if self.half_nodes_dead_round is None and dead_count >= self.n_nodes // 2:
                self.half_nodes_dead_round = r
            if dead_count == self.n_nodes:
                self.last_node_dead_round = r
                break

            # # DEBUGGING: Clustering logs summary
            # if r % 1000 == 0:
            #     self.cluster_manager.summary()

            # Log detailed metrics every 50 rounds
            if r % 50 == 0 or r == self.rounds or alive_count == 0:
                total_initial = sum(n.init_energy for n in self.nodes)
                total_remaining = sum(n.energy for n in self.nodes)
                EC = total_initial - total_remaining
                avg_RE = total_remaining / len(self.nodes) if self.nodes else 0
                TH = self.total_delivered / r if r > 0 else 0
                PDR = self.total_delivered / max(1, self.total_generated)
                PLR = 1 - PDR

                # Coverage (CA)
                grid_size = 20
                covered = sum(
                    any(np.hypot(n.x - gx, n.y - gy) <=
                        self.comm_range for n in self.nodes if n.is_alive())
                    for gx in np.linspace(0, self.area_size[0], grid_size)
                    for gy in np.linspace(0, self.area_size[1], grid_size)
                )
                CA = covered / (grid_size * grid_size)

                # Avg CH-to-sink distance (RL)
                RL = 0
                alive_chs = [
                    ch for ch in self.cluster_manager.cluster_heads if ch.is_alive()]
                if not alive_chs:
                    RL = 0.0
                    RL_nearest = 0
                    RL_per_sink = 0
                    RL_assigned = 0
                    per_sink_RL = {}
                else:
                    sinks = getattr(self, 'sink', None)
                    if self.num_sinks == 1:
                        # fallback to single sink object
                        primary_sink = getattr(self, 'sink', None)
                        if primary_sink is not None:
                            sink_pos = primary_sink.get_position()
                            dists = [
                                np.hypot(ch.x - sink_pos[0], ch.y - sink_pos[1]) for ch in alive_chs]
                            RL = float(np.mean(dists))
                            RL_nearest = RL_assigned = RL
                            per_sink_RL = {0: RL}
                        else:
                            RL = 0.0
                            per_sink_RL = {}
                    else:
                        # Multi-sink case:
                        # 1) RL_nearest: avg distance from CH to nearest sink
                        nearest_dists = []
                        # Also accumulate lists per sink index (by nearest)
                        per_sink_lists = {i: [] for i in range(len(sinks))}
                        for ch in alive_chs:
                            dists_to_sinks = [np.hypot(
                                ch.x - s.get_position()[0], ch.y - s.get_position()[1]) for s in sinks]
                            min_idx = int(np.argmin(dists_to_sinks))
                            min_dist = dists_to_sinks[min_idx]
                            nearest_dists.append(min_dist)
                            per_sink_lists[min_idx].append(min_dist)
                        RL_nearest = float(
                            np.mean(nearest_dists)) if nearest_dists else 0.0

                        # 2) RL_assigned: avg distance to sink chosen by routing policy (if chooser exists)
                        assigned_dists = []
                        assigned_per_sink = {i: [] for i in range(len(sinks))}
                        for ch in alive_chs:
                            try:
                                # attempt to use routing.choose_sink_for_node if available
                                sink_pos, sink_obj = self.routing.choose_sink_for_node(
                                    ch, sinks)
                                dist_assigned = np.hypot(
                                    ch.x - sink_pos[0], ch.y - sink_pos[1])
                                # find index of sink_obj in sinks for bookkeeping (fallback to nearest if not found)
                                try:
                                    sink_idx = sinks.index(sink_obj)
                                except Exception:
                                    sink_idx = int(np.argmin(
                                        [np.hypot(ch.x - s.get_position()[0], ch.y - s.get_position()[1]) for s in sinks]))
                                assigned_dists.append(dist_assigned)
                                assigned_per_sink[sink_idx].append(
                                    dist_assigned)
                            except Exception:
                                # if chooser missing/fails, fallback to nearest
                                dists_to_sinks = [np.hypot(
                                    ch.x - s.get_position()[0], ch.y - s.get_position()[1]) for s in sinks]
                                min_idx = int(np.argmin(dists_to_sinks))
                                min_dist = dists_to_sinks[min_idx]
                                assigned_dists.append(min_dist)
                                assigned_per_sink[min_idx].append(min_dist)

                        RL_assigned = float(
                            np.mean(assigned_dists)) if assigned_dists else 0.0

                        # compute per-sink means (None if no CHs assigned)
                        per_sink_RL = {i: (float(np.mean(lst)) if lst else None)
                                       for i, lst in assigned_per_sink.items()}

                        # legacy RL: primary sink mean distance (sink[0]) for compatibility/plots
                        primary_pos = sinks[0].get_position()
                        primary_dists = [
                            np.hypot(ch.x - primary_pos[0], ch.y - primary_pos[1]) for ch in alive_chs]
                        RL = float(np.mean(primary_dists)
                                   ) if primary_dists else 0.0

                EE = TH / max(1e-9, EC)

                # Load balancing & fairness
                energies = [n.energy for n in self.nodes if n.is_alive()]
                LB = 1.0
                FI = 0.0
                if len(energies) > 1:
                    mu_E, sigma_E = np.mean(energies), np.std(energies)
                    LB = 1 - (sigma_E / max(mu_E, 1e-9))
                if energies:
                    FI = (sum(energies) ** 2) / (len(energies)
                                                 * sum(e ** 2 for e in energies))

                CE = CA / max(1e-9, EC)

                # E2E Delay
                avg_delay_rounds = self.total_e2e_delay / \
                    self.delivered_packet_count if self.delivered_packet_count > 0 else 0.0
                avg_delay_sec = avg_delay_rounds * self.round_duration_sec

                # Advanced Metrics
                total_time_sec = r * self.round_duration_sec
                TH_pps = self.total_delivered / total_time_sec if total_time_sec > 0 else 0.0
                EE_joule_sec = self.total_delivered / \
                    (EC * total_time_sec) if (EC >
                                              0 and total_time_sec > 0) else 0.0

                # Buffer Overflow Rate (%)
                buffer_overflow_rate = (
                    self.buffer_overflow_count / self.total_generated) * 100 if self.total_generated > 0 else 0.0

                # Routing Overhead (bytes) — assumed from control packets
                routing_overhead = getattr(self, 'routing_overhead_bytes', 0)

                # Traffic Load (%)
                total_tx = sum(n.packets_sent for n in self.nodes)
                traffic_load_pct = (
                    total_tx / (self.n_nodes * r)) * 100 if r > 0 else 0.0

                # Normalized Overhead
                total_traffic_bytes = self.total_control_bytes + self.total_data_bytes
                overhead_normalized = (
                    self.total_control_bytes / total_traffic_bytes
                    if total_traffic_bytes > 0
                    else 0.0
                )

                # Record all
                self.detailed_metrics["round"].append(r)
                self.detailed_metrics["EC"].append(EC)
                self.detailed_metrics["avg_RE"].append(avg_RE)
                self.detailed_metrics["TH"].append(TH)
                self.detailed_metrics["TH_pps"].append(TH_pps)
                self.detailed_metrics["PDR"].append(PDR)
                self.detailed_metrics["CA"].append(CA)
                self.detailed_metrics["RL"].append(RL)
                self.detailed_metrics["EE"].append(EE)
                self.detailed_metrics["EE_Js"].append(EE_joule_sec)
                self.detailed_metrics["PLR"].append(PLR)
                self.detailed_metrics["LB"].append(LB)
                self.detailed_metrics["FI"].append(FI)
                self.detailed_metrics["CE"].append(CE)
                self.detailed_metrics["E2E_Delay_Rounds"].append(
                    avg_delay_rounds)
                self.detailed_metrics["E2E_Delay_Sec"].append(avg_delay_sec)
                self.detailed_metrics["Buffer_Overflow_Rate"].append(
                    buffer_overflow_rate)
                self.detailed_metrics["Routing_Overhead_Bytes"].append(
                    routing_overhead)
                self.detailed_metrics["Traffic_Load_Pct"].append(
                    traffic_load_pct)
                self.detailed_metrics["Overhead_Normalized"].append(
                    overhead_normalized)
                self.detailed_metrics.setdefault("RL_primary", []).append(RL)
                self.detailed_metrics.setdefault(
                    "RL_nearest", []).append(RL_nearest)
                self.detailed_metrics.setdefault(
                    "RL_assigned", []).append(RL_assigned)
                self.detailed_metrics.setdefault(
                    "RL_per_sink", []).append(per_sink_RL)

        # Final metrics
        self.metrics = {
            "FND": int(self.first_node_dead_round) if self.first_node_dead_round else self.rounds,
            "HND": int(self.half_nodes_dead_round) if self.half_nodes_dead_round else self.rounds,
            "LND": int(self.last_node_dead_round) if self.last_node_dead_round else self.rounds,
            "TotalGenerated": self.total_generated,
            "TotalDelivered": self.total_delivered,
            "PDR": PDR,
            "Avg_E2E_Delay_Rounds": self.total_e2e_delay / self.delivered_packet_count if self.delivered_packet_count > 0 else 0,
            "Avg_E2E_Delay_Sec": (self.total_e2e_delay / self.delivered_packet_count) * self.round_duration_sec if self.delivered_packet_count > 0 else 0,
            "RoundsSimulated": self.results["round"][-1] if self.results["round"] else 0,
            "RoutingOverhead": overhead_normalized
        }
        return self.metrics, self.detailed_metrics

    def to_detailed_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.detailed_metrics)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)

    def plot(self, show: bool = True):
        # Add E2E delay to plot_comparison if desired
        df = self.to_dataframe()
        if df.empty:
            return
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        axes[0].plot(df["round"], df["alive"], 'b-')
        axes[0].set_title("Alive Nodes")
        axes[0].set_xlabel("Round")

        axes[1].plot(df["round"], df["avg_energy"], 'g-')
        axes[1].set_title("Average Energy")
        axes[1].set_xlabel("Round")

        pdr_series = np.array(df["delivered_cum"]) / \
            np.maximum(1, np.array(df["generated_cum"]))
        axes[2].plot(df["round"], pdr_series, 'r-')
        axes[2].set_title("Cumulative PDR")
        axes[2].set_xlabel("Round")

        # E2E Delay
        df_detail = self.to_detailed_dataframe()
        if not df_detail.empty:
            axes[3].plot(df_detail["round"], df_detail["E2E_Delay_Sec"], 'm-')
            axes[3].set_title("Avg End-to-End Delay (sec)")
            axes[3].set_xlabel("Round")

        plt.tight_layout()
        if show:
            plt.show()

    def save_results(self, filename: str = "wsn_simulation_results.csv"):
        df = self.to_dataframe()
        df.to_csv(filename, index=False)
        df2 = self.to_detailed_dataframe()
        df2.to_csv("detailed_" + filename, index=False)
        print(f"Results saved to {filename}")
