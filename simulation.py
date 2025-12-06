from ModelClasses.routing import RoutingManager
from ModelClasses.mobile_sink import MobileSink
from ModelClasses.energy_model import EnergyModel
from ModelClasses.reclustering_policy import ReclusteringPolicy
from ModelClasses.cluster_manager import ClusterManager
from ModelClasses.oppositional_gravitational_search import GravitationalOptimizer
from ModelClasses.sensor_node import SensorNode
import ConfigClass.config


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import random
from scipy.spatial import Voronoi


class Simulation:
    """
    End-to-end delay-aware WSN simulation with mobile sink, adaptive clustering,
    and multi-criteria reclustering.
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
        seed: Optional[int] = 42,
        localization_mode: str = "DRL",
        head_selection_strategy: str = "optimizer",
        round_duration_sec: float = 50.0,  # 1 round = 50 seconds

        weight_distance: float = 0.5,
        weight_energy: float = 0.3,
        weight_load: float = 0.1,
        weight_trust: float = 0.1,

        recluster_period: int = 75,
        energy_threshold: float = 0.1,
        load_threshold: int = 10,
        sink_move_threshold: float = 20.0,

        go_iterations: int = 15,
        population_size: int = 10,
        alpha: float = 0.6,
        G0: float = 50.0,
        beta: float = 0.4,

        energy_weight: float = 0.4,
        distance_weight: float = 0.6,
        visit_period: int = 5
    ):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.area_size = area_size
        self.n_nodes = n_nodes
        self.rounds = rounds
        self.init_energy = init_energy
        self.comm_range = comm_range
        self.round_duration_sec = round_duration_sec  # seconds per round
        self.sink_mode = sink_mode
        self.routing_mode = routing_mode
        self.localization_mode = localization_mode
        self.head_selection_strategy = head_selection_strategy

        # Routing parameters
        self.weight_distance = weight_distance
        self.weight_energy = weight_energy
        self.weight_load = weight_load
        self.weight_trust = weight_trust
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

        # list of (trigger_round, resolved_round)
        self.reclustering_events = []
        self.last_recluster_round = 0  # last round when reclustering occurred

        self.total_data_bytes = 0
        self.total_control_bytes = 0  # already partially tracked as routing_overhead_bytes

        # Movements plots
        self.sink_trajectory = []     # sink trajectory
        self.previous_edge_pos = []
        self.changed_edge_pos = []

        # Create sensor nodes
        self.nodes: List[SensorNode] = [
            SensorNode(
                i,
                x=float(np.random.rand() * area_size[0]),
                y=float(np.random.rand() * area_size[1]),
                init_energy=self.init_energy,
                comm_range=self.comm_range,
                area_size=self.area_size,
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
                print(edge_ids)
                self.edge_node_ids = edge_ids

                self.previous_edge_pos = [
                    (node.id, (node.x, node.y)) for node in self.nodes if node.id in edge_ids]
                final_edge_ids = self._fine_tune_edge_nodes_with_drl(edge_ids)
                self.changed_edge_pos = [(node.id, (node.x, node.y))
                                         for node in self.nodes if node.id in final_edge_ids]

        # Mobile sink (velocity = self.sink.speed m/round)
        self.sink = MobileSink(
            area_size=self.area_size,
            mode=self.sink_mode,
            speed=25.0,  # meters per round
            visit_period=self.visit_period,
            energy_weight=self.energy_weight,
            distance_weight=self.distance_weight
        )
        print(f"[Info] Sink speed: {self.sink.speed} m/round = "
              f"{self.sink.speed / self.round_duration_sec:.2f} m/s")

        # Energy, clustering, routing
        self.energy_model = EnergyModel(packet_size=4000)
        self.cluster_manager = ClusterManager(
            nodes=self.nodes,
            area_size=self.area_size,
            comm_range=self.comm_range,
            k_min=8,
            k_max=20,
            head_selection_strategy=self.head_selection_strategy,
            optimizer_factory=lambda nodes, k, sink: GravitationalOptimizer(
                nodes=nodes, num_heads=k, sink_pos=sink,
                iterations=self.go_iterations, population_size=self.population_size,
                alpha=self.alpha, beta=self.beta, G0=self.G0
            ),
        )
        self.cluster_manager.sink_pos = self.sink.get_position()

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
        )

        self.routing = RoutingManager(
            nodes=self.nodes,
            energy_model=self.energy_model,
            mode=self.routing_mode,
            comm_range=self.comm_range,
            weight_distance=self.weight_distance,
            weight_energy=self.weight_energy,
            weight_load=self.weight_load,
            weight_trust=self.weight_trust
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
        self.cluster_manager.form_clusters(sink_pos=self.sink.get_position())
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

    def _identify_edge_nodes(self, edge_threshold: float = 0.4) -> List[int]:
        """Identify edge nodes based on neighbor count."""
        edge_ids = []
        positions = np.array([[n.x, n.y] for n in self.nodes])
        for i, node in enumerate(self.nodes):
            distances = np.linalg.norm(positions - positions[i], axis=1)
            neighbor_count = np.sum(distances <= self.comm_range) - 1
            max_neighbors = len(self.nodes) - 1
            if neighbor_count < max(1, edge_threshold * max_neighbors):
                edge_ids.append(node.id)
        return edge_ids

    def _compute_reward(self, changed_node_id: Optional[int] = None) -> float:
        """
        Compute reward based on coverage, connectivity, and edge coverage.
        If changed_node_id is provided, focus on local improvement metrics.
        """
        if not self.nodes:
            return 0.0

        # 1. Coverage (grid-based)
        grid_size = 40  # Increased resolution for better sensitivity
        covered_cells = 0
        for gx in np.linspace(0, self.area_size[0], grid_size):
            for gy in np.linspace(0, self.area_size[1], grid_size):
                if any(np.hypot(n.x - gx, n.y - gy) <= self.comm_range
                       for n in self.nodes if n.is_alive()):
                    covered_cells += 1
        coverage = covered_cells / (grid_size * grid_size)

        # 2. Edge coverage (coverage near boundaries)
        edge_margin = self.comm_range * 0.5
        edge_covered = 0
        edge_total = 0
        for gx in np.linspace(0, self.area_size[0], grid_size):
            for gy in np.linspace(0, self.area_size[1], grid_size):
                # Check if point is near boundary
                if (gx < edge_margin or gx > self.area_size[0] - edge_margin or
                        gy < edge_margin or gy > self.area_size[1] - edge_margin):
                    edge_total += 1
                    if any(np.hypot(n.x - gx, n.y - gy) <= self.comm_range
                           for n in self.nodes if n.is_alive()):
                        edge_covered += 1

        edge_coverage = edge_covered / max(1, edge_total)

        # 3. Connectivity (average neighbors)
        positions = np.array([[n.x, n.y] for n in self.nodes if n.is_alive()])
        if len(positions) > 1:
            neighbor_counts = []
            for i, node in enumerate(self.nodes):
                if node.is_alive():
                    distances = np.linalg.norm(
                        positions - positions[i], axis=1)
                    neighbor_count = np.sum(
                        distances <= self.comm_range) - 1  # Exclude self
                    neighbor_counts.append(neighbor_count)

            avg_neighbors = np.mean(neighbor_counts)
            min_neighbors = np.min(neighbor_counts)
            # Penalize if any node is isolated or poorly connected
            connectivity_score = min(
                1.0, (min_neighbors / 3.0) * (avg_neighbors / 5.0))
        else:
            connectivity_score = 0.0

        # 4. Node distribution uniformity (inverse of clustering coefficient)
        if len(positions) > 1:
            # Compute pairwise distances
            from scipy.spatial.distance import pdist
            distances = pdist(positions)
            distance_std = np.std(distances)
            distance_mean = np.mean(distances)
            uniformity = 1.0 / \
                (1.0 + (distance_std / max(distance_mean, 1e-6)))
        else:
            uniformity = 0.0

        # Weighted combination
        reward = (
            0.35 * coverage +           # Overall coverage
            # Edge/boundary coverage (important for edge nodes)
            0.30 * edge_coverage +
            0.20 * connectivity_score +  # Network connectivity
            0.15 * uniformity           # Spatial uniformity
        )

        return reward

    def _compute_local_reward(self, node_id: int) -> float:
        """
        Compute localized reward for a specific node.
        Which is more sensitive to individual node movements.
        """
        node = self.nodes[node_id]
        if not node.is_alive():
            return 0.0

        # 1. Local coverage contribution
        grid_size = 20
        local_coverage = 0
        search_radius = self.comm_range * 1.5  # Look in expanded neighborhood

        for gx in np.linspace(max(0, node.x - search_radius),
                              min(self.area_size[0], node.x + search_radius), grid_size):
            for gy in np.linspace(max(0, node.y - search_radius),
                                  min(self.area_size[1], node.y + search_radius), grid_size):
                dist = np.hypot(node.x - gx, node.y - gy)
                if dist <= self.comm_range:
                    local_coverage += 1

        local_coverage_score = local_coverage / (grid_size * grid_size)

        # 2. Connectivity (number of neighbors)
        positions = np.array([[n.x, n.y] for n in self.nodes if n.is_alive()])
        distances = np.linalg.norm(
            positions - np.array([node.x, node.y]), axis=1)
        neighbor_count = np.sum(
            distances <= self.comm_range) - 1  # Exclude self

        # Normalize by expected neighbors (5-8 is good)
        connectivity_score = min(1.0, neighbor_count / 6.0)

        # 3. Distance to nearest boundary (edge nodes should be near boundaries)
        dist_to_boundary = min(
            node.x,
            self.area_size[0] - node.x,
            node.y,
            self.area_size[1] - node.y
        )
        # Reward being closer to boundary (for edge nodes)
        boundary_score = 1.0 - \
            min(1.0, dist_to_boundary / (self.comm_range * 2))

        # 4. Overlap penalty (avoid being too close to other nodes)
        min_distance = np.min(distances[distances > 0]) if len(
            distances) > 1 else self.comm_range
        overlap_penalty = max(
            0.0, 1.0 - (min_distance / (self.comm_range * 0.3)))

        reward = (
            0.30 * local_coverage_score +
            0.25 * connectivity_score +
            0.30 * boundary_score +
            0.15 * (1.0 - overlap_penalty)
        )

        return reward

    def _fine_tune_edge_nodes_with_drl(self, edge_node_ids: List[int], iterations: int = 20):
        """
        Use reinforcement learning to fine-tune edge node positions.
        This implementation uses a proper reward-based optimization.
        """
        learning_rate = 0.1
        epsilon = 0.3  # Exploration rate
        changes = []

        print(f"[DRL] Fine-tuning {len(edge_node_ids)} edge nodes...")

        for iteration in range(iterations):
            improved = 0

            for node_id in edge_node_ids:
                node_idx = node_id

                # Store old position
                old_x, old_y = self.nodes[node_idx].x, self.nodes[node_idx].y

                # Compute old reward (both global and local)
                old_global_reward = self._compute_reward()
                old_local_reward = self._compute_local_reward(node_id)
                old_combined_reward = 0.4 * old_global_reward + 0.6 * old_local_reward

                # Epsilon-greedy exploration
                if np.random.random() < epsilon:
                    # Exploration: larger random move
                    dx = np.random.uniform(-self.comm_range *
                                           0.5, self.comm_range * 0.5)
                    dy = np.random.uniform(-self.comm_range *
                                           0.5, self.comm_range * 0.5)
                else:
                    # Exploitation: smaller, directed move
                    # Move toward boundary if not close enough
                    move_direction = np.random.choice(
                        ['toward_boundary', 'random'])

                    if move_direction == 'toward_boundary':
                        # Determine which boundary is closest
                        distances_to_boundaries = [
                            old_x,                          # left
                            self.area_size[0] - old_x,     # right
                            old_y,                          # bottom
                            self.area_size[1] - old_y      # top
                        ]
                        closest_boundary = np.argmin(distances_to_boundaries)

                        # Move toward that boundary
                        step_size = np.random.uniform(5, 15)
                        if closest_boundary == 0:  # Move left
                            dx = -step_size
                            dy = np.random.uniform(-5, 5)
                        elif closest_boundary == 1:  # Move right
                            dx = step_size
                            dy = np.random.uniform(-5, 5)
                        elif closest_boundary == 2:  # Move down
                            dx = np.random.uniform(-5, 5)
                            dy = -step_size
                        else:  # Move up
                            dx = np.random.uniform(-5, 5)
                            dy = step_size
                    else:
                        dx = np.random.uniform(-10, 10)
                        dy = np.random.uniform(-10, 10)

                # Apply move with clipping
                new_x = np.clip(old_x + dx, 0, self.area_size[0])
                new_y = np.clip(old_y + dy, 0, self.area_size[1])

                # Temporarily apply new position
                self.nodes[node_idx].x, self.nodes[node_idx].y = new_x, new_y

                # Compute new reward
                new_global_reward = self._compute_reward()
                new_local_reward = self._compute_local_reward(node_id)
                new_combined_reward = 0.4 * new_global_reward + 0.6 * new_local_reward

                # Decide whether to keep the move
                reward_improvement = new_combined_reward - old_combined_reward

                if reward_improvement > 0 or (reward_improvement > -0.01 and np.random.random() < 0.1):
                    # Accept improvement or small degradation with small probability (simulated annealing)
                    if node_id not in changes:
                        changes.append(node_id)
                    improved += 1
                else:
                    # Reject move
                    self.nodes[node_idx].x, self.nodes[node_idx].y = old_x, old_y

            # Decay epsilon (less exploration over time)
            epsilon *= 0.95

            if iteration % 10 == 0:
                final_reward = self._compute_reward()
                print(f"  Iteration {iteration}: {improved}/{len(edge_node_ids)} nodes improved, "
                      f"Global Reward: {final_reward:.4f}")

        final_reward = self._compute_reward()
        print(f"[DRL] Optimization complete. Final reward: {final_reward:.4f}, "
              f"{len(changes)} nodes changed position.")

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

            # Update mobile sink
            self.sink.update_position(r, self.nodes)

            # Add this to plot MS movements
            self.sink_trajectory.append(self.sink.get_position())

            # Check for reclustering
            should_recluster, _ = self.reclustering_policy.should_recluster(
                r, self.sink.get_position())
            if should_recluster:
                self.cluster_manager.form_clusters(
                    sink_pos=self.sink.get_position())
                self._apply_control_packet_cost()
                self.reclustering_policy.update_after_recluster(
                    r, self.sink.get_position())
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
                data_bytes_per_packet = self.energy_model.packet_size // 8
                self.total_data_bytes += len(nodes_to_send) * \
                    data_bytes_per_packet

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
                                self.energy_model.consume_da(ch)

                self.total_delivered += delivered_this_round

            # Mark dead nodes
            for node in self.nodes:
                if node.energy <= self.min_energy_threshold:
                    node.alive = False

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
                if self.cluster_manager.cluster_heads:
                    sink_pos = self.sink.get_position()
                    RL = np.mean([
                        np.hypot(ch.x - sink_pos[0], ch.y - sink_pos[1])
                        for ch in self.cluster_manager.cluster_heads if ch.is_alive()
                    ])

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

        # Final metrics
        self.metrics = {
            "FND": int(self.first_node_dead_round) if self.first_node_dead_round else -1,
            "HND": int(self.half_nodes_dead_round) if self.half_nodes_dead_round else -1,
            "LND": int(self.last_node_dead_round) if self.last_node_dead_round else -1,
            "TotalGenerated": self.total_generated,
            "TotalDelivered": self.total_delivered,
            "PDR": PDR,
            "Avg_E2E_Delay_Rounds": self.total_e2e_delay / self.delivered_packet_count if self.delivered_packet_count > 0 else 0,
            "Avg_E2E_Delay_Sec": (self.total_e2e_delay / self.delivered_packet_count) * self.round_duration_sec if self.delivered_packet_count > 0 else 0,
            "RoundsSimulated": self.results["round"][-1] if self.results["round"] else 0,
            "RoutingOverhead": overhead_normalized
        }
        return self.metrics

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
