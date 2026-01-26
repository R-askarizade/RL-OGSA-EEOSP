import numpy as np
import math
from typing import Dict, List, Optional, Union, Tuple

from sensor_node import SensorNode


class RoutingManager:
    """
    # TODO -> FILL DOCUMENTATION
    """

    def __init__(
        self,
        nodes: List['SensorNode'],
        energy_model: 'EnergyModel',
        mode: str = "multi-hop",
        area_size: Tuple[int, int] = (100, 100),

        # Optimal sink selection mode
        policy: str = 'load_aware',
        num_sinks: bool = False,
        sink_status_table: Optional[Dict['MobileSink', dict]] = None,

        weight_distance: float = 0.5,
        weight_energy: float = 0.2,
        weight_load: float = 0.15,
        weight_trust: float = 0.15,
        comm_range: float = 40.0,
        multipath: bool = True,
        seed: int = 42
    ):
        if mode not in {"single-hop", "multi-hop"}:
            raise ValueError("Mode must be 'single-hop' or 'multi-hop'")
        self.nodes = [n for n in nodes if n.is_alive()
                      and n.has_known_position()]
        self.energy_model = energy_model
        self.mode = mode
        self.area_width, self.area_height = area_size[0], area_size[1]

        self.comm_range = comm_range
        self.wd = weight_distance
        self.we = weight_energy
        self.wl = weight_load
        self.wt = weight_trust

        # Multi sinks support
        self.policy = policy
        self.num_sinks = num_sinks
        self.sink_status_table = sink_status_table

        self.multipath = multipath
        self.load_count: Dict[int, int] = {n.id: 0 for n in self.nodes}
        self.trust_score: Dict[int, float] = {n.id: 1.0 for n in self.nodes}

        self.total_retransmissions = 0

        self.seed = seed
        np.random.seed(self.seed)

    def update_trust(self, node_id: int, success: bool):
        """Update trust score based on transmission success."""
        if success:
            self.trust_score[node_id] = min(
                1.0, self.trust_score[node_id] + 0.05)
        else:
            self.trust_score[node_id] = max(
                0.1, self.trust_score[node_id] - 0.1)

    def transmit_with_retransmit(
        self,
        sender: 'SensorNode',
        receiver: Union['SensorNode', Tuple[float, float]],
        data_size: int,
        packet_loss_prob: float,
        max_attempts: int = 2
    ) -> tuple[bool, int]:
        """
        # TODO -> FILL DOCUMENTATIONS
        """
        if not sender.is_alive():
            return False, 0

        # Handle sink position (no energy consumption for sink)
        is_sink = isinstance(receiver, tuple) or (
            hasattr(receiver, 'id') and receiver.id == -1)

        for attempt in range(1, max_attempts+1):

            if isinstance(receiver, SensorNode):
                dist = sender.distance_to(receiver)
            else:
                dist = math.hypot(
                    sender.x - receiver[0], sender.y - receiver[1])

            self.energy_model.consume_tx(sender, dist, data_size)
            if not sender.is_alive():
                return False, attempt

            if not is_sink and isinstance(receiver, SensorNode):
                self.energy_model.consume_rx(receiver, data_size)
                if not receiver.is_alive():
                    return False, attempt

            # ACK phase
            ack_loss_prob = min(0.2, packet_loss_prob / 2)
            if self.energy_model.include_ack_energy and (not is_sink) and isinstance(receiver, SensorNode):
                # receiver transmits ACK
                self.energy_model.consume_ack_tx(receiver, dist)
                if not receiver.is_alive():
                    return False, attempt
                # sender receives ACK
                self.energy_model.consume_ack_rx(sender)
                if not sender.is_alive():
                    return False, attempt
                # ACK success?
                if np.random.rand() <= ack_loss_prob:
                    # ACK lost → retransmit DATA
                    continue

            # Energy consumption for routing
            if np.random.rand() > packet_loss_prob:
                return True, attempt

        return False, max_attempts

    def compute_cost(self, src: 'SensorNode', dst) -> float:
        if isinstance(dst, (tuple, list)):
            dx, dy = dst[0], dst[1]
        else:
            dx, dy = dst.x, dst.y

        dist = math.hypot(src.x - dx, src.y - dy)
        if dist > self.comm_range:
            return float('inf')

        # Energy factor
        dst_energy = getattr(dst, 'energy', None)
        e_factor = 1.0 / \
            (dst_energy + 1e-12) if dst_energy is not None else 1.0

        # Load factor
        dst_id = getattr(dst, 'id', None)
        load = self.load_count.get(dst_id, 0)
        l_factor = 1.0 + load / 10.0

        # Trust factor
        t_score = self.trust_score.get(dst_id, 0.5)
        t_factor = 1.0 / (t_score + 1e-6)  # lower cost for high-trust nodes

        return (
            self.wd * (dist / self.comm_range) +
            self.we * e_factor +
            self.wl * l_factor +
            self.wt * t_factor
        )

    def _find_k_best_paths(
        self,
        src: 'SensorNode',
        dst: 'SensorNode',
        domain: List['SensorNode'],
        k: int = 2
    ) -> List[List['SensorNode']]:
        """
        Find K best disjoint paths.
        """
        if not self.multipath or k == 1:
            return [self._find_multihop_path(src, dst, domain)]

        paths = []
        excluded_nodes = set()
        for _ in range(k):
            temp_domain = [n for n in domain if n.id not in excluded_nodes]
            path = self._find_multihop_path(src, dst, temp_domain)
            if len(path) <= 1 or path[-1].id != dst.id:
                break
            paths.append(path)
            # Exclude intermediate nodes for next path (disjoint)
            excluded_nodes.update(n.id for n in path[1:-1])
        return paths if paths else [self._find_multihop_path(src, dst, domain)]

    def choose_sink_for_node(self,
                             node: 'SensorNode',
                             sinks: Union['MobileSink', List['MobileSink']],
                             alpha: float = 0.6, beta: float = 0.4) -> Tuple[Tuple[float, float], 'MobileSink']:
        """
        Choose sink position and sink object according to policy.
        policy: 'nearest', 'load_aware', 'balanced'
        alpha, beta: weights for distance vs. load (only used by load_aware)
        Returns (sink_pos, sink_obj)
        """
        if not isinstance(sinks, list):
            return sinks.get_position(), sinks

        # Compute candidate info
        candidates = []
        # field diagonal normalization
        field_diag = math.hypot(self.area_width, self.area_height) if hasattr(
            self, 'area_width') else 1.0

        for s in sinks:
            pos = s.get_position()
            dist = math.hypot(node.x - pos[0], node.y - pos[1])

            if getattr(self, 'sink_status_table', None) is not None and s in self.sink_status_table:
                advert = self.sink_status_table[s]
                load_norm = advert.get(
                    "current_load", 0.0) / max(1.0, advert.get("capacity", 1.0))
            else:
                load_norm = (getattr(s, "current_load", 0.0) /
                             max(1.0, getattr(s, "capacity", 1.0)))

            # Normalize distance
            dist_norm = dist / (field_diag + 1e-9)

            candidates.append((s, pos, dist, dist_norm, load_norm))

        if self.policy == 'nearest':
            s, pos, *_ = min(candidates, key=lambda t: t[2])  # min dist
            return pos, s

        if self.policy == 'load_aware':
            # cost = alpha * dist_norm + beta * load_norm
            s, pos, *_ = min(candidates, key=lambda t: alpha *
                             t[3] + beta * t[4])
            return pos, s

        # balanced: prefer sinks with spare capacity, within some threshold distance
        if self.policy == 'balanced':
            # choose sinks under 80% utilization first, else fallback nearest
            underloaded = [c for c in candidates if c[4] < 0.8]
            if underloaded:
                s, pos, *_ = min(underloaded, key=lambda t: t[2])
                return pos, s
            s, pos, *_ = min(candidates, key=lambda t: t[2])
            return pos, s

        # default fallback
        underloaded = [c for c in candidates if c[4] < 0.8]
        if underloaded:
            s, pos, *_ = min(underloaded, key=lambda t: t[2])
            return pos, s
        s, pos, *_ = min(candidates, key=lambda t: t[2])
        return pos, s

    def route_to_sink(self, ch: 'SensorNode', sinks: Union['MobileSink', List['MobileSink']], max_hops: int = 10) -> bool:
        """
        # TODO -> FILL DOCUMENTATIONS
        """
        if not (ch.is_alive() and ch.has_known_position()):
            return False

        # choose sink
        if isinstance(sinks, list):
            sink_pos, sink_obj = self.choose_sink_for_node(ch, sinks)
        else:
            sink_pos = sinks.get_position()
            sink_obj = sinks

        sink_x, sink_y = sink_pos
        dist_to_sink = math.hypot(ch.x - sink_x, ch.y - sink_y)
        packet_loss_prob = min(
            0.4, 0.1 + 0.3 * (dist_to_sink / self.comm_range))

        # Create a dummy node for the sink to handle energy calculations
        dummy_sink_node = SensorNode(-1, sink_x, sink_y, init_energy=1e9)

        if self.mode == "single-hop" or dist_to_sink <= self.comm_range:
            # Single-hop transmission with retransmission
            success, attempts = self.transmit_with_retransmit(
                sender=ch,
                receiver=dummy_sink_node,
                data_size=ch.data_packet_size,
                packet_loss_prob=packet_loss_prob,
                max_attempts=2
            )
            self.total_retransmissions += (attempts - 1)
            self.update_trust(ch.id, success)
            return success and ch.is_alive()

        # Multi-hop transmission
        current = ch
        visited = {ch.id}
        current_dist_to_sink = dist_to_sink
        hops = 0

        while hops < max_hops:
            candidates = [
                n for n in self.nodes
                if n.id not in visited
                and n.is_alive()
                and math.hypot(n.x - current.x, n.y - current.y) <= self.comm_range
            ]
            if not candidates:
                break

            best_candidate = None
            best_score = float('inf')

            for c in candidates:
                d_c_to_sink = math.hypot(c.x - sink_x, c.y - sink_y)
                if d_c_to_sink >= current_dist_to_sink - 1e-6:
                    continue

                link_cost = self.compute_cost(current, c)
                if link_cost == float('inf'):
                    continue

                score = 0.6 * link_cost + 0.4 * (d_c_to_sink / self.comm_range)
                if score < best_score:
                    best_score = score
                    best_candidate = c

            if best_candidate is None:
                break

            # Transmit to the next hop with retransmission
            dist_tx = math.hypot(current.x - best_candidate.x,
                                 current.y - best_candidate.y)
            packet_loss_prob_hop = min(
                0.4, 0.1 + 0.3 * (dist_tx / self.comm_range))
            success, attempts = self.transmit_with_retransmit(
                sender=current,
                receiver=best_candidate,
                data_size=current.data_packet_size,
                packet_loss_prob=packet_loss_prob_hop,
                max_attempts=2
            )
            self.total_retransmissions += (attempts - 1)
            self.update_trust(current.id, success)
            if success:
                self.load_count[best_candidate.id] += 1

            if not current.is_alive():
                return False
            if not best_candidate.is_alive():
                return False

            if math.hypot(best_candidate.x - sink_x, best_candidate.y - sink_y) <= self.comm_range:
                # Final hop to sink — recompute loss prob for THIS link
                dist_final = math.hypot(
                    best_candidate.x - sink_x, best_candidate.y - sink_y)
                packet_loss_prob_final = min(
                    0.4, 0.1 + 0.3 * (dist_final / self.comm_range))
                final_success, attempts = self.transmit_with_retransmit(
                    sender=best_candidate,
                    receiver=dummy_sink_node,
                    data_size=best_candidate.data_packet_size,
                    packet_loss_prob=packet_loss_prob_final,
                    max_attempts=2
                )

                self.total_retransmissions += (attempts - 1)
                self.update_trust(best_candidate.id, final_success)
                return final_success and best_candidate.is_alive()

            visited.add(best_candidate.id)
            current = best_candidate
            current_dist_to_sink = math.hypot(
                current.x - sink_x, current.y - sink_y)
            hops += 1

        # Fallback: attempt direct transmission to sink if multi-hop fails
        success, attempts = self.transmit_with_retransmit(
            sender=current,
            receiver=dummy_sink_node,
            data_size=current.data_packet_size,
            packet_loss_prob=packet_loss_prob,
            max_attempts=2
        )
        self.total_retransmissions += (attempts - 1)
        self.update_trust(current.id, success)
        return success and current.is_alive()

    def route_to_ch(
        self,
        node: 'SensorNode',
        cluster_head: 'SensorNode',
        cluster_members: Optional[List['SensorNode']] = None
    ) -> bool:
        """
        Route data from a cluster member (node) to its cluster head (cluster_head).
        This function handles the CM-to-CH communication, potentially using multi-hop within the cluster.
        It does NOT handle data aggregation energy (Eda) - that is handled by the CH after receiving data.
        """
        if not (node.is_alive() and node.has_known_position()):
            return False
        if not (cluster_head.is_alive() and cluster_head.has_known_position()):
            return False

        dist = node.distance_to(cluster_head)
        packet_loss_prob = min(0.4, 0.1 + 0.3 * (dist / self.comm_range))

        if self.mode == "single-hop":
            # Direct transmission from CM to CH
            success, attempts = self.transmit_with_retransmit(
                sender=node,
                receiver=cluster_head,
                data_size=node.data_packet_size,
                packet_loss_prob=packet_loss_prob,
                max_attempts=2
            )
            self.total_retransmissions += (attempts - 1)
            # Trust updated for the sender (CM)
            self.update_trust(node.id, success)
            # Load updated for the receiver (CH)
            if success and cluster_head.is_alive():
                self.load_count[cluster_head.id] += 1
            return success and node.is_alive() and cluster_head.is_alive()

        # Multi-hop transmission within the cluster
        routing_domain = cluster_members if cluster_members is not None else self.nodes
        routing_domain = [
            n for n in routing_domain if n.is_alive() and n.has_known_position()]

        paths = self._find_k_best_paths(
            node, cluster_head, routing_domain, k=2)
        best_path = paths[0]

        if len(best_path) < 2 or best_path[-1].id != cluster_head.id:
            return False

        for i in range(len(best_path) - 1):
            # s = sender, d = receiver in this hop
            s, d = best_path[i], best_path[i + 1]
            if not (s.is_alive() and d.is_alive()):
                return False
            dist = s.distance_to(d)
            packet_loss_prob = min(0.4, 0.1 + 0.3 * (dist / self.comm_range))
            success, attempts = self.transmit_with_retransmit(
                sender=s,
                receiver=d,
                data_size=s.data_packet_size,
                packet_loss_prob=packet_loss_prob,
                max_attempts=2
            )
            self.total_retransmissions += (attempts - 1)
            # Trust updated for the sender of this hop
            self.update_trust(s.id, success)
            # Load updated for the receiver of this hop
            if success and d.is_alive():
                self.load_count[d.id] += 1
            # Check if sender died during transmission
            if not s.is_alive():
                return False
        return True

    def _find_multihop_path(
        self,
        src: 'SensorNode',
        dst: 'SensorNode',
        domain: List['SensorNode']
    ) -> List['SensorNode']:
        if src.id == dst.id:
            return [src]
        path = [src]
        current = src
        visited = {src.id}
        while current.id != dst.id:
            neighbors = [
                n for n in domain
                if n.id not in visited
                and n.is_alive()
                and current.distance_to(n) <= self.comm_range
            ]
            if not neighbors:
                break
            next_node = min(
                neighbors, key=lambda n: self.compute_cost(current, n))
            path.append(next_node)
            visited.add(next_node.id)
            current = next_node
            if current.id == dst.id:
                break
        if path[-1].id != dst.id:
            path.append(dst)
        return path

    def reset_loads(self):
        self.load_count = {n.id: 0 for n in self.nodes}
