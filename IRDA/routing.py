import random
import numpy as np
from optimizer import KNNTrafficPredictor, IRDAOptimizer
from config import Config, Metrics
from energy_model import EnergyModel
from nodes import Node


class RoutingProtocol:
    def __init__(self, network):
        self.net = network
        self.predictor = KNNTrafficPredictor()
        self.predictor.train([random.uniform(10, 20) for _ in range(50)])

        self.ms_pos = np.array([0.0, Config.SINK_PATH_Y])
        self.ms_velocity = 10
        self.stats = Metrics()

        self.nodes = [Node(i) for i in range(Config.NUM_NODES)]

        # State for stable clusters
        self.current_chs = []
        self.cluster_valid_rounds = 0

    def perform_round(self, round_num):
        alive_count = np.sum(self.net.alive)
        self.stats.update_node_status(round_num, self.net.alive)

        if alive_count == 0:
            return False

        # 1. Move MS
        self.ms_velocity = max(5, min(25, np.random.normal(15, 5)))
        self.ms_pos[0] += self.ms_velocity * Config.PACKET_INTERVAL
        if self.ms_pos[0] > Config.FIELD_LENGTH:
            self.ms_pos[0] = 0

        # 2. Check if we need to re-cluster
        if self.cluster_valid_rounds <= 0 or not self.current_chs:
            optimizer = IRDAOptimizer(self.net, self.ms_pos[0], self.ms_pos[1])
            candidate_chs = optimizer.run()
            self.current_chs = self.clean_chs(candidate_chs)
            self.cluster_valid_rounds = Config.CLUSTER_ROUNDS
        else:
            self.cluster_valid_rounds -= 1
            # Filter out any CHs that have died since the last clustering
            self.current_chs = [
                ch for ch in self.current_chs if self.net.alive[ch]]

        # 3. Generate packets stochastically
        for i in range(Config.NUM_NODES):
            if self.net.alive[i] and round_num >= self.nodes[i].next_data_gen_round:
                self.nodes[i].generate_packet(round_num)
                self.nodes[i].schedule_next_data_gen(round_num)
                self.stats.total_packets_generated += 1

        # 4. Routing & Transmission
        if not self.current_chs:
            return True

        self.transmit_phase(self.current_chs, round_num)
        return True

    def clean_chs(self, chs):
        if not chs:
            return []
        chs = sorted(chs, key=lambda i: self.net.energies[i], reverse=True)
        keep = []
        covered = set()

        ch_indices = np.array(chs)
        sub_dist = self.net.dist_matrix[ch_indices][:, ch_indices]

        for i, ch in enumerate(chs):
            if ch in covered:
                continue
            keep.append(ch)
            neighbors = sub_dist[i] < Config.D_0
            for idx in np.where(neighbors)[0]:
                covered.add(chs[idx])
        return keep

    def transmit_phase(self, chs, round_num):
        if not chs:
            return

        member_counts = np.zeros(len(self.net.alive))
        ch_arr = np.array(chs)
        alive_indices = np.where(self.net.alive)[0]

        d_cm_ch = self.net.dist_matrix[alive_indices][:, ch_arr]
        d_ch_bs = np.sqrt((self.net.coords[ch_arr, 0] - self.ms_pos[0])**2 +
                          (self.net.coords[ch_arr, 1] - self.ms_pos[1])**2)

        d_cm_ch = np.maximum(d_cm_ch, 0.1)
        d_ch_bs = np.maximum(d_ch_bs, 0.1)

        e_ch = self.net.energies[ch_arr]
        weights = e_ch / (d_cm_ch * d_ch_bs)

        best_ch_indices = np.argmax(weights, axis=1)

        bits = Config.PACKET_SIZE
        cm_dists = d_cm_ch[np.arange(len(alive_indices)), best_ch_indices]

        # Identify nodes that are CHs themselves
        is_ch_mask = np.isin(alive_indices, ch_arr)

        # Nodes that are CMs and have packets
        cm_indices = np.where(~is_ch_mask)[0]
        nodes_with_packets = np.array([i for i in cm_indices
                                      if len(self.nodes[alive_indices[i]].pending_packets) > 0])

        if len(nodes_with_packets) > 0:
            # 1. CM Send Energy
            e_tx = EnergyModel.calc_tx_energy(
                cm_dists[nodes_with_packets], bits)
            e_sense = bits * Config.E_SENS
            self.net.energies[alive_indices[nodes_with_packets]
                              ] -= (e_tx + e_sense)

            # 2. CH Receive & Aggregate Energy
            member_counts = np.bincount(
                best_ch_indices[nodes_with_packets], minlength=len(ch_arr))

            e_rx = member_counts * bits * Config.E_ELEC
            e_aggr = member_counts * bits * Config.E_AGGR
            self.net.energies[ch_arr] -= (e_rx + e_aggr)

        # 3. CH -> Sink Transmission
        # Only CHs that have data (either their own or from members) transmit
        transmitting_chs = []
        for i, ch_id in enumerate(ch_arr):
            # CH's own data
            has_own_data = len(self.nodes[ch_id].pending_packets) > 0
            # Data from members
            has_member_data = member_counts[i] > 0 if i < len(
                member_counts) else False
            if has_own_data or has_member_data:
                transmitting_chs.append(ch_id)

        if not transmitting_chs:
            self.net.alive = self.net.energies > 0
            return

        # Sort transmitting CHs by distance to sink for a simple chain routing
        dist_to_sink = np.sqrt((self.net.coords[transmitting_chs, 0] - self.ms_pos[0])**2 +
                               (self.net.coords[transmitting_chs, 1] - self.ms_pos[1])**2)
        sorted_indices = np.argsort(dist_to_sink)
        ordered_chs = np.array(transmitting_chs)[sorted_indices]

        # The furthest CH sends to the next furthest, and so on, until the closest CH sends to sink
        for i in range(len(ordered_chs) - 1):
            src = ordered_chs[i]
            dst = ordered_chs[i+1]
            d = self.net.get_distance(src, dst)
            e_tx = EnergyModel.calc_tx_energy(d, bits)
            e_rx = bits * Config.E_ELEC
            self.net.energies[src] -= e_tx
            self.net.energies[dst] -= e_rx

        # Final hop: Closest CH to Sink
        final_ch = ordered_chs[-1]
        d_sink = np.sqrt((self.net.coords[final_ch, 0] - self.ms_pos[0])**2 +
                         (self.net.coords[final_ch, 1] - self.ms_pos[1])**2)

        loss_prob = min(0.1, 0.1 + 0.3 * (d_sink / Config.COMM_RANGE))
        pdr = 1.0 - loss_prob

        packet_delivered = random.random() < pdr

        e_tx_sink = EnergyModel.calc_tx_energy(d_sink, bits)
        self.net.energies[final_ch] -= e_tx_sink

        if packet_delivered:
            packets_delivered = sum(
                len(self.nodes[ch].pending_packets) for ch in ordered_chs)
            self.stats.total_packets_delivered += packets_delivered

            for ch in ordered_chs:
                self.nodes[ch].pending_packets = []

            hops = len(ordered_chs)
            delay = hops * Config.HOP_DELAY
            self.stats.total_delay_sum += delay
            self.stats.delay_samples += 1

        # Update Dead Nodes
        self.net.alive = self.net.energies > 0
