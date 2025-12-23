import numpy as np
import math
from config import CONFIG
from base import gauss_map, euclidean, compute_intra_distance, compute_inter_distance, compute_Dis, compute_En
from typing import Tuple
from collections import deque
import random


class SensorNode:
    def __init__(self, nid: int, pos: Tuple[float, float], energy: float):
        self.id = nid
        self.pos = pos
        self.energy = energy
        self.alive = True
        # Stochastic data generation state
        self.pending_packets = deque()  # stores generation round numbers
        self.next_data_gen_round = 1    # first data at round 1 or later

    def generate_packet(self, current_round: int):
        """Add a new packet generated at current_round."""
        self.pending_packets.append(current_round)

    def schedule_next_data_gen(self, current_round: int, avg_interval: int = 3):
        """Schedule next data generation using uniform random interval (1 to 2*avg_interval)."""
        next_interval = random.randint(1, avg_interval * 2)
        self.next_data_gen_round = current_round + next_interval

    def should_generate_now(self, current_round: int) -> bool:
        return current_round >= self.next_data_gen_round


class BaseStation:
    def __init__(self, pos):
        self.pos = pos
        self.ch_table = {}
        self.requests = []

    def compute_trust(self, ch, all_nodes):
        # Direct trust: Eq. (27)
        dist = euclidean(ch.pos, self.pos)
        direct = ch.energy / (dist + 1e-9)

        # Indirect trust: Eq. (28)
        indirect = 0.0
        count = 0
        for nid, node in all_nodes.items():
            if nid == ch.id:
                continue
            d1 = euclidean(node.pos, ch.pos)
            d2 = euclidean(node.pos, self.pos)
            t1 = node.energy / (d1 + 1e-9)
            t2 = ch.energy / (d2 + 1e-9)
            indirect += t1 * t2
            count += 1
        indirect = indirect / count if count > 0 else 0.0

        w = CONFIG["trust_weight"]
        return w * direct + (1 - w) * indirect  # Eq. (29)

    def compute_pdr(self, ch):
        return min(1.0, ch.energy / CONFIG["initial_energy"])  # Eq. (30)

    def compute_risk(self, ch):
        sec = ch.energy / CONFIG["initial_energy"]  # proxy for security
        if sec <= 0:
            return 1.0
        elif sec <= 1:
            return 1 - math.exp(-1.5 * sec)
        elif sec <= 2:
            return 1 - math.exp(-0.5 * sec)
        else:
            return 0.0  # Eq. (31)

    def install_charger(self, ch, all_nodes):
        dist = euclidean(ch.pos, self.pos)
        trust = self.compute_trust(ch, all_nodes)
        pdr = self.compute_pdr(ch)
        risk = self.compute_risk(ch)
        self.ch_table[ch.id] = {
            "pos": ch.pos,
            "energy": ch.energy,
            "distance": dist,
            "trust": trust,
            "pdr": pdr,
            "risk": risk,
        }


class MobileVAN:
    def __init__(self, van_id, bs_pos):
        self.id = van_id
        self.bs_pos = bs_pos
        self.current_pos = bs_pos
        self.speed = CONFIG["speed"]

    def plan_route(self, ch_list):
        if not ch_list:
            return []
        unvisited = ch_list[:]
        route = []
        current = self.current_pos
        while unvisited:
            nearest = min(unvisited, key=lambda ch: euclidean(current, ch.pos))
            route.append(nearest)
            current = nearest.pos
            unvisited.remove(nearest)
        return route

    # def execute_jdcwc(self, ch_list, round_num, metrics):
    #     if not ch_list:
    #         return 0.0
    #     route = self.plan_route(ch_list)
    #     total_delay_sec = 0.0
    #     current = self.bs_pos
    #     delivered = 0

    #     for ch in route:
    #         dist = euclidean(current, ch.pos)
    #         delay_sec = dist / self.speed
    #         total_delay_sec += delay_sec

    #         # Recharge CH
    #         ch.energy = CONFIG["initial_energy"]

    #         # Aggregate and deliver data
    #         aggregated = ch.aggregate_data()

    #         # TODO : ADDING NOISE TO DELIVERING DATA PACKETS
    #         # ch.packet_delivered = ch.packet_generated
    #         # Compute distance from CH to BS (logical sink)
    #         dist_to_bs = euclidean(ch.pos, self.bs_pos)
    #         comm_range = CONFIG["comm_range"]

    #         # Adaptive packet loss probability (0.1 to 0.4)
    #         loss_prob = min(0.4, 0.1 + 0.3 * (dist_to_bs / comm_range))
    #         pdr = 1.0 - loss_prob

    #         # Apply stochastic packet loss (more realistic than deterministic rounding)
    #         # Option A: Deterministic (for reproducibility)
    #         # delivered = int(ch.packet_generated * pdr)

    #         # Option B: Stochastic (more realistic)
    #         delivered = 0
    #         for _ in range(ch.packet_generated):
    #             if np.random.rand() < pdr:
    #                 delivered += 1

    #         ch.packet_delivered = delivered

    #         metrics["total_generated"] += ch.packet_generated
    #         metrics["total_delivered"] += ch.packet_delivered
    #         metrics["total_e2e_delay_sec"] += total_delay_sec * \
    #             ch.packet_delivered
    #         delivered += ch.packet_delivered
    #         current = ch.pos

    #     # Return to BS
    #     dist_back = euclidean(current, self.bs_pos)
    #     total_delay_sec += dist_back / self.speed
    #     self.current_pos = self.bs_pos
    #     return total_delay_sec

    def execute_jdcwc(self, ch_list, round_num, metrics):
        if not ch_list:
            return
        route = self.plan_route(ch_list)
        current = self.bs_pos
        comm_range = CONFIG["comm_range"]

        for ch in ch_list:
            # Recharge first
            ch.energy = CONFIG["initial_energy"]

            # Aggregate
            raw_count = ch.aggregate_and_prepare_for_delivery()
            if raw_count == 0:
                continue

            # Compute PDR based on distance to BS (logical sink)
            dist_to_bs = euclidean(ch.pos, self.bs_pos)
            loss_prob = min(0.1, 0.1 + 0.3 * (dist_to_bs / comm_range))
            pdr = 1.0 - loss_prob

            # Stochastic delivery
            delivered = ch.deliver_with_loss(raw_count, pdr)

            # Update global metrics
            metrics["total_generated"] += raw_count
            metrics["total_delivered"] += delivered
            # For E2E delay: assume packets were generated over recent rounds,
            # but for simplicity, assign current round as collection time
            # or estimate based on packet timestamps
            metrics["total_e2e_delay_sec"] += 0.0

            current = ch.pos
