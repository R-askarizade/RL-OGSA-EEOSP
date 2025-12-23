from nodes import SensorNode
from data_aggregator import USEKLMSAggregator
from config import CONFIG
from typing import Tuple
import random


# class ClusterHead(SensorNode):
#     def __init__(self, nid, pos, energy):
#         super().__init__(nid, pos, energy)
#         self.members = []
#         self.aggregator = USEKLMSAggregator(**CONFIG["use_klms"])
#         self.packet_generated = 0
#         self.packet_delivered = 0
#         self.generation_rounds = []

#     def aggregate_data(self):
#         data_vals = [node.data for node in self.members]
#         self.packet_generated = len(data_vals) + 1  # self + members
#         aggregated = self.aggregator.aggregate(data_vals + [self.data])
#         return aggregated


class ClusterHead(SensorNode):
    def __init__(self, nid: int, pos: Tuple[float, float], energy: float):
        super().__init__(nid, pos, energy)
        self.members = []  # list of SensorNode (non-CH members)
        self.aggregator = USEKLMSAggregator(**CONFIG["use_klms"])
        # Metrics
        self.total_packets_generated = 0
        self.total_packets_delivered = 0

    def collect_member_packets(self):
        """Aggregate all pending packets from self and members."""
        all_data = []

        # Self packet
        if self.pending_packets:
            all_data.extend([1.0 for _ in self.pending_packets])  # dummy data
            self.total_packets_generated += len(self.pending_packets)
            self.pending_packets.clear()

        # Member packets
        for member in self.members:
            if member.pending_packets:
                all_data.extend([1.0 for _ in member.pending_packets])
                self.total_packets_generated += len(member.pending_packets)
                member.pending_packets.clear()

        return all_data

    def aggregate_and_prepare_for_delivery(self):
        """Aggregate collected data using USE-KLMS."""
        data_list = self.collect_member_packets()
        if not data_list:
            return 0  # no packets to deliver
        # USE-KLMS reduces redundancy
        aggregated_value = self.aggregator.aggregate(data_list)
        # We treat aggregation as **1 logical packet** per round of collection
        # But count actual input packets for PDR denominator
        return len(data_list)  # return count of raw packets (for PDR)

    def deliver_with_loss(self, raw_packet_count: int, pdr: float) -> int:
        """Apply stochastic packet loss based on PDR."""
        delivered = 0
        for _ in range(raw_packet_count):
            if random.random() < pdr:
                delivered += 1
        return delivered
