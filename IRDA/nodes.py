import random


class Node:
    def __init__(self, node_id):
        self.id = node_id
        self.pending_packets = []
        self.next_data_gen_round = random.randint(1, 6)

    def generate_packet(self, current_round: int):
        self.pending_packets.append(current_round)

    def schedule_next_data_gen(self, current_round: int, avg_interval: int = 5):
        next_interval = random.randint(1, avg_interval*2)
        self.next_data_gen_round = current_round + next_interval
