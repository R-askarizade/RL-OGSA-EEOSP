import math
import numpy as np


class Config:
    # Network Parameters
    FIELD_LENGTH = 100
    FIELD_WIDTH = 100
    NUM_NODES = 100
    SINK_PATH_Y = 110

    # Energy Model Parameters
    E_0 = 0.5            # Initial Energy (J)
    E_ELEC = 50e-9       # 50 nJ/bit
    E_FS = 10e-12        # Free space
    E_MP = 0.0013e-12    # Multipath
    E_SENS = 1e-9
    E_AGGR = 5e-9
    D_0 = math.sqrt(E_FS / E_MP)

    # Data Packet
    PACKET_SIZE = 4000   # bits
    PACKET_INTERVAL = 30  # seconds

    # Delay Constants (Assumed for metrics)
    HOP_DELAY = 0.005

    # IRDA Parameters
    IRDA_POPULATION = 40
    IRDA_MAX_ITER = 30
    NUM_CHS = 10         # Target CHs
    ALPHA = 0.8
    BETA = 0.2
    GAMMA = 0.4
    DELTA = 0.99
    ZETA = 0.6

    # Communication Range
    COMM_RANGE = 30

    # Cluster stability period
    CLUSTER_ROUNDS = 20  # Re-cluster every 20 rounds


class Metrics:
    def __init__(self):
        self.dead_nodes = []
        self.fnd = -1
        self.hnd = -1
        self.lnd = -1
        self.total_packets_generated = 0
        self.total_packets_delivered = 0
        self.total_delay_sum = 0
        self.delay_samples = 0

    def update_node_status(self, round_num, alive_array):
        num_dead = Config.NUM_NODES - np.sum(alive_array)
        if self.fnd == -1 and num_dead >= 1:
            self.fnd = round_num
        if self.hnd == -1 and num_dead >= Config.NUM_NODES / 2:
            self.hnd = round_num
        if self.lnd == -1 and num_dead == Config.NUM_NODES:
            self.lnd = round_num
