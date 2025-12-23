import math
import numpy as np


def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def gauss_map(x):
    if x == 0:
        return 0.0
    return (1.0 / x) % 1.0


def compute_intra_distance(node_positions, ch_positions):
    if not ch_positions or not node_positions:
        return float('inf')
    total = 0.0
    for node in node_positions:
        dists = [euclidean(node, ch) for ch in ch_positions]
        total += min(dists)
    return total / len(node_positions)  # Eq. (3)


def compute_inter_distance(ch_positions):
    m = len(ch_positions)
    if m <= 1:
        return 0.0
    total = 0.0
    count = 0
    for i in range(m):
        for j in range(i + 1, m):
            total += euclidean(ch_positions[i], ch_positions[j])
            count += 1
    return total / count if count > 0 else 0.0  # Eq. (2)


def compute_Dis(node_positions, ch_positions):
    intra = compute_intra_distance(node_positions, ch_positions)
    inter = compute_inter_distance(ch_positions)
    return (intra + inter) / 2.0  # as described in text below Eq. (6)


def compute_En(energies):
    return np.mean(energies) if energies else 0.0  # Eq. (4)


def compute_D(dis, speed=5.0):
    return dis / speed  # Eq. (5)
