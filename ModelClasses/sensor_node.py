from typing import Optional, List, Tuple, Union
import math
import random

import ConfigClass.config


class SensorNode:

    def __init__(
        self,
        node_id: int,
        x: Optional[float] = None,
        y: Optional[float] = None,
        init_energy: float = 1.0,
        comm_range: float = 50.0,
        area_size: Tuple[float, float] = (500.0, 500.0),
    ):
        """
        # TODO -> FILL DOCUMENTATION
        """
        self.id = node_id
        self.x = x
        self.y = y
        self.init_energy = float(init_energy)
        self.energy = float(init_energy)
        self.comm_range = float(comm_range)
        self.area_size = area_size
        self.alive = self.energy > 0

        self.is_cluster_head: bool = False
        self.cluster_id: Optional[int] = None
        self.next_data_gen_round = 0

        # Statistics
        self.packets_sent = 0
        self.packets_received = 0
        self.schedule_next_data_gen(current_round=0)

        # list of generation rounds for undelivered packets
        self.pending_packets: List[int] = []
        # generation rounds of packets waiting to be sent to sink
        self.buffered_packets: List[int] = []

        # max total packets (pending + buffered) - for: Buffer Overflow, Traffic Load
        self.buffer_size = 10

    def generate_packet(self, current_round: int):
        """Add a new packet generated at current_round."""
        self.pending_packets.append(current_round)

    def schedule_next_data_gen(self, current_round: int, avg_interval: int = 3):
        """Schedule the next data generation round based on an exponential-like random interval."""
        next_interval = random.randint(1, avg_interval*2)
        self.next_data_gen_round = current_round + next_interval

    def position(self) -> Optional[Tuple[float, float]]:
        """Return (x, y) if known, else None."""
        if self.x is None or self.y is None:
            return None
        return (self.x, self.y)

    def has_known_position(self) -> bool:
        """Check if node's position is known."""
        return self.position() is not None

    def distance_to(self, other: Union['SensorNode', Tuple[float, float]]) -> float:
        """Compute Euclidean distance to another node or point."""
        if not self.has_known_position():
            raise ValueError("This node's position is unknown")

        if isinstance(other, SensorNode):
            if not other.has_known_position():
                raise ValueError("Other node's position is unknown")
            ox, oy = other.x, other.y
        else:
            ox, oy = other

        return math.hypot(self.x - ox, self.y - oy)

    def become_cluster_head(self, cluster_id: int):
        """Mark this node as cluster head for a given cluster."""
        self.is_cluster_head = True
        self.cluster_id = cluster_id

    def leave_cluster_head(self):
        """Remove cluster head status."""
        self.is_cluster_head = False
        self.cluster_id = None

    def is_alive(self) -> bool:
        return self.alive

    def __repr__(self):
        pos = self.position()
        pos_str = f"({pos[0]:.2f},{pos[1]:.2f})" if pos is not None else "(unknown)"
        ch_str = ", CH=True" if self.is_cluster_head else ""
        cl_str = f", CL={self.cluster_id}" if self.cluster_id is not None else ""
        return f"SensorNode(id={self.id}, pos={pos_str}, E={self.energy:.6f}, alive={self.alive}{ch_str}{cl_str})"
