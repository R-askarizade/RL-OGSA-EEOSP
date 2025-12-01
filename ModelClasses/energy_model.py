import numpy as np
from dataclasses import dataclass

import ConfigClass.config
from ModelClasses.sensor_node import SensorNode


@dataclass
class EnergyModel:
    """
    # TODO -> FILL DOCUMENTATION
    """

    E_elec: float = 50e-9        # Energy for electronics (J/bit)
    E_fs: float = 10e-12         # Free-space amplifier energy (J/bit/m^2)
    E_mp: float = 0.0013e-12     # Multipath amplifier energy (J/bit/m^4)
    E_da: float = 5e-9           # Data aggregation energy (J/bit)
    packet_size: int = 4000      # Default bits per message

    def __post_init__(self):
        if self.E_mp <= 0:
            raise ValueError("E_mp must be positive")
        self._d0 = np.sqrt(self.E_fs / self.E_mp)

    @property
    def d0(self) -> float:
        """Threshold distance between free-space and multipath models."""
        return self._d0

    def tx_energy(self, distance: float, bits: int = None) -> float:
        """Compute transmission energy for 'bits' over 'distance' meters."""
        bits = bits if bits is not None else self.packet_size
        if distance < 0:
            raise ValueError("Distance must be non-negative")
        if distance < self.d0:
            return bits * (self.E_elec + self.E_fs * (distance ** 2))
        else:
            return bits * (self.E_elec + self.E_mp * (distance ** 4))

    def rx_energy(self, bits: int = None) -> float:
        """Compute reception energy."""
        bits = bits if bits is not None else self.packet_size
        return bits * self.E_elec

    def da_energy(self, bits: int = None) -> float:
        """Compute data aggregation energy."""
        bits = bits if bits is not None else self.packet_size
        return bits * self.E_da

    def consume_tx(self, node: 'SensorNode', distance: float, bits: int = None) -> float:
        """Consume energy for transmission and update node's energy."""
        e = self.tx_energy(distance, bits)
        node.energy = max(0.0, node.energy - e)
        # Optional: update node.alive if your SensorNode supports it
        if hasattr(node, 'alive') and node.energy <= 0:
            node.alive = False
        return e

    def consume_rx(self, node: 'SensorNode', bits: int = None) -> float:
        """Consume energy for reception and update node's energy."""
        e = self.rx_energy(bits)
        node.energy = max(0.0, node.energy - e)
        if hasattr(node, 'alive') and node.energy <= 0:
            node.alive = False
        return e

    def consume_da(self, node: 'SensorNode', bits: int = None) -> float:
        """Consume energy for data aggregation and update node's energy."""
        e = self.da_energy(bits)
        node.energy = max(0.0, node.energy - e)
        if hasattr(node, 'alive') and node.energy <= 0:
            node.alive = False
        return e
