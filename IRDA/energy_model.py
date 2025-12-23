from config import Config
import numpy as np


class EnergyModel:
    @staticmethod
    def calc_tx_energy(dist, bits):
        e_tx = bits * Config.E_ELEC
        amp = np.where(dist <= Config.D_0,
                       Config.E_FS * (dist**2),
                       Config.E_MP * (dist**4))
        return e_tx + (bits * amp)
