import math

CONFIG = {
    "data_gen_avg_interval": 11,

    "field_size": (100, 100),
    "bs_pos": (50, 50),
    "num_nodes": 100,
    "num_vans": 10,
    "initial_energy": 0.5,  # Joules
    "E_elec": 50e-9,
    "E_fs": 10e-12,
    "E_mp": 0.0013e-12,
    "E_da": 5e-9,
    "packet_size_bits": 4000,
    "max_rounds": 60000,
    "threshold_energy": 0.1,
    "pubmo": {
        "pop_size": 10,
        "max_iter": 50,
        "teams": 5,
    },
    "use_klms": {
        "eta": 0.1,
        "alpha": 1.0,
        "c": 0.5,
        "sigma": 1.0,
    },
    "trust_weight": 0.7,  # ϖ in Eq. (29)
    "gauss_map_init": 0.37,
    "speed": 5.0,  # m/s (assumed, not in paper but needed for delay)

    "comm_range": 50,
}
# d0 = math.sqrt(10e-12 / (0.0013e-12))  # ≈ 87.7 m
# CONFIG["comm_range"] = d0  # ~87.7 meters
