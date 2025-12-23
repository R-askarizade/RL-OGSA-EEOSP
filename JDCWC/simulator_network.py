from nodes import BaseStation, MobileVAN, SensorNode
from optimization import PUBMO
from CH_selection import ClusterHead
from config import CONFIG
from base import gauss_map, euclidean, compute_intra_distance, compute_inter_distance, compute_Dis, compute_En
import math
import numpy as np
import random


class JDCWCSimulator:
    def __init__(self):
        self.nodes = self._deploy_nodes()
        self.bs = BaseStation(CONFIG["bs_pos"])
        # CREATE MULTIPLE VANS (num_vans = 10)
        self.vans = [
            MobileVAN(van_id=i, bs_pos=CONFIG["bs_pos"])
            for i in range(CONFIG["num_vans"])
        ]
        self.metrics = {
            "total_generated": 0,
            "total_delivered": 0,
            "total_e2e_delay_sec": 0.0,
            "fnd": None,
            "hnd": None,
            "lnd": None,
            "alive_history": [],
        }
        self.total_nodes = CONFIG["num_nodes"]

    def _deploy_nodes(self):
        nodes = {}
        for i in range(CONFIG["num_nodes"]):
            x = np.random.uniform(0, CONFIG["field_size"][0])
            y = np.random.uniform(0, CONFIG["field_size"][1])
            nodes[i] = SensorNode(i, (x, y), CONFIG["initial_energy"])
        return nodes

    def run(self):

        for rnd in range(1, CONFIG["max_rounds"] + 1):
            alive_nodes = [n for n in self.nodes.values()
                           if n.energy > 0 and n.alive]

            for node in alive_nodes:
                if node.should_generate_now(rnd):
                    node.generate_packet(rnd)
                    node.schedule_next_data_gen(
                        rnd, avg_interval=CONFIG["data_gen_avg_interval"])

            num_alive = len(alive_nodes)
            self.metrics["alive_history"].append(num_alive)

            # Record FND, HND, LND
            if self.metrics["fnd"] is None and num_alive < self.total_nodes:
                self.metrics["fnd"] = rnd
            if self.metrics["hnd"] is None and num_alive <= self.total_nodes // 2:
                self.metrics["hnd"] = rnd
            if num_alive == 0 and self.metrics["lnd"] is None:
                self.metrics["lnd"] = rnd
                break
            if num_alive < 3:
                if self.metrics["lnd"] is None:
                    self.metrics["lnd"] = rnd
                break

            # Clustering
            pos_list = [n.pos for n in alive_nodes]
            energy_list = [n.energy for n in alive_nodes]
            k = max(1, int(0.1 * len(alive_nodes)))
            pubmo = PUBMO(pos_list, energy_list, **CONFIG["pubmo"])
            try:
                ch_indices = pubmo.optimize(k)
                ch_nodes = [alive_nodes[i] for i in ch_indices]
            except:
                ch_nodes = alive_nodes[:k]

            # Create CH objects
            ch_objects = []
            for node in ch_nodes:
                ch_obj = ClusterHead(node.id, node.pos, node.energy)
                ch_objects.append(ch_obj)

            # Assign members
            for node in alive_nodes:
                if node.id in [ch.id for ch in ch_objects]:
                    continue
                if ch_objects:
                    closest_ch = min(
                        ch_objects, key=lambda ch: euclidean(node.pos, ch.pos))
                    closest_ch.members.append(node)

            # Identify CHs needing service
            service_chs = []
            for ch in ch_objects:
                if ch.energy < CONFIG["threshold_energy"]:
                    self.bs.install_charger(ch, self.nodes)
                    service_chs.append(ch)

            # DISTRIBUTE REQUESTS AMONG 10 VANS
            if service_chs:
                # Simple round-robin or chunk assignment
                van_loads = [[] for _ in range(CONFIG["num_vans"])]
                for idx, ch in enumerate(service_chs):
                    van_idx = idx % CONFIG["num_vans"]
                    van_loads[van_idx].append(ch)

                # Each VAN executes its assigned tasks
                for van_id, ch_batch in enumerate(van_loads):
                    if ch_batch:
                        self.vans[van_id].execute_jdcwc(
                            ch_batch, rnd, self.metrics)

            # Energy consumption model (standard first-order radio)
            for node in alive_nodes:
                dist_to_bs = euclidean(node.pos, self.bs.pos)
                if node.id in [ch.id for ch in ch_objects]:
                    # CH: transmit aggregated packet to BS
                    energy_tx = (
                        CONFIG["E_elec"] * CONFIG["packet_size_bits"]
                        + CONFIG["E_fs"] *
                        CONFIG["packet_size_bits"] * (dist_to_bs ** 2)
                    )
                    node.energy -= energy_tx + \
                        CONFIG["E_da"] * CONFIG["packet_size_bits"]
                else:
                    # Member: transmit to its CH
                    if ch_objects:
                        closest_ch = min(
                            ch_objects, key=lambda ch: euclidean(node.pos, ch.pos))
                        dist_to_ch = euclidean(node.pos, closest_ch.pos)
                        energy_tx = (
                            CONFIG["E_elec"] * CONFIG["packet_size_bits"]
                            + CONFIG["E_fs"] *
                            CONFIG["packet_size_bits"] * (dist_to_ch ** 2)
                        )
                        node.energy -= energy_tx
                if node.energy <= 0:
                    node.alive = False

        # Finalize metrics
        total_gen = self.metrics["total_generated"]
        total_del = self.metrics["total_delivered"]
        self.metrics["pdr"] = total_del / total_gen if total_gen > 0 else 0.0
        self.metrics["avg_e2e_delay_sec"] = (
            self.metrics["total_e2e_delay_sec"] /
            total_del if total_del > 0 else 0.0
        )
        # Assume 1 round = 1 second
        self.metrics["avg_e2e_delay_rounds"] = self.metrics["avg_e2e_delay_sec"]

        if self.metrics["lnd"] is None:
            self.metrics["lnd"] = CONFIG["max_rounds"]

        return {
            "FND": self.metrics["fnd"] or CONFIG["max_rounds"],
            "HND": self.metrics["hnd"] or CONFIG["max_rounds"],
            "LND": self.metrics["lnd"],
            "Total_Packet_Generated": self.metrics["total_generated"],
            "Total_Packet_Delivered": self.metrics["total_delivered"],
            "PDR": self.metrics["pdr"],
            "Avg_E2E_Delay_Rounds": self.metrics["avg_e2e_delay_rounds"],
            "Avg_E2E_Delay_Sec": self.metrics["avg_e2e_delay_sec"],
        }


def run_multiple_simulations(n_runs):
    FND, HND, LND, PDR = [], [], [], []
    TotalGenerated = []
    TotalDelivered = []
    Avg_E2E_Delay_Rounds = []
    for seed in range(n_runs):
        np.random.seed(seed)
        random.seed(seed)
        sim = JDCWCSimulator()
        metrics = sim.run()
        FND.append(metrics['FND'])
        HND.append(metrics['HND'])
        LND.append(metrics['LND'])
        TotalGenerated.append(metrics['Total_Packet_Generated'])
        TotalDelivered.append(metrics['Total_Packet_Delivered'])
        PDR.append(metrics['PDR'])

        print(
            f"Run {seed + 1}: \n LND = {metrics['LND']} \n TotalGenerated = {metrics['Total_Packet_Generated']} \n TotalDelivered = {metrics['Total_Packet_Delivered']} \n PDR = {metrics['PDR']}")

    return [np.array(FND), np.array(HND), np.array(LND), np.array(TotalGenerated), np.array(TotalDelivered), np.array(PDR)]


if __name__ == "__main__":
    # Set seed for reproducibility
    result_metrics = run_multiple_simulations(30)

print(
    f"Mean result FND: {result_metrics[0].mean():.2f} ± {result_metrics[0].std():.2f}")
print(
    f"Mean result HND: {result_metrics[1].mean():.2f} ± {result_metrics[1].std():.2f}")
print(
    f"Mean result LND: {result_metrics[2].mean():.2f} ± {result_metrics[2].std():.2f}")
print(
    f"Mean result total data generated: {result_metrics[3].mean():.2f} ± {result_metrics[3].std():.2f}")
print(
    f"Mean result total data delivered: {result_metrics[4].mean():.2f} ± {result_metrics[4].std():.2f}")
print(
    f"Mean result PDR: {result_metrics[5].mean():.2f} ± {result_metrics[5].std():.2f}")


"""
Mean result FND: 1415.77 ± 108.38
Mean result HND: 2304.80 ± 32.89
Mean result LND: 2463.17 ± 13.34
Mean result total data generated: 17398.63 ± 481.90
Mean result total data delivered: 15660.60 ± 441.40
Mean result PDR: 0.90 ± 0.00
"""
