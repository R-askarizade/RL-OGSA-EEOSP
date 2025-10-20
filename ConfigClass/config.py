import random
import numpy as np
from typing import List
from matplotlib import pyplot as plt

def plot_comparison(sims: List['Simulation'], labels: List[str], show: bool = True):
    """
    # TODO -> FILL DOCUMENTATION
    """
    plt.figure(figsize=(18, 12))
    
    metrics_to_plot = [
        ("alive", "Alive Nodes"),
        ("avg_energy", "Average Energy"),
        ("delivered_cum", "Cumulative Delivered Packets"),
        ("PDR", "Packet Delivery Ratio"),
        ("EC", "Energy Consumption"),
        ("TH", "Throughput"),
    ]
    
    for idx, (metric_key, title) in enumerate(metrics_to_plot, 1):
        plt.subplot(2, 3, idx)
        for sim, label in zip(sims, labels):
            if metric_key in ["PDR", "EC", "TH"]:  
                df = sim.to_detailed_dataframe()
            else:  
                df = sim.to_dataframe()
                
            if df.empty:
                continue
                
            x = df["round"]
            y = df[metric_key]
            
            plt.plot(x, y, 'o-', label=label, markersize=4)
            
        plt.title(title)
        plt.xlabel("Round")
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    if show:
        plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

