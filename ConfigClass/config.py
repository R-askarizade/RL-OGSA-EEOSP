from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import Rectangle, Polygon, Circle
from matplotlib.collections import PatchCollection
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Point, Polygon as ShapelyPolygon

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import os
import numpy as np
from typing import List


import warnings
warnings.filterwarnings('ignore')


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
        ("TH_pps", "Throughput (packets/sec)"),
        ("EE_Js", "Energy Efficiency (packets/(J·s))"),
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


def visualize_deployment(sim: "Simulation", save_path: str = r"D:\Papers\4) Finished Articles\6. MWSN - DCHPC\DCHPC", show: bool = False):
    # Academic font settings
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'] + plt.rcParams['font.serif'],
        'font.size': 10
    })

    # Extract simulation data
    area_w, area_h = sim.area_size
    nodes = sim.nodes
    n_nodes = len(nodes)
    clusters = sim.cluster_manager.get_clusters()
    cluster_heads = sim.cluster_manager.cluster_heads
    sink_pos = sim.sink.get_position()
    sink_trajectory = np.array(getattr(sim, 'sink_trajectory', [sink_pos]))

    # Edge node data
    edge_ids = getattr(sim, 'edge_node_ids', [])
    pre_edge = getattr(sim, 'previous_edge_pos', [])
    post_edge = getattr(sim, 'changed_edge_pos', [])

    # Align edge movements by node ID
    pre_dict = dict(pre_edge) if pre_edge else {}
    post_dict = dict(post_edge) if post_edge else {}
    common_edge_ids = [
        nid for nid in edge_ids if nid in pre_dict and nid in post_dict]

    init_positions = np.array([[n.x, n.y] for n in nodes])
    final_positions = init_positions  # same as after optimization

    # Color map for clusters
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(cluster_heads))))

    # Helper: plot edge movements
    def plot_edge_movements(ax, only_arrows=False):
        for i, nid in enumerate(common_edge_ids):
            start = np.array(pre_dict[nid], dtype=float)
            end = np.array(post_dict[nid], dtype=float)

            if not only_arrows:
                ax.scatter(start[0], start[1], c='deeppink', s=80, linewidth=0.6, zorder=15,
                           label='Initial Edge Node' if i == 0 else None)
                ax.scatter(end[0], end[1], c='red', s=80, marker='^',
                           edgecolor='darkred', linewidth=0.6, zorder=20,
                           label='Tuned Edge Node' if i == 0 else None)
            if i == 0:
                # Draw once with label using a proxy artist
                from matplotlib.lines import Line2D
                ax.plot([], [], color='purple', lw=2.5,
                        label='Edge Node Movement')
            arrow = FancyArrowPatch(
                start, end,
                arrowstyle='-|>,head_width=2,head_length=4',
                connectionstyle='arc3,rad=0.2',
                color='purple',
                lw=1.5,
                alpha=0.9,
                zorder=5
            )
            ax.add_patch(arrow)

    #  Plot 1: Random Deployment
    def plot_initial(ax):
        ax.set_title('Random Deployment', fontsize=10, fontweight='bold')
        np.random.seed(42)
        rand_pos = np.column_stack((np.random.rand(n_nodes) * area_w,
                                    np.random.rand(n_nodes) * area_h))
        ax.scatter(rand_pos[:, 0], rand_pos[:, 1],
                   c='blue', s=30, alpha=0.6, label='Nodes')
        ax.add_patch(Rectangle((0, 0), area_w, area_h, linewidth=1,
                     edgecolor='black', facecolor='none'))
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend(loc='best', bbox_to_anchor=(1, 1))
        ax.grid(True, alpha=0.3)

    #  Plot 2: Voronoi + Edge Tuning
    def plot_voronoi(ax):
        ax.set_title('Voronoi-based Repulsion & Edge Tuning',
                     fontsize=10, fontweight='bold')
        vor = Voronoi(final_positions)
        voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='gray',
                        line_width=0.5, line_alpha=0.5, point_size=0)

        non_edge_positions = np.array([[n.x, n.y]
                                      for n in nodes if n.id not in edge_ids])
        if len(non_edge_positions) > 0:
            ax.scatter(non_edge_positions[:, 0], non_edge_positions[:, 1],
                       c='green', s=40, alpha=0.7, zorder=10, label='Nodes')

        if common_edge_ids:
            plot_edge_movements(ax)

        ax.add_patch(Rectangle((0, 0), area_w, area_h, linewidth=1,
                               edgecolor='black', facecolor='none'))
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend(loc='best', bbox_to_anchor=(1, 1))
        ax.grid(True, alpha=0.3)

    #  Plot 3: Clusters
    def plot_clusters(ax):
        ax.set_title('Cluster Formation with CHs',
                     fontsize=10, fontweight='bold')
        for idx, (ch_id, members) in enumerate(clusters.items()):
            ch_node = next((n for n in nodes if n.id == ch_id), None)
            if ch_node:
                color = colors[idx % len(colors)]
                member_pos = np.array([[m.x, m.y]
                                      for m in members if m.id != ch_id])
                if len(member_pos) > 0:
                    ax.scatter(member_pos[:, 0], member_pos[:, 1], c=[color], s=30, alpha=0.5,
                               label='Cluster Members' if idx == 0 else "")
                    for pos in member_pos:
                        ax.plot([pos[0], ch_node.x], [pos[1], ch_node.y],
                                color=color, alpha=0.2, linewidth=0.5)
                ax.scatter(ch_node.x, ch_node.y, c=[color], s=150, marker='*',
                           edgecolor='black', linewidth=1.5, zorder=10,
                           label='Cluster Head' if idx == 0 else "")
        ax.scatter(sink_pos[0], sink_pos[1], c='purple',
                   s=200, marker='p', label='Mobile Sink', zorder=15)
        if len(sink_trajectory) > 1:
            ax.plot(sink_trajectory[:, 0], sink_trajectory[:, 1],
                    'purple', alpha=0.3, linestyle='-->', linewidth=1)
        ax.add_patch(Rectangle((0, 0), area_w, area_h, linewidth=1,
                               edgecolor='black', facecolor='none'))
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend(loc='best', bbox_to_anchor=(1, 1))
        ax.grid(True, alpha=0.3)

    #  Plot 4: Voronoi Coverage
    def plot_coverage_voronoi(ax):
        ax.set_title('Coverage Area with Voronoi Cells',
                     fontsize=10, fontweight='bold')
        vor = Voronoi(final_positions)
        legend_added = {'member': False, 'head': False}
        node_id_to_idx = {n.id: i for i, n in enumerate(nodes)}
        for idx, (ch_id, members) in enumerate(clusters.items()):
            color = colors[idx % len(colors)]
            for member in members:
                if member.id in node_id_to_idx:
                    ridx = vor.point_region[node_id_to_idx[member.id]]
                    region = vor.regions[ridx]
                    if region and -1 not in region:
                        poly = Polygon(vor.vertices[region], facecolor=color, alpha=0.3,
                                       edgecolor='gray', linewidth=0.5)
                        ax.add_patch(poly)
            member_pos = np.array([[m.x, m.y] for m in members])
            ax.scatter(member_pos[:, 0], member_pos[:, 1],
                       c=[color], s=20, alpha=0.8)
            ch_node = next((n for n in nodes if n.id == ch_id), None)
            if ch_node:
                ax.scatter(ch_node.x, ch_node.y, c=[color], s=150, marker='*',
                           edgecolor='black', linewidth=1.5, zorder=15)
                if not legend_added['member']:
                    legend_added.update({'member': True, 'head': True})
        ax.add_patch(Rectangle((0, 0), area_w, area_h, linewidth=1,
                               edgecolor='black', facecolor='none'))
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        # Custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=colors[0], markersize=5, label='Member Nodes'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor=colors[0],
                   markersize=10, markeredgecolor='black', label='Cluster Head')
        ]
        ax.legend(handles=legend_elements, loc='best', bbox_to_anchor=(1, 1))
        ax.grid(True, alpha=0.3)

    #  Plot 5: Effective Coverage (Comm Range)
    def plot_coverage_effective(ax):
        ax.set_title('Effective Coverage (Communication Range)',
                     fontsize=10, fontweight='bold')
        grid_size = 50
        x = np.linspace(0, area_w, grid_size)
        y = np.linspace(0, area_h, grid_size)
        coverage = np.zeros((grid_size, grid_size))
        alive = [n for n in nodes if n.is_alive()]
        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                if any(np.hypot(n.x - xi, n.y - yj) <= sim.comm_range for n in alive):
                    coverage[j, i] = 1
        im = ax.imshow(coverage, extent=[0, area_w, 0, area_h], origin='lower',
                       cmap='viridis', alpha=0.7, vmin=0, vmax=1)
        ax.scatter(final_positions[:, 0], final_positions[:, 1],
                   c='blue', s=20, alpha=0.8, zorder=5, label='Sensor Node')
        ch_pos = np.array([[ch.x, ch.y]
                          for ch in cluster_heads if ch.is_alive()])
        if len(ch_pos) > 0:
            ax.scatter(ch_pos[:, 0], ch_pos[:, 1], c='red', s=100,
                       marker='*', zorder=10, label='Cluster Head')
        ax.scatter(sink_pos[0], sink_pos[1], c='purple',
                   s=150, marker='p', label='Mobile Sink', zorder=15)
        if len(sink_trajectory) > 1:
            ax.plot(sink_trajectory[:, 0], sink_trajectory[:, 1],
                    'purple', alpha=0.3, linestyle='-->', linewidth=1)
        ax.add_patch(Rectangle((0, 0), area_w, area_h, linewidth=1,
                               edgecolor='black', facecolor='none'))
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        legend_elements = [
            plt.Line2D([], [], marker='o', color='w',
                       markerfacecolor='blue', markersize=5, label='Sensor Node'),
            plt.Line2D([], [], marker='*', color='w',
                       markerfacecolor='red', markersize=10, label='Cluster Head'),
            plt.Line2D([], [], marker='p', color='w',
                       markerfacecolor='purple', markersize=10, label='Mobile Sink'),
            ]
        ax.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1.0, -0.1))
        ax.grid(True, alpha=0.3)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Coverage')

    #  Generate Plots
    plot_funcs = [
        (plot_initial, "plot_initial"),
        (plot_voronoi, "plot_voronoi"),
        (plot_clusters, "plot_clusters"),
        (plot_coverage_voronoi, "plot_coverage_voronoi"),
        (plot_coverage_effective, "plot_coverage_effective"),
    ]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    base, ext = os.path.splitext(save_path)

    for func, name in plot_funcs:
        fig, ax = plt.subplots(figsize=(12, 9))
        func(ax)
        ax.set_xlim(-5, area_w + 5)
        ax.set_ylim(-5, area_h + 5)
        plt.tight_layout(rect=[0, 0, 1, 1])
        full_path = f"{base}_{name}{ext}"
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {full_path}")
        if show:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
