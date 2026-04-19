# RL-OGSA-EEOSP: A Hybrid Framework for Energy-Efficient Clustering and Routing in MWSNs

This repository contains the official source code and implementation for the paper **"RL-OGSA-EEOSP: A Hybrid Framework for Energy-Efficient Clustering and Routing in MWSNs"**.

## 📖 Abstract

A comprehensive and intelligent framework (RL-OGSA-EEOSP) is presented with the primary goal of reducing energy consumption and extending network lifetime in Mobile Wireless Sensor Networks (MWSNs). 

The framework integrates:
* **Node Positioning**: Determining the positions of static network nodes by integrating the Voronoi method and a Reinforcement Learning (RL) algorithm, and examining the impact of node placement on network longevity.
* **Clustering Strategy**: Executing clustering and cluster head selection utilizing the Oppositional Gravitational Search Algorithm (OGSA) and implementing optimal re-clustering based on a multi-objective fitness function.
* **MS Placement**: Implementing an Energy-Efficient Optimal Sink Placement (EEOSP)-based placement strategy for the MS to reduce the energy consumption of CHs and minimize transmission distances to them.
* **Routing Protocol**: Hybrid one-hop and multi-hop routing utilizing trust criteria, node energy, current load, and distance to identify optimal nodes for data transfer.

To further enhance the findings, a multi-criteria re-clustering policy and trust-based sensor node routing were implemented. Experimental results demonstrate that the proposed methodology is highly effective, outperforming existing methods by achieving considerable gains in energy efficiency, coverage area, and load balancing, ultimately increasing the overall network lifetime.


---

## 🚀 Getting Started

### Prerequisites
To run this project, you will need Python installed along with Jupyter Notebook. 
* Python 3.x
* Jupyter Notebook (`pip install notebook`)

### How to Run

The simulation and model results are executed through Jupyter Notebooks. 

1. Clone the repository to your local machine.
2. The core framework and models are implemented within the various `.py` files.
3. To view and generate the results, open the provided `.ipynb` (Jupyter Notebook) files. These notebooks call the underlying Python modules.
4. Run the cells sequentially within the notebooks to execute the simulations and view the outputs.
```bash

## 📝 Citation

If you use this code or framework in your research, please cite our paper:

bibtex
@article{askarizade2026rlogsaeeosp,
  title={RL-OGSA-EEOSP: A Hybrid Framework for Energy-Efficient Clustering and Routing in MWSNs},
  author={Askarizade, Razieh and Kuchaki Rafsanjani, Marjan and Fanian, Fakhrosadat},
  journal={International Journal on Semantic Web and Information Systems},
  year={2026}
}
