# GVRRI Algorithm: Visual Receptive Region Identification for Node–Link Diagrams

## Algorithm Overview
This repository contains the implementation of **GVRRI** (Graph Visual Receptive Region Identification), an algorithm that automatically identifies VRRs(Visual Receptive Regions) in node-link diagrams based on a heat diffusion model. GVRRI initializes unit heat at a target node and diffuses it along edges from high- heat to low-heat nodes. Both graph structural information in data space and visual-spatial relationships in visual space jointly govern this process, generating node heat values that are consistent with human VRR cognition after unit-time diffusion. This is useful for perception‑aware graph analysis, interactive exploration, and layout enhancement.


## Repository Structure

├── resistance.py # Main implementation of the GVRRI algorithm
├── data/ # Example dataset (see format below)
│ ├── graph_structure.json # Topology: nodes and edges
│ └── node_layout.csv # 2D coordinates of each node after layout
└── README.md


## Input Data

Running `resistance.py` requires two data files:

- **JSON file** – graph topology (nodes and edges), with optional `"highdegree": 1` flag to mark recommended target nodes.
- **CSV file** – node layout coordinates (x, y)

Both files must be provided as input. Example files are included in the `data/` folder.
