import csv
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from math import sqrt, exp
import os
import time


def read_nodes_from_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    nodes = data.get('nodes', [])
    return nodes


def read_edges_from_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    edges = data.get('links', [])
    return edges


def read_positions_from_csv(filename):
    positions = {}
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  
        for row in reader:
            node_id = int(row[0])
            x, y = map(float, row[1:])
            positions[node_id] = (x, y)
    return positions


def calculate_gaussian_resistance(pos, u, v, sigma=0.1, resistance_per_unit=0.1):
    """Calculate the resistance between two nodes based on Gaussian kernel transformation."""
    x1, y1 = pos[u]
    x2, y2 = pos[v]
    distance = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    distance = exp(-distance ** 2 / (2 * sigma ** 2))
    resistance = resistance_per_unit * max(distance, 5.6 * 1e-8)
    return resistance


def weighted_heatOperator(G, t):
    A = nx.to_numpy_array(G, weight='resistance')
    degree_list = [sum(G[u][v]['resistance'] for v in G.neighbors(u)) for u in G.nodes()]
    D = np.diag(degree_list)
    L = D - A
    eig_val, eig_vec = np.linalg.eig(L)
    etem = np.exp(-t * eig_val)
    return eig_vec, etem


def MaxMinNormalizationList(x, Max, Min, scale=1.0):
    return [(val - Min) / (Max - Min) * scale for val in x]


def onlyheat(g, seed, args, weighted=False):
    if weighted:
        eig_vec, etem = weighted_heatOperator(g, args.tOnlyHeat)

    nodelist = list(g.nodes)
    final_activated = [seed]

    heat_graph = np.sum(etem * (eig_vec * eig_vec[nodelist.index(seed), :]), axis=1)

    scale_heat = MaxMinNormalizationList(x=heat_graph.copy(), Max=np.max(heat_graph), Min=np.min(heat_graph), scale=1.0)

    for i in range(len(nodelist)):
        node = nodelist[i]
        if node == seed:
            continue
        if scale_heat[i] > args.threshOnlyHeat:
            final_activated.append(node)

    return final_activated, scale_heat


def build_weighted_graphs(nodes, edges, positions, resistance_per_unit=100, sigma=30):
    G_weighted_gaussian = nx.Graph()

    node_identifier = 'id' if any('id' in node for node in nodes) else 'name'

    for node in nodes:
        identifier = node.get(node_identifier)
        if identifier is None:
            raise ValueError(f"Node does not contain '{node_identifier}' key: {node}")
        for G in [G_weighted_gaussian]:
            G.add_node(identifier, pos=positions.get(identifier), highdegree=node.get('highdegree', 0))

    for edge in edges:
        u, v = edge['source'], edge['target']

        resistance_gaussian = calculate_gaussian_resistance(positions, u, v, sigma=sigma, resistance_per_unit=resistance_per_unit)

        G_weighted_gaussian.add_edge(u, v, resistance=resistance_gaussian)

    return G_weighted_gaussian


def plot_network(G, positions, activated_nodes, seed, output_folder, filename):
    fig, ax = plt.subplots(figsize=(8, 8))


    nx.draw_networkx_edges(G, positions, alpha=0.2, ax=ax)


    pos_vals = np.array(list(positions.values()))
    x_min, y_min = pos_vals.min(axis=0)
    x_max, y_max = pos_vals.max(axis=0)
    padding = (x_max - x_min) * 0.1

    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)


    node_colors = ['#c95644' if node in activated_nodes and node != seed else ('#666666' if node == seed else '#d1d0d4') for node in G.nodes()]


    nx.draw_networkx_nodes(G, positions, node_color=node_colors, node_size=15, ax=ax)


    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    save_path = os.path.join(output_folder, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Network image saved to {save_path}")


class Args:
    def __init__(self, tOnlyHeat, threshOnlyHeat):
        self.tOnlyHeat = tOnlyHeat
        self.threshOnlyHeat = threshOnlyHeat


def update_json_highdegree(original_data, activated_nodes, seed):

    updated_nodes = []
    for node in original_data.get('nodes', []):
        identifier = node.get('id') or node.get('name')
        if identifier in activated_nodes:
            updated_node = node.copy()
            updated_node['highdegree'] = 2
            updated_nodes.append(updated_node)
        else:
            updated_nodes.append(node)

    updated_data = {
        "nodes": updated_nodes,
        "links": original_data.get('links', [])
    }

    return updated_data


if __name__ == "__main__":
    start_time = time.time()

    RESISTANCE_PER_UNIT = 100
    SIGMA = 25
    json_file = '../data/cases/graph_25.02.25/GD6-toy_case8.json'
    csv_file = '../data/cases/graph_25.02.25/GD6-toy_case8.csv'

    nodes = read_nodes_from_json(json_file)
    edges = read_edges_from_json(json_file)
    positions = read_positions_from_csv(csv_file)

    if not all([nodes, edges, positions]):
        print("Warning: One or more data structures are empty. Exiting.")
        exit(1)
    else:
        print("Nodes and edges loaded successfully.")

    G_weighted_gaussian = build_weighted_graphs(
        nodes, edges, positions,
        resistance_per_unit=RESISTANCE_PER_UNIT,
        sigma=SIGMA
    )
    print("Weighted graphs constructed successfully.")

    output_folder = './image/GD6-toy_case8'
    html_output_folder = './data'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(html_output_folder):
        os.makedirs(html_output_folder)

    args = Args(tOnlyHeat=1, threshOnlyHeat=0.8)
    seeds = [node.get('id') or node.get('name') for node in nodes if node.get('highdegree', 0) == 1]

    with open(json_file, 'r') as file:
        original_data = json.load(file)

    for seed in seeds:
        start_time_seed = time.time()
        activated_nodes_weighted_gaussian, _ = onlyheat(G_weighted_gaussian, seed, args, weighted=True)
        end_time_seed = time.time()

        plot_network(G_weighted_gaussian, positions, activated_nodes_weighted_gaussian, seed, output_folder, f'network_seed_{seed}_gaussian.png')

        updated_data = update_json_highdegree(original_data, activated_nodes_weighted_gaussian, seed)

        output_json_file = os.path.join(html_output_folder, f'GD6-toy_case8_{seed}.json')
        with open(output_json_file, 'w') as outfile:
            json.dump(updated_data, outfile, indent=4)

        print(f"Updated JSON saved to {output_json_file}")
        print(f"Seed {seed} processed in {end_time_seed - start_time_seed:.6f} seconds")

    total_end_time = time.time()
    print(f"Total execution time: {total_end_time - start_time:.6f} seconds")


