import copy
import functools
from typing import List

import jax
import networkx
import networkx as nx
import numpy as np


def is_inside_aabb(obj_pos, obj_size, agent_pos):
    """
    Check if an agent at `agent_pos` is inside the AABB defined by `obj_pos` and `obj_size`.
    Returns True if inside, False if outside.
    """
    # Only check x (0) and z (2) dimensions, skip y (1)
    x_inside = jnp.logical_and(
        obj_pos[0] - obj_size[0] <= agent_pos[0],
        agent_pos[0] <= obj_pos[0] + obj_size[0]
    )
    z_inside = jnp.logical_and(
        obj_pos[2] - obj_size[2] <= agent_pos[2],
        agent_pos[2] <= obj_pos[2] + obj_size[2]
    )
    return jnp.logical_and(x_inside, z_inside)

@jax.jit
def _filter_agent_positions(agent_positions, object_positions, object_sizes, margin=0.0):
    """
    Remove agent positions that are inside any object's AABB, considering a margin.
    """
    inflated_sizes = object_sizes + margin

    def is_inside_any_object(single_agent_pos, all_object_poses, all_object_sizes):
        return jax.vmap(functools.partial(is_inside_aabb, agent_pos=single_agent_pos))( all_object_poses, all_object_sizes).any()

    mask = jax.vmap(functools.partial(is_inside_any_object, all_object_poses=object_positions, all_object_sizes=inflated_sizes))(agent_positions)
    return jnp.logical_not(mask)


def build_grid_graph(points, GRID_SIZE, diagonal=False, clean_convert=lambda x: x):
    #points = [(float(p[0]), float(p[1]), float(p[2])) for p in points]
    points = [clean_convert(p) for p in points]

    G = nx.Graph()
    points_set = set(points)
    dim = len(points[0])  # infer dimensionality

    # Add all points as nodes
    for p in points:
        G.add_node(p)

    # Define neighbor directions (axis-aligned)
    directions = []
    for d in range(dim):
        if d == 1:
            continue
        for offset in [-GRID_SIZE, GRID_SIZE]:
            dir_vector = [0] * dim
            dir_vector[d] = offset
            directions.append(tuple(dir_vector))

    if diagonal:
        # Optional: include diagonals
        from itertools import product
        directions = list(product([-1, 0, 1], repeat=dim))
        directions.remove((0,) * dim)  # remove self

    # Add edges between neighbors
    for p in points:
        for d in directions:
            neighbor = tuple(np.add(p, d))
            if neighbor in points_set:
                G.add_edge(p, neighbor)

    return G

import matplotlib.pyplot as plt

def draw_grid_graph_2d(G, path=None, node_size=100, node_color='lightblue', edge_color='gray', show_labels=False):
    pos = {node: (node[2], node[0]) for node in G.nodes}  # node positions are their coordinates

    plt.figure(figsize=(6, 6))

    # Draw the base graph
    nx.draw(G, pos,
            node_size=node_size,
            node_color=node_color,
            edge_color=edge_color,
            with_labels=show_labels,
            font_size=8)

    # If a path is provided, draw it in red
    if path:
        # Draw path edges first (so nodes will be on top)
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(G, pos,
                               edgelist=path_edges,
                               edge_color='red',
                               width=2)

        # Draw path nodes
        nx.draw_networkx_nodes(G, pos,
                               nodelist=path,
                               node_size=node_size,
                               node_color='red')

    plt.gca().set_aspect('equal')
    plt.gca().invert_yaxis()  # Optional: invert y for "matrix-style" layout
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(sorted(set(x for x, y, z in G.nodes())))
    plt.yticks(sorted(set(z for x, y, z in G.nodes())))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('2D Grid Graph')
    plt.show()
    i=0

from jax import numpy as jnp

import copy
import networkx as nx
import numpy as np
import jax.numpy as jnp

def filter_reachable_positions(reachable_positions, runtime_container):
    if isinstance(reachable_positions, list):
        todo_positions = reachable_positions
    elif isinstance(reachable_positions, nx.Graph):
        todo_positions = reachable_positions.nodes
        original_reachable_positions = reachable_positions
        reachable_positions = copy.deepcopy(reachable_positions)

    centers = [obj.position for obj in runtime_container.objects_map.values()]
    sizes = [obj.size for obj in runtime_container.objects_map.values()]

    todo_positions = jnp.array(todo_positions)
    centers = jnp.array(centers)
    sizes = jnp.array(sizes)

    mask = _filter_agent_positions(todo_positions.at[:,1].set(0.95),  centers.at[:,1].set(0.95), sizes, margin=0)

    print("Num positions to remove", jnp.sum(jnp.logical_not(mask)))

    mask = np.array(mask)
    todo_positions = np.array(todo_positions)
    filtered_reachable_positions = todo_positions[mask]

    if isinstance(reachable_positions, nx.Graph):

        for position in todo_positions[jnp.logical_not(mask)]:
            #position = (float(position[0]), float(position[1]), float(position[2]))
            #position = round_position(position)
            for node in nx.nodes(reachable_positions):
                if np.all(np.allclose(node, position)):
                    reachable_positions.remove_node(node)
                    break

            #x= 0


        # Step 1: Remove unreachable nodes
       # def round_position(pos):
       #     base = min(min(pos) for pos in todo_positions)
       #     return tuple(round(coord / base) * base for coord in pos)
       # filtered_reachable_positions = list(map(round_position, [tuple(x.tolist()) for x in filtered_reachable_positions]))
       # todo_positions = list(map(round_position,[tuple(x.tolist()) for x in todo_positions]))
       # to_remove = list(set(todo_positions) - set(filtered_reachable_positions))
       # reachable_positions.remove_nodes_from(to_remove)

        # Step 2: Keep only the largest connected component
        if reachable_positions.number_of_nodes() > 0:
            largest_cc_nodes = max(nx.connected_components(reachable_positions), key=len)
            reachable_positions = reachable_positions.subgraph(largest_cc_nodes).copy()
        #draw_grid_graph_2d(reachable_positions, show_labels=False)

        return reachable_positions

