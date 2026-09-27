"""Provider-neutral graph presentations; never consume evaluator labels."""

from __future__ import annotations

import networkx as nx

from .config import PresentationConfig


def state_for(graph: nx.Graph, presentation: PresentationConfig) -> dict:
    nodes = list(graph.nodes())
    if presentation.graph_representation == "adjacency_list_v1":
        return {"nodes": nodes, "adjacency": [sorted(graph.neighbors(n)) for n in nodes]}
    return {"nodes": nodes, "edges": [list(edge) for edge in graph.edges()]}


def question_for(u: int, v: int, threshold: int | None, presentation: PresentationConfig) -> str:
    question = "Use only the given undirected, unweighted graph. "
    question += (
        f"Are nodes {u} and {v} directly connected by an edge?"
        if threshold is None else
        f"Is there a path from node {u} to node {v} using at most {threshold} edges?"
    )
    if presentation.graph_representation == "adjacency_list_v1":
        question += " In the state, adjacency[i] lists all neighbors of node i."
    if presentation.prompt_variant == "algorithmic_v1":
        if presentation.graph_representation == "nodes_edges":
            question += " Each pair [a, b] in edges represents an edge usable in both directions."
        question += " The provided topology is complete; do not assume any additional edges."
        if threshold is None:
            question += (
                " Check whether the queried pair itself is an edge, in either direction."
                " An indirect path does not count. Select yes exactly when that edge exists; otherwise select no."
            )
        else:
            question += (
                f" Determine the shortest-path distance from node {u} to node {v}."
                f" One method is breadth-first search: start at node {u}, expand neighbors layer by layer,"
                " and do not revisit nodes already reached. Each layer adds one edge."
                f" Select yes exactly when node {v} is reached within {threshold} layers; otherwise select no."
            )
    return question


def graph_from_state(state: object, node_count: int) -> nx.Graph:
    """Strictly reconstruct both supported states for independent truth checks."""
    if not isinstance(state, dict) or set(state) not in ({"nodes", "edges"}, {"nodes", "adjacency"}):
        raise ValueError("Unexpected model-visible fields")
    nodes = state["nodes"]
    if (not isinstance(nodes, list) or nodes != list(range(node_count))
            or any(type(n) is not int for n in nodes)):
        raise ValueError("Nodes must be anonymous contiguous integer IDs")
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    if "edges" in state:
        edges = state["edges"]
        if not isinstance(edges, list) or any(
            not isinstance(e, list) or len(e) != 2
            or any(type(n) is not int or n not in graph for n in e) or e[0] == e[1]
            for e in edges
        ):
            raise ValueError("Invalid topology payload")
        graph.add_edges_from(edges)
        if graph.number_of_edges() != len(edges):
            raise ValueError("Duplicate edge in topology payload")
    else:
        adjacency = state["adjacency"]
        if (not isinstance(adjacency, list) or len(adjacency) != node_count
                or any(not isinstance(row, list) or any(type(v) is not int or v not in graph for v in row)
                       or len(set(row)) != len(row) or u in row for u, row in enumerate(adjacency))):
            raise ValueError("Invalid adjacency payload")
        for u, row in enumerate(adjacency):
            for v in row:
                if u not in adjacency[v]:
                    raise ValueError("Undirected adjacency must be symmetric")
                graph.add_edge(u, v)
    return graph