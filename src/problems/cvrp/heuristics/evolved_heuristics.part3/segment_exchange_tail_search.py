from src.problems.cvrp.components import ReplaceSolutionOperator
import random


def segment_exchange_tail_search(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[ReplaceSolutionOperator, dict]:
    """
    Intensification neighbourhood for tight-capacity X instances.

    Complements relocate/swap/SWAP* by explicitly testing:
    - block exchanges (2-1, 2-2, 3-1)
    - short tail exchanges (a lightweight SwapTails analogue)
    """
    dist = problem_state["distance_matrix"]
    demands = problem_state["demands"]
    capacity = problem_state["capacity"]
    penalty_factor = problem_state.get("capacity_penalty_factor", 200.0)
    current_solution = problem_state["current_solution"]

    routes = [list(route) for route in current_solution.routes]
    active_route_ids = [idx for idx, route in enumerate(routes) if len(route) > 1]
    if len(active_route_ids) < 2:
        return None, algorithm_data

    random.shuffle(active_route_ids)

    def route_cost(route: list[int]) -> float:
        if len(route) <= 1:
            return 0.0
        return sum(dist[route[i]][route[(i + 1) % len(route)]] for i in range(len(route)))

    def route_load(route: list[int]) -> float:
        return sum(demands[node] for node in route)

    def penalised_score(route: list[int], load: float | None = None) -> float:
        current_load = route_load(route) if load is None else load
        return route_cost(route) + max(0.0, current_load - capacity) * penalty_factor

    loads = [route_load(route) for route in routes]
    best_delta = 0.0
    best_routes = None

    for idx_a in range(len(active_route_ids)):
        for idx_b in range(idx_a + 1, len(active_route_ids)):
            route_a_id = active_route_ids[idx_a]
            route_b_id = active_route_ids[idx_b]
            route_a = routes[route_a_id]
            route_b = routes[route_b_id]

            old_score = penalised_score(route_a, loads[route_a_id]) + penalised_score(route_b, loads[route_b_id])

            for len_a, len_b in ((2, 1), (2, 2), (3, 1)):
                if len(route_a) - 1 < len_a or len(route_b) - 1 < len_b:
                    continue

                for pos_a in range(1, len(route_a) - len_a + 1):
                    seg_a = route_a[pos_a:pos_a + len_a]
                    demand_a = sum(demands[node] for node in seg_a)
                    prefix_a = route_a[:pos_a]
                    suffix_a = route_a[pos_a + len_a:]

                    for pos_b in range(1, len(route_b) - len_b + 1):
                        seg_b = route_b[pos_b:pos_b + len_b]
                        demand_b = sum(demands[node] for node in seg_b)
                        prefix_b = route_b[:pos_b]
                        suffix_b = route_b[pos_b + len_b:]

                        new_route_a = prefix_a + seg_b + suffix_a
                        new_route_b = prefix_b + seg_a + suffix_b
                        new_load_a = loads[route_a_id] - demand_a + demand_b
                        new_load_b = loads[route_b_id] - demand_b + demand_a
                        new_score = penalised_score(new_route_a, new_load_a) + penalised_score(new_route_b, new_load_b)
                        delta = new_score - old_score

                        if delta < best_delta - 1e-4:
                            candidate_routes = [list(route) for route in routes]
                            candidate_routes[route_a_id] = new_route_a
                            candidate_routes[route_b_id] = new_route_b
                            best_delta = delta
                            best_routes = candidate_routes

            tested_tail_pairs = 0
            for cut_a in range(1, len(route_a)):
                tail_a = route_a[cut_a:]
                if not tail_a:
                    continue

                for cut_b in range(1, len(route_b)):
                    tail_b = route_b[cut_b:]
                    if not tail_b:
                        continue

                    moved_size = len(tail_a) + len(tail_b)
                    if moved_size < 3 or moved_size > 10:
                        continue

                    anchor_a = route_a[cut_a - 1]
                    anchor_b = route_b[cut_b - 1]
                    old_anchor_cost = dist[anchor_a][tail_a[0]] + dist[anchor_b][tail_b[0]]
                    new_anchor_cost = dist[anchor_a][tail_b[0]] + dist[anchor_b][tail_a[0]]
                    if new_anchor_cost > old_anchor_cost + 60.0:
                        continue

                    new_route_a = route_a[:cut_a] + tail_b
                    new_route_b = route_b[:cut_b] + tail_a
                    new_load_a = route_load(new_route_a)
                    new_load_b = route_load(new_route_b)
                    new_score = penalised_score(new_route_a, new_load_a) + penalised_score(new_route_b, new_load_b)
                    delta = new_score - old_score
                    tested_tail_pairs += 1

                    if delta < best_delta - 1e-4:
                        candidate_routes = [list(route) for route in routes]
                        candidate_routes[route_a_id] = new_route_a
                        candidate_routes[route_b_id] = new_route_b
                        best_delta = delta
                        best_routes = candidate_routes

                    if tested_tail_pairs >= 12:
                        break

                if tested_tail_pairs >= 12:
                    break

    if best_routes is None:
        return None, algorithm_data

    return ReplaceSolutionOperator(routes=best_routes), algorithm_data