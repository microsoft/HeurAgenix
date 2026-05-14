from src.problems.cvrp.components import ReplaceSolutionOperator
import time
import numpy as np

try:
    from pyvrp import solve, stop, Model
except Exception:
    pass

def pyvrp_blackbox(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[ReplaceSolutionOperator, dict]:
    """
    Standard heuristic interface for PyVRP warm start/reseed.
    Constructs the problem completely from memory (no file IO).
    """
    runtime = kwargs.get("runtime", 10)
    attempts = kwargs.get("attempts", 1)
    
    seed = kwargs.get("seed")
    if seed is None:
        seed = int((time.time() * 1000) % 10_000_000)
    else:
        seed = int(seed)
        
    worker_id = kwargs.get("worker_id", "0")
    
    try:
        dist_matrix = problem_state.get("distance_matrix")
        demands = problem_state.get("demands")
        capacity = problem_state.get("capacity")
        vehicle_num = problem_state.get("vehicle_num", 999)
        depot_idx = problem_state.get("depot", 0)

        if dist_matrix is None or demands is None or capacity is None:
            return ReplaceSolutionOperator(routes=[]), algorithm_data

        if vehicle_num is None or vehicle_num <= 0:
            vehicle_num = 999

        # Build PyVRP Model from memory
        m = Model()
        m.add_vehicle_type(num_available=vehicle_num, capacity=capacity)

        pyvrp_nodes = []
        pyvrp_to_original = {}
        pyvrp_client_idx = 1
        
        # Add Depot first (PyVRP requires exactly one depot, implicitly index 0)
        depot_node = m.add_depot(x=0, y=0)
        
        # Add nodes
        for i in range(len(demands)):
            if i == depot_idx:
                pyvrp_nodes.append(depot_node)
            else:
                client_node = m.add_client(x=0, y=0, delivery=int(demands[i]))
                pyvrp_nodes.append(client_node)
                pyvrp_to_original[pyvrp_client_idx] = i
                pyvrp_client_idx += 1

        # Add edges using the exact distance matrix
        for i in range(len(dist_matrix)):
            for j in range(len(dist_matrix)):
                if i != j:
                    m.add_edge(pyvrp_nodes[i], pyvrp_nodes[j], distance=int(dist_matrix[i][j]))
        
        data = m.data()

        best_routes = []
        best_cost = float("inf")
        
        for k in range(attempts):
            seed_k = int((seed + k * 9973 + int(worker_id) * 131) % 10_000_000)
            result = solve(
                data,
                stop=stop.MaxRuntime(runtime),
                seed=seed_k,
                collect_stats=False,
                display=False,
            )

            routes_k = []
            for r in result.best.routes():
                visits = list(r)
                if visits:
                    # Map PyVRP client indices back to our original node indices
                    mapped_visits = [depot_idx] + [pyvrp_to_original[v] for v in visits]
                    routes_k.append(mapped_visits)

            if vehicle_num > 0 and vehicle_num != 999:
                if len(routes_k) < vehicle_num:
                    routes_k.extend([[depot_idx] for _ in range(vehicle_num - len(routes_k))])
                elif len(routes_k) > vehicle_num:
                    routes_k = routes_k[:vehicle_num]

            if not routes_k:
                continue

            cand_cost = float(result.cost())
            if cand_cost < best_cost:
                best_cost = cand_cost
                best_routes = routes_k
                
        if not best_routes:
            return ReplaceSolutionOperator(routes=[]), algorithm_data

        return ReplaceSolutionOperator(routes=best_routes), algorithm_data
    except Exception as e:
        import traceback
        traceback.print_exc()
        return ReplaceSolutionOperator(routes=[]), algorithm_data
