from src.problems.base.components import BaseSolution, BaseOperator

class Solution(BaseSolution):
    """The solution of CVRP.
    A list of lists where each sublist represents a vehicle's route.
    Each sublist contains integers representing the nodes (customers) visited by the vehicle in the order of visitation.
    The routes are sorted by vehicle identifier and the nodes in the list sorted by visited order.
    """
    def __init__(self, routes: list[list[int]], depot: int, total_cost: float = None, loads: list[float] = None, fitness: float = None, capacity_violation: float = 0.0, unassigned_nodes: list[int] = None):
        self.routes = routes
        self.depot = depot
        self.total_cost = total_cost
        self.loads = loads
        self.fitness = fitness if fitness is not None else total_cost
        self.capacity_violation = capacity_violation
        self.unassigned_nodes = unassigned_nodes if unassigned_nodes is not None else []

    def __str__(self) -> str:
        route_string = ""
        for index, route in enumerate(self.routes):
            depot_index = route.index(self.depot)
            rotated_route = route[depot_index:] + route[:depot_index] + [self.depot]
            route = [self.depot] + route + [self.depot]
            route_string += f"vehicle_{index}: " + "->".join(map(str, rotated_route)) + "\n"
        return route_string


class AppendOperator(BaseOperator):
    """Append a node at the end of the specified vehicle's route."""
    def __init__(self, vehicle_id: int, node: int):
        self.vehicle_id = vehicle_id
        self.node = node


class InsertOperator(BaseOperator):
    """Insert a node at a specified position within the route of a specified vehicle."""
    def __init__(self, vehicle_id: int, node: int, position: int):
        self.vehicle_id = vehicle_id
        self.node = node
        self.position = position


class SwapOperator(BaseOperator):
    """Swap two nodes between or within vehicle routes."""
    def __init__(self, vehicle_id1: int, position1: int, vehicle_id2: int, position2: int):
        # Always store sorted to simplify delta calculation logic later
        if vehicle_id1 < vehicle_id2 or (vehicle_id1 == vehicle_id2 and position1 <= position2):
            self.vehicle_id1 = vehicle_id1
            self.position1 = position1
            self.vehicle_id2 = vehicle_id2
            self.position2 = position2
        else:
            self.vehicle_id1 = vehicle_id2
            self.position1 = position2
            self.vehicle_id2 = vehicle_id1
            self.position2 = position1


class ReverseSegmentOperator(BaseOperator):
    """Reverse multiple segments of indices in the solution (2-opt, 3-opt)."""
    def __init__(self, vehicle_id: int, segments: list[tuple[int, int]]):
        # Simplified to standard 2-opt segment reversal within one route
        self.vehicle_id = vehicle_id
        self.segments = segments


class RelocateOperator(BaseOperator):
    """Move a node from one position in a route to another, possibly in a different route."""
    def __init__(self, source_vehicle_id: int, source_position: int, target_vehicle_id: int, target_position: int):
        self.source_vehicle_id = source_vehicle_id
        self.source_position = source_position
        self.target_vehicle_id = target_vehicle_id
        self.target_position = target_position


class BatchRemoveOperator(BaseOperator):
    """Remove a set of nodes from the solution (Ruin phase)."""
    def __init__(self, nodes: list[int]):
        self.nodes = nodes


class BatchInsertOperator(BaseOperator):
    """Insert multiple nodes into specific positions (Recreate phase)."""
    def __init__(self, insertions: list[tuple[int, int, int]]):
        # List of (vehicle_id, position, node)
        # Note: Position handling must be careful if inserting into same route.
        # Ideally, insertions should be sorted or handled such that position indices remain valid.
        self.insertions = insertions


class MergeRoutesOperator(BaseOperator):
    """Merge two routes by appending the route of the source vehicle to the beginning of the route of the target vehicle. 
    The merged route is assigned to the target vehicle, and the source vehicle's route is cleared."""
    def __init__(self, source_vehicle_id: int, target_vehicle_id: int):
        self.source_vehicle_id = source_vehicle_id
        self.target_vehicle_id = target_vehicle_id

class ReplaceSolutionOperator(BaseOperator):
    """Replace the current solution with a completely new solution (useful for crossover/relinking)."""
    def __init__(self, routes: list[list[int]]):
        self.routes = routes

class ReplaceSolutionOperator(BaseOperator):
    """Replace the entire solution routes with new routes."""
    def __init__(self, routes: list[list[int]]):
        self.routes = routes

class SwapStarOperator(BaseOperator):
    """SWAP* Operator: Swap two nodes between two different routes, but insert them into their BEST positions in the target routes rather than their original positions."""
    def __init__(self, vehicle_id1: int, node1: int, best_pos_for_1_in_2: int, 
                 vehicle_id2: int, node2: int, best_pos_for_2_in_1: int):
        self.vehicle_id1 = vehicle_id1
        self.node1 = node1
        self.best_pos_for_1_in_2 = best_pos_for_1_in_2
        self.vehicle_id2 = vehicle_id2
        self.node2 = node2
        self.best_pos_for_2_in_1 = best_pos_for_2_in_1

class BlockRelocateOperator(BaseOperator):
    """SREX/Macro-Ruin base: Relocate a contiguous block of nodes from a source route to a target position in another route."""
    def __init__(self, source_vehicle_id: int, start_idx: int, end_idx: int, 
                 target_vehicle_id: int, target_position: int):
        self.source_vehicle_id = source_vehicle_id
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.target_vehicle_id = target_vehicle_id
        self.target_position = target_position

class EjectNodeOperator(BaseOperator):
    """Remove a node from a route and place it into the unassigned nodes pool (Dynamic Penalty search)."""
    def __init__(self, vehicle_id: int, position: int):
        self.vehicle_id = vehicle_id
        self.position = position

class InjectNodeOperator(BaseOperator):
    """Take a node from the unassigned pool and insert it into a route."""
    def __init__(self, node: int, vehicle_id: int, position: int):
        self.node = node
        self.vehicle_id = vehicle_id
        self.position = position



class RemoveNodesOperator(BaseOperator):
    def __init__(self, nodes: list[int]):
        super().__init__()
        self.nodes = nodes

    def _get_description(self) -> str:
        return f"RemoveNodesOperator(nodes={self.nodes})"
