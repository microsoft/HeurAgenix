from src.problems.base.components import BaseSolution, BaseOperator

class Solution(BaseSolution):
    """The solution of CVRP.
    A list of lists where each sublist represents a vehicle's route.
    Each sublist contains integers representing the nodes (customers) visited by the vehicle in the order of visitation.
    The routes are sorted by vehicle identifier and the nodes in the list sorted by visited order.
    """
    def __init__(self, routes: list[list[int]], depot: int, total_cost: float = None, loads: list[float] = None):
        self.routes = routes
        self.depot = depot
        self.total_cost = total_cost
        self.loads = loads

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
