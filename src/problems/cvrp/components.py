from src.problems.base.components import BaseSolution, BaseOperator

class Solution(BaseSolution):
    """The solution of CVRP.
    A list of lists where each sublist represents a vehicle's route.
    Each sublist contains integers representing the nodes (customers) visited by the vehicle in the order of visitation.
    The routes are sorted by vehicle identifier and the nodes in the list sorted by visited order.
    """
    def __init__(self, routes: list[list[int]], depot: int):
        self.routes = routes
        self.depot = depot

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

    def run(self, solution: Solution) -> Solution:
        new_routes = [route[:] for route in solution.routes]
        new_routes[self.vehicle_id].append(self.node)
        return Solution(new_routes, solution.depot)


class InsertOperator(BaseOperator):
    """Insert a node at a specified position within the route of a specified vehicle."""
    def __init__(self, vehicle_id: int, node: int, position: int):
        self.vehicle_id = vehicle_id
        self.node = node
        self.position = position

    def run(self, solution: Solution) -> Solution:
        new_routes = [route[:] for route in solution.routes]
        new_routes[self.vehicle_id].insert(self.position, self.node)
        return Solution(new_routes, solution.depot)


class SwapOperator(BaseOperator):
    """Swap two nodes between or within vehicle routes."""
    def __init__(self, vehicle_id1: int, position1: int, vehicle_id2: int, position2: int):
        if vehicle_id1 <= vehicle_id2:
            self.vehicle_id1 = vehicle_id1
            self.position1 = position1
            self.vehicle_id2 = vehicle_id2
            self.position2 = position2
        else:
            self.vehicle_id1 = vehicle_id2
            self.position1 = position2
            self.vehicle_id2 = vehicle_id1
            self.position2 = position1

    def run(self, solution: Solution) -> Solution:
        new_routes = [route[:] for route in solution.routes]
        new_routes[self.vehicle_id1][self.position1], new_routes[self.vehicle_id2][self.position2] = new_routes[self.vehicle_id2][self.position2], new_routes[self.vehicle_id1][self.position1]
        return Solution(new_routes, solution.depot)


class ReverseSegmentOperator(BaseOperator):
    """Reverse multiple segments of indices in the solution."""
    def __init__(self, vehicle_id: int, segments: list[tuple[int, int]]):
        self.vehicle_id = vehicle_id
        self.segments = segments

    def run(self, solution: Solution) -> Solution:
        new_routes = [route[:] for route in solution.routes]
        new_route = new_routes[self.vehicle_id][:]

        for segment in self.segments:
            start_index, end_index = segment

            # Ensure the indices are within the bounds of the tour list
            assert 0 <= start_index < len(new_routes[self.vehicle_id])
            assert 0 <= end_index < len(new_routes[self.vehicle_id])

            if start_index <= end_index:
                # Reverse the segment between start_index and end_index (inclusive)
                new_route[start_index:end_index + 1] = reversed(new_route[start_index:end_index + 1])
            else:
                # Reverse the segment outside start_index and end_index (inclusive)
                new_route = list(reversed(new_route[start_index:])) + new_route[end_index + 1:start_index] + list(reversed(new_route[:end_index + 1]))

        new_routes = solution.routes[:self.vehicle_id] + [new_route] + solution.routes[self.vehicle_id + 1:]
        return Solution(new_routes, solution.depot)


class RelocateOperator(BaseOperator):
    """Move a node from one position in a route to another, possibly in a different route."""
    def __init__(self, source_vehicle_id: int, source_position: int, target_vehicle_id: int, target_position: int):
        self.source_vehicle_id = source_vehicle_id
        self.source_position = source_position
        self.target_vehicle_id = target_vehicle_id
        self.target_position = target_position

    def run(self, solution: Solution) -> Solution:
        new_routes = [route[:] for route in solution.routes]
        node = new_routes[self.source_vehicle_id].pop(self.source_position)
        assert node != solution.depot
        # If target route is empty, append the node directly
        if not new_routes[self.target_vehicle_id]:
            new_routes[self.target_vehicle_id].append(node)
        else:
            # Insert node into target route at the specified position
            new_routes[self.target_vehicle_id].insert(self.target_position, node)
        return Solution(new_routes, solution.depot)


class MergeRoutesOperator(BaseOperator):
    """Merge two routes by appending the route of the source vehicle to the beginning of the route of the target vehicle. 
    The merged route is assigned to the target vehicle, and the source vehicle's route is cleared."""
    def __init__(self, source_vehicle_id: int, target_vehicle_id: int):
        self.source_vehicle_id = source_vehicle_id
        self.target_vehicle_id = target_vehicle_id

    def run(self, solution: Solution) -> Solution:
        new_routes = [route[:] for route in solution.routes]
        depot_index1 = new_routes[self.source_vehicle_id].index(solution.depot)
        rotated_route1 = new_routes[self.source_vehicle_id][depot_index1 + 1:] + new_routes[self.source_vehicle_id][:depot_index1]
        depot_index2 = new_routes[self.target_vehicle_id].index(solution.depot)
        rotated_route2 = new_routes[self.target_vehicle_id][depot_index2 + 1:] + new_routes[self.target_vehicle_id][:depot_index2]

        # Append source route to target route, then clear the source route
        new_routes[self.target_vehicle_id] = [solution.depot] + rotated_route1 + rotated_route2
        new_routes[self.source_vehicle_id] = [solution.depot]
        return Solution(new_routes, solution.depot)
