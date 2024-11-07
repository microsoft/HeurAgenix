import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from src.problems.dposp.env import Env
from src.problems.dposp.components import *

class ORSolver:
    """OR Solver to find the upper bound for DPOSP.
    To save the calculation time, the data for production, transition and order deadline will be set to int."""
    def __init__(self, **kwargs):
        pass
    
    def run(self, env: Env, time_limitation: int=600, **kwargs) -> None:
        vehicles = list(range(env.production_line_num))
        orders = list(range(env.order_num))
        vehicle_num = len(vehicles)
        task_num = len(orders) + 1  # +1 for the depot

        # Create VRPTW model
        manager = pywrapcp.RoutingIndexManager(task_num, vehicle_num, 0)
        routing = pywrapcp.RoutingModel(manager)

        # Set the longest task duration as inf.
        inf = int(env.order_deadline.max()) + 1

        def _travel_time(src_order, dst_order, production_line):
            if src_order is None or dst_order is None:
                return 0
            if env.transition_time[production_line, env.order_product[src_order], env.order_product[dst_order]] == np.inf:
                return inf
            return env.transition_time[production_line, env.order_product[src_order], env.order_product[dst_order]]

        def _service_time(dst_order, production_line):
            if dst_order is None:
                return 0
            if env.production_rate[production_line, env.order_product[dst_order]] == 0:
                return inf
            return env.order_quantity[dst_order] / env.production_rate[production_line, env.order_product[dst_order]]

        def _time_callback(src_task, dst_task, vehicle):
            src_node = manager.IndexToNode(src_task)
            dst_node = manager.IndexToNode(dst_task)
            src_order = None if src_node == 0 else src_node - 1
            dst_order = None if dst_node == 0 else dst_node - 1
            production_line = vehicles[vehicle]

            travel_time = _travel_time(src_order, dst_order, production_line)
            service_time = _service_time(dst_order, production_line)

            if travel_time + service_time >= inf:
                return inf
            return int(travel_time + service_time)

        def _create_transition_call_back(vehicle):
            return lambda src_task, dst_task: _time_callback(src_task, dst_task, vehicle)

        transit_callback_indices = []
        for vehicle_id in range(vehicle_num):
            transit_callback = _create_transition_call_back(vehicle_id)
            transit_callback_index = routing.RegisterTransitCallback(transit_callback)
            transit_callback_indices.append(transit_callback_index)
            routing.SetArcCostEvaluatorOfVehicle(transit_callback_index, vehicle_id)

        # Add time dimension
        routing.AddDimensionWithVehicleTransitAndCapacity(
            evaluator_indices=transit_callback_indices,
            slack_max=0,
            vehicle_capacities=[inf] * vehicle_num,
            fix_start_cumul_to_zero=False,
            name="Time"
        )

        # Add disjunctions to allow partial solutions
        for task in range(1, manager.GetNumberOfNodes()):
            penalty = 1000000
            routing.AddDisjunction([manager.NodeToIndex(task)], penalty)

        # Set time window for tasks
        time_dimension = routing.GetDimensionOrDie('Time')
        for order in orders:
            index = manager.NodeToIndex(order + 1)
            time_dimension.CumulVar(index).SetRange(0, int(env.order_deadline[order]))

        # Search solution with time limitation
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.GLOBAL_CHEAPEST_ARC)
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC
        search_parameters.time_limit.seconds = time_limitation
        routing_result = routing.SolveWithParameters(search_parameters)

        if routing_result:
            for vehicle in range(manager.GetNumberOfVehicles()):
                index = routing.Start(vehicle)
                while not routing.IsEnd(index):
                    task = manager.IndexToNode(index)
                    if task != 0:
                        order = task - 1
                        result = env.run_operator(AppendOperator(vehicle, order))
                        assert result is True
                    index = routing_result.Value(routing.NextVar(index))
        return env.state_data