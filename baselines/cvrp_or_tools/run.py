import os
import tsplib95
import networkx as nx  
import numpy as np  
from ortools.constraint_solver import routing_enums_pb2  
from ortools.constraint_solver import pywrapcp
  
def load_data(data_path: str):  
    try:
        problem = tsplib95.load(data_path)   
        depot = problem.depots[0] - 1  
        distance_matrix = nx.to_numpy_array(problem.get_graph())  
        node_num = len(distance_matrix)  
        vehicle_num = int(open(data_path).readlines()[-1].strip().split(" : ")[-1])  
        capacity = problem.capacity  
        demands = np.array(list(problem.demands.values()))  
        return distance_matrix, depot, vehicle_num, capacity, demands  
    except Exception as e:
        return None, None, None, None, None
  
def create_data_model(data_path):  
    """Stores the data for the problem."""  
    data = {}  
    distance_matrix, depot, vehicle_num, capacity, demands = load_data(data_path)  
    if distance_matrix is None:
        return None
    data['distance_matrix'] = distance_matrix  
    data['num_vehicles'] = vehicle_num  
    data['depot'] = depot  
    data['demands'] = demands  
    data['vehicle_capacities'] = [capacity] * vehicle_num  
    return data  
  
def solve(data_path):  
    # Instantiate the data problem.  
    data = create_data_model(data_path)
    if data is None:
        return None
  
    # Create the routing index manager.  
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])  
  
    # Create Routing Model.  
    routing = pywrapcp.RoutingModel(manager)  
  
    # Create and register a transit callback.  
    def distance_callback(from_index, to_index):  
        """Returns the distance between the two nodes."""  
        from_node = manager.IndexToNode(from_index)  
        to_node = manager.IndexToNode(to_index)  
        return data['distance_matrix'][from_node][to_node]  
  
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)  
  
    # Define cost of each arc.  
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)  
  
    # Add Capacity constraint.  
    def demand_callback(from_index):  
        """Returns the demand of the node."""  
        from_node = manager.IndexToNode(from_index)  
        return data['demands'][from_node]  
  
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)  
    routing.AddDimensionWithVehicleCapacity(  
        demand_callback_index,  
        0,  # null capacity slack  
        data['vehicle_capacities'],  # vehicle maximum capacities  
        True,  # start cumul to zero  
        'Capacity')  
  
    # Setting first solution heuristic.  
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()  
    search_parameters.first_solution_strategy = (  
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
  
    # Solve the problem.  
    solution = routing.SolveWithParameters(search_parameters)  
  
    # Print solution on console.  
    return solution.ObjectiveValue()
  
if __name__ == "__main__":
    data_dir = os.path.join("..", "..", "output", "cvrp", "data", "test_data")
    data_name = "A-n80-k10.vrp"
    data_path = os.path.join(data_dir, data_name)

    cost = solve(data_path)
    print(data_name, cost)