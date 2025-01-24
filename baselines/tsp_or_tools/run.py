
import os
import tsplib95  
import networkx as nx  
from ortools.constraint_solver import pywrapcp  
from ortools.constraint_solver import routing_enums_pb2  
def solve_tsp(data_file):  
    problem = tsplib95.load(data_file)  
    graph = problem.get_graph()  
  
    distance_matrix = nx.to_numpy_array(graph, nodelist=sorted(graph.nodes))  
  
    data = {  
        'distance_matrix': distance_matrix,  
        'num_vehicles': 1,
        'depot': 0
    }  

    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),  
                                           data['num_vehicles'], data['depot'])  
    routing = pywrapcp.RoutingModel(manager)  
  
    def distance_callback(from_index, to_index):  
        from_node = manager.IndexToNode(from_index)  
        to_node = manager.IndexToNode(to_index)  
        return int(data['distance_matrix'][from_node][to_node])  # 确保是整数  
  
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)  
  
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)  
  
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()  
    search_parameters.first_solution_strategy = (  
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)  
  
    solution = routing.SolveWithParameters(search_parameters)  
  
    if solution:
        route_distance = print_solution(manager, routing, solution)  
    return route_distance
  
def print_solution(manager, routing, solution):  
    index = routing.Start(0)  
    route_distance = 0  
    while not routing.IsEnd(index):  
        previous_index = index  
        index = solution.Value(routing.NextVar(index))  
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)  
    return route_distance  


if __name__ == "__main__":
    data_dir = os.path.join("..", "..", "output", "tsp", "data", "test_data")
    data_name = "a280.tsp"
    data_path = os.path.join(data_dir, data_name)
    cost = solve_tsp(data_path)
    print(data_name, cost)