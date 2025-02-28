import os
import os
# 设置环境变量，强制修改缓存目录
os.environ["TRANSFORMERS_CACHE"] = "/Data/haolong/model_deploy/models_cache/"
from src.util.gpt_helper import GPTHelper


gpt_helper = GPTHelper(output_dir=os.path.join("output", "chat"))

A = '''
I am working on Hyper-heuristics for Combinatorial Operation (CO) problem.
In this conversation, I will introduce the problem and then framework we have built now, you just remember this.
In next conversation, I will describe the challenges I'm encountering and explore how we can collaborate to resolve them.

Currently, I am working on tsp problem:
Traveling Salesman Problem (TSP) is the challenge of finding the shortest possible route that visits a given list of cities exactly once and returns to the origin city, based on the distances between each pair of cities.

To support different heuristic algorithms, I build the Solution and Operator framework.
The Solution is designed as:
class Solution:
    """The solution of TSP.
A list of integers where each integer represents a node (city) in the TSP tour.
The order of the nodes in the list defines the order in which the cities are visited in the tour."""
    def __init__(self, tour: list[int]):
        self.tour = tour
Operator servers as a mechanism to modify solution, which enables the application of heuristic algorithms. 
To support heuristic algorithm, we have build the following operators:
class AppendOperator(BaseOperator):
    """Append the node at the end of the solution."""
    def __init__(self, node: int):
        self.node = node
    def run(self, solution: Solution) -> Solution:
        new_tour = solution.tour + [self.node]
        return Solution(new_tour)
class InsertOperator(BaseOperator):
    """Insert the node into the solution at the target position."""
    def __init__(self, node: int, position: int):
        self.node = node
        self.position = position
    def run(self, solution: Solution) -> Solution:
        new_tour = solution.tour[:self.position] + [self.node] + solution.tour[self.position:]
        return Solution(new_tour)
class SwapOperator(BaseOperator):
    """Swap two nodes in the solution. swap_node_pairs is a list of tuples, each containing the two nodes to swap."""
    def __init__(self, swap_node_pairs: list[tuple[int, int]]):
        self.swap_node_pairs = swap_node_pairs
    def run(self, solution: Solution) -> Solution:
        node_to_index = {node: index for index, node in enumerate(solution.tour)}
        new_tour = solution.tour.copy()
        for node_a, node_b in self.swap_node_pairs:
            index_a = node_to_index.get(node_a)
            index_b = node_to_index.get(node_b)
            assert index_a is not None
            assert index_b is not None
            new_tour[index_a], new_tour[index_b] = (new_tour[index_b], new_tour[index_a])
        return Solution(new_tour)
class ReplaceOperator(BaseOperator):
    """Replace a node with another one in the solution."""
    def __init__(self, node: int, new_node: int):
        self.node = node
        self.new_node = new_node
    def run(self, solution: Solution) -> Solution:
        index = solution.tour.index(self.node)
        new_tour = solution.tour[:index] + [self.new_node] + solution.tour[index + 1:]
        return Solution(new_tour)
class ReverseSegmentOperator(BaseOperator):
    """Reverse multiple segments of indices in the solution."""
    def __init__(self, segments: list[tuple[int, int]]):
        self.segments = segments
    def run(self, solution: Solution) -> Solution:
        new_tour = solution.tour[:]
        for segment in self.segments:
            start_index, end_index = segment
            assert 0 <= start_index < len(new_tour)
            assert 0 <= end_index < len(new_tour)
            if start_index <= end_index:
                new_tour[start_index:end_index + 1] = reversed(new_tour[start_index:end_index + 1])
            else:
                new_tour = list(reversed(new_tour[start_index:])) + new_tour[end_index + 1:start_index] + list(reversed(new_tour[:end_index + 1]))
        return Solution(new_tour)

In pursuit of augmenting our heuristic algorithmic suite, we require the following standardized heuristic function signature:
def heuristic(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: call, **kwargs) -> tuple[TargetOperatorType, dict]:
The inputs are:
global_data (dict): The global data dict containing the global instance data with:
    - "node_num" (int): The total number of nodes in the problem.
    - "distance_matrix" (numpy.ndarray): A 2D array representing the distances between nodes.
state_data (dict): The state data dict containing the solution state data with:
    - "current_solution" (Solution): An instance of the Solution class representing the current solution.
    - "visited_nodes" (list[int]): A list of integers representing the IDs of nodes that have been visited.
    - "visited_num" (int): Number of nodes that have been visited.
    - "unvisited_nodes" (list[int]): A list of integers representing the IDs of nodes that have not yet been visited.
    - "unvisited_num" (int): Number of nodes that have not been visited.
    - "current_cost" (int): The total cost of current solution. The cost to return to the starting point is not included until the path is fully constructed.
    - "last_visited" (int): The last visited node.
    - "validation_solution" (callable): def validation_solution(solution: Solution) -> bool: function to check whether new solution is valid.
algorithm_data(dict): Algorithm data contains the data that necessary for some algorithms.
get_state_data_function(callable): The get_state_data_function is the function that receives the new solution as input and return the state dictionary for new solution. It will not modify the origin solution.
Other hyper-parameters in kwargs.
The outputs includes the operator that must be an instance of a predefined target operator type and updated algorithm dict, which contains new information for future work for both this or other algorithm.

The response format is very important. For better communication, please respond to me in this format:
***I remembered
***
Ensure there is no other content inside the ***, and analysis outside *** are welcome.
If you have no information to provide, simply respond with ***None***.
'''

B = 'hi! What is your name?'
C = [{'role': 'user', 'content': [{"type": "text", "text": 'hi!'}]}, {'role': 'assistant', 'content': [{"type": "text", "text": 'Hello! How can I assist you today?'}]}, {'role': 'user', 'content': [{"type": "text", "text": 'What is your name?'}]}]
D = [{'role': 'user', 'content': 'hi!'}, {'role': 'assistant', 'content': 'Hello! How can I assist you today?'}, {'role': 'user', 'content': 'What is your name?'}]
# # gpt_helper.load(B)
# gpt_helper.chat(C)
# response = gpt_helper.dump()
# print(response)


import transformers
import torch
model_id = "/Data/haolong/model_deploy/models_cache/models--deepseek-ai--DeepSeek-R1-Distill-Llama-70B/snapshots/1772b078b94935926dcc8715c1afdd04ae447080"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

outputs = pipeline(
    B,
    max_new_tokens=256,
)
print(outputs)
print(outputs[0]["generated_text"][-1])