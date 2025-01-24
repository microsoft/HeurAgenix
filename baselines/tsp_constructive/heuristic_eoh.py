# This heuristic is generated by [EoH](https://github.com/FeiLiu36/EoH) and run by GPT-4o.
# In order to be compatible with the framework, we modified the function name and interface while keeping the core of the algorithm. 

import numpy as np

def heuristic_eoh(current_node: int, destination_node: int, unvisited_nodes: set, distance_matrix: np.ndarray) -> int:
    # Adjusted threshold to potentially improve decision making
    threshold = 0.6
    
    # Slightly refined coefficients to balance influence
    c1, c2, c3, c4 = 0.5, 0.25, 0.15, 0.1
    
    scores = {}
    for node in unvisited_nodes:
        all_distances = [distance_matrix[node][i] for i in unvisited_nodes if i != node]
        
        # Add small epsilon to avoid division by zero issues
        epsilon = 1e-5
        
        # Use median instead of mean for robustness against outliers
        median_distance_to_unvisited = np.median(all_distances)
        
        # Normalize standard deviation to keep scores comparable
        std_dev_distance_to_unvisited = np.std(all_distances) / (np.mean(all_distances) + epsilon)
        
        score = (c1 * distance_matrix[current_node][node] 
                 - c2 * median_distance_to_unvisited 
                 + c3 * std_dev_distance_to_unvisited 
                 - c4 * distance_matrix[destination_node][node])
        
        scores[node] = score
    
    # Choose the node with the minimum score as the next node
    next_node = min(scores, key=scores.get)
    
    return next_node