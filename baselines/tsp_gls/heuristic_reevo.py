# This heuristic is generated by [ReEvo](https://github.com/ai4co/reevo) and run by GPT-4o.
# In order to be compatible with the framework, we modified the function name and interface while keeping the core of the algorithm. 

import numpy as np

import numpy as np

def heuristic_reevo(distance_matrix, local_opt_tour, edge_n_used):
    # Calculate the average distance of each row
    average_distance = np.mean(distance_matrix, axis=1)
    
    # Calculate the distance ranking for each city
    distance_ranking = np.argsort(distance_matrix, axis=1)
    
    # Calculate the mean of the closest four cities, excluding the city itself
    closest_mean_distance = np.mean(
        distance_matrix[np.arange(distance_matrix.shape[0])[:, None], distance_ranking[:, 1:5]],
        axis=1
    )
    
    # Normalize the distance matrix by average distance
    indicators = distance_matrix / average_distance[:, np.newaxis]
    
    # Add weight to the closest mean distance
    np.fill_diagonal(indicators, np.inf)
    total_distance_sum = np.sum(distance_matrix, axis=1)
    weight_closest_mean = 0.5
    indicators += (weight_closest_mean * closest_mean_distance[:, np.newaxis]) / total_distance_sum[:, np.newaxis]
    
    return indicators