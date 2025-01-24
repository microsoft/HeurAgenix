import numpy as np
from sklearn.preprocessing import StandardScaler

def heuristic_reevo(edge_attr: np.ndarray) -> np.ndarray:
    num_edges = edge_attr.shape[0]
    num_attributes = edge_attr.shape[1]
    heuristic_values = np.zeros_like(edge_attr)
    
    # Apply feature engineering on edge attributes
    transformed_attr = np.log1p(np.abs(edge_attr))  # Taking logarithm of absolute value of attributes
    
    # Normalize edge attributes
    scaler = StandardScaler()
    edge_attr_norm = scaler.fit_transform(transformed_attr)
    
    # Calculate correlation coefficients
    correlation_matrix = np.corrcoef(edge_attr_norm, rowvar=False)  # Ensure correct orientation
    
    # Calculate heuristic value for each edge attribute
    # Increase the influence of correlation by adjusting the formula
    influence_factor = 10  # Adjusted factor for more control over the influence
    for i in range(num_edges):
        for j in range(num_attributes):
            if edge_attr_norm[i][j] != 0:
                heuristic_values[i][j] = np.exp(-influence_factor * edge_attr_norm[i][j] * correlation_matrix[j][j])
    
    return heuristic_values