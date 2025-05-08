import numpy as np

def heuristic_base(values, weights_norm):  
    values_min, values_max = np.min(values), np.max(values)  
    values_norm = (values - values_min) / (values_max - values_min) if values_max > values_min else np.ones_like(values)  
    value_density = np.divide(values_norm[:, np.newaxis], weights_norm, out=np.zeros_like(weights_norm), where=weights_norm!=0)  
    max_value_density = np.max(value_density, axis=1)  
    min_density, max_density = np.min(max_value_density), np.max(max_value_density)  
    heuristics = (max_value_density - min_density) / (max_density - min_density) if max_density > min_density else np.ones_like(max_value_density)  
      
    return heuristics