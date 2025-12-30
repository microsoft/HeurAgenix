import numpy as np
import scipy.sparse as sp
from src.problems.base.components import BaseOperator
from src.problems.max_cut.components import Solution, BatchInsertNodeOperator

def cosm_heuristic_detailed(problem_state: dict, algorithm_data: dict, steps: int = None, **kwargs) -> tuple[BaseOperator, dict]:
    """
    Cosm (Continuous Spin Machine) Heuristic - CPU Optimized Version (Detailed Mode).
    
    A hardware-centric inspired heuristic that relaxes the discrete Max Cut problem 
    into a continuous dynamical system. It simulates the evolution of spins under 
    a mean-field coupling to find high-quality ground states.
    
    Mechanism:
    1. Relax s_i in {-1, 1} to continuous x_i in R.
    2. Evolve x_i using gradient descent on the energy landscape with a bifurcation term.
    3. Discretize x_i back to s_i.
    
    Optimizations for CPU:
    - Uses scipy.sparse for efficient Matrix-Vector Multiplication (MVM) on sparse graphs (G-set).
    - Uses a momentum-based update rule for faster convergence.
    """
    
    node_num = problem_state["node_num"]
    
    # Dynamic steps based on graph size if not provided
    # For Detailed mode: We want precision.
    # Formula: 200 + 3 * sqrt(N)
    # For 20k nodes: 200 + 3*141 = 623 steps.
    if steps is None:
        steps = int(200 + 3 * np.sqrt(node_num))
    
    # 1. Build or Retrieve Sparse Weight Matrix (J)
    # We cache this matrix in algorithm_data to avoid rebuilding it every time.
    # For Max Cut, we want to maximize Cut = sum W_ij (1 - s_i s_j) / 2
    # This is equivalent to minimizing Ising Energy E = -0.5 * sum J_ij s_i s_j
    # where J_ij = -W_ij (Anti-ferromagnetic coupling).
    
    if "cosm_J_matrix" in algorithm_data:
        J = algorithm_data["cosm_J_matrix"]
    else:
        adj = problem_state["adj"]
        # Build CSR matrix
        rows = []
        cols = []
        data = []
        
        # G-set graphs are undirected. adj[u][v] = w means edge (u, v) has weight w.
        # We need to ensure symmetric entries for the matrix multiplication.
        for u in range(node_num):
            for v, w in adj[u].items():
                rows.append(u)
                cols.append(v)
                data.append(-w) # J_ij = -W_ij
        
        # Create sparse matrix (float32 for speed)
        J = sp.csr_matrix((data, (rows, cols)), shape=(node_num, node_num), dtype=np.float32)
        algorithm_data["cosm_J_matrix"] = J

    # 2. Initialization
    # Start with small random noise around 0
    x = np.random.normal(0, 0.1, node_num).astype(np.float32)
    # Momentum vector
    v = np.zeros(node_num, dtype=np.float32)
    
    # 3. Dynamics Parameters
    dt = 0.2
    mass = 1.0
    friction = 0.05 # Damping to settle into minima
    
    # Pump parameter (Bifurcation control)
    # Linearly anneal from negative (stable at 0) to positive (bistable at +/- 1)
    alpha_start = -0.5
    alpha_end = 1.5
    
    # 4. Evolution Loop
    # MVM is the bottleneck, but sparse matrix makes it fast on CPU.
    for i in range(steps):
        # Current pump value
        alpha = alpha_start + (alpha_end - alpha_start) * (i / steps)
        
        # Calculate Mean Field: h = J @ x
        # This is the most expensive operation
        h = J.dot(x)
        
        # Force calculation
        # F = (alpha - x^2) * x + h
        # (alpha - x^2) * x is the bifurcation term (double-well potential derivative)
        # h is the coupling term
        # We use a simplified form: F = (alpha - 1) * x + h for stability
        # Or the full cubic form: F = alpha * x - x**3 + h
        
        # Using cubic form for better separation:
        force = alpha * x - np.power(x, 3) + h
        
        # Symplectic update with friction (Momentum)
        v = (1 - friction) * v + force * dt
        x += v * dt / mass
        
        # Clip to prevent explosion (numerical stability)
        np.clip(x, -2.0, 2.0, out=x)
        
    # 5. Discretization
    # Map continuous x back to discrete sets
    # x > 0 -> Set A, x <= 0 -> Set B
    
    # We need to identify which nodes go where.
    # Since this is a constructive heuristic, we assume we are filling the solution.
    # However, the Env expects an Operator.
    # We will return a BatchInsertNodeOperator that assigns ALL nodes.
    # Note: This assumes the current solution is empty or we are overwriting it.
    # If the solution is partial, we should only assign unselected nodes.
    
    unselected_nodes = problem_state.get("unselected_nodes", None)
    
    # If unselected_nodes is explicitly provided and empty, it means the solution is complete.
    # In this case, we should stop (return None) to avoid infinite loops in SingleHyperHeuristic.
    if unselected_nodes is not None and len(unselected_nodes) == 0:
        return None, algorithm_data
    
    nodes_to_a = []
    nodes_to_b = []
    
    if unselected_nodes:
        # Only assign unselected nodes based on their x values
        # This allows Cosm to be used as a repair/completion heuristic too
        for node in unselected_nodes:
            if x[node] > 0:
                nodes_to_a.append(node)
            else:
                nodes_to_b.append(node)
    else:
        # Assign all nodes (Overwrite mode or fresh start)
        # Note: If solution is not empty, BatchInsert might fail or duplicate.
        # But usually this is called when solution is empty.
        # We iterate all nodes.
        # Using numpy mask for speed
        mask = x > 0
        nodes_to_a = np.where(mask)[0].tolist()
        nodes_to_b = np.where(~mask)[0].tolist()
        
    op = BatchInsertNodeOperator(nodes_to_a=nodes_to_a, nodes_to_b=nodes_to_b)
    
    return op, algorithm_data
