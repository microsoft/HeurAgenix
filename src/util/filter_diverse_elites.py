import os
import sys
import glob
import pickle
import argparse
import shutil
import time

# Add workspace root to sys.path
cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.append(cwd)

try:
    from src.problems.max_cut.components import Solution
except ImportError:
    # Fallback mock if import fails (though it shouldn't in this workspace)
    class Solution:
        pass

def calculate_hamming_distance(s1, s2):
    # MaxCut solution is a partition. A node is either in set_a or set_b.
    # Distance is min flips to match partitions.
    # Since (A, B) is same cut as (B, A), we check both alignments.
    
    s1_a = s1.set_a
    s2_a = s2.set_a
    # s2_b = s2.set_b
    
    d1 = len(s1_a.symmetric_difference(s2_a))
    
    # If s2.B is not readily available or reliable, we assume total nodes N
    # d2 = N - d1
    # But explicit set_b calculation is safer
    if hasattr(s2, 'set_b') and s2.set_b:
        d2 = len(s1_a.symmetric_difference(s2.set_b))
    else:
        # Approximate invert
        d2 = 99999999 # Cannot calc without B or N
        
    return min(d1, d2)

def get_diverse_elites(instance_name, top_k=20, threshold=200, base_output_dir=None):
    """
    Scans the elite_pool for the given instance, filters solution to ensure diversity,
    and returns a list of selected solution path/objects.
    
    Returns:
        list of dict: [{'path': str, 'val': float, 'obj': Solution}, ...]
    """
    if base_output_dir is None:
        # Try to infer from environment or default
        if os.getenv("AMLT_OUTPUT_DIR"):
            base_output_dir = os.path.join(os.getenv("AMLT_OUTPUT_DIR"), "..", "..", "orllm", "output")
        else:
             base_output_dir = "output"
             
    # Old structure: output/max_cut/elite_pool/{instance_name}
    # New structure: output/max_cut/{instance_name}/elite_pool
    pool_dir = os.path.join(base_output_dir, "max_cut", instance_name, "elite_pool")
    
    if not os.path.exists(pool_dir):
        print(f"Warning: Directory {pool_dir} not found. Returning empty list.")
        return []

    print(f"Scanning {pool_dir} for pickle files...", flush=True)
    # Recursive search
    pkl_files = glob.glob(os.path.join(pool_dir, "**", "*.pkl"), recursive=True)
    
    if not pkl_files:
        print("No solution files found.", flush=True)
        return []

    print(f"Found {len(pkl_files)} files. Loading headers/content...", flush=True)
    
    solutions = []
    
    start_time = time.time()
    for fpath in pkl_files:
        try:
            # Quick check filename for value? 
            # sol_{value}_{timestamp}_...
            fname = os.path.basename(fpath)
            parts = fname.split('_')
            val = 0
            if len(parts) >= 2 and parts[0] == 'sol':
                try:
                    val = float(parts[1])
                except ValueError:
                    val = 0
            
            with open(fpath, 'rb') as f:
                sol = pickle.load(f)
                # Ensure object has value consistent with filename (or trust object)
                if not hasattr(sol, 'cut_value'):
                    sol.cut_value = val
                solutions.append({'obj': sol, 'path': fpath, 'val': sol.cut_value})
        except Exception as e:
            continue
            
    print(f"Loaded {len(solutions)} valid solutions in {time.time()-start_time:.2f}s.", flush=True)
    
    # Sort descending
    solutions.sort(key=lambda x: x['val'], reverse=True)
    
    final_selection = []
    
    print(f"Selecting top {top_k} diverse solutions (Threshold={threshold})...", flush=True)
    for item in solutions:
        candidate = item['obj']
        if len(final_selection) >= top_k:
            break
            
        is_duplicate = False
        for selected_item in final_selection:
            selected = selected_item['obj']
            dist = calculate_hamming_distance(candidate, selected)
            
            if dist < threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            final_selection.append(item)
            
    return final_selection

def main():
    parser = argparse.ArgumentParser(description="Filter and diversity elite pool solutions")
    parser.add_argument("instance_name", type=str, help="Instance name (e.g. sg3dl141000.txt)")
    parser.add_argument("-k", "--top_k", type=int, default=20, help="Number of solutions to keep")
    parser.add_argument("-t", "--threshold", type=int, default=200, help="Min hamming distance to be considered different")
    
    args = parser.parse_args()
    
    # Call the reusable function
    selection = get_diverse_elites(args.instance_name, args.top_k, args.threshold)
    
    for i, item in enumerate(selection):
        print(f"Rank {i+1}: Val={item['val']} | {os.path.basename(item['path'])}")
        
    # Legacy Save Logic (Optional, kept for backward compatibility if running as script)
    base_dir = "output/max_cut"
    pool_dir = os.path.join(base_dir, args.instance_name, "elite_pool")
    output_dir = os.path.join(pool_dir, "high_quality_solution")
    
    if os.path.exists(pool_dir): # Only save if pool_dir exists locally
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        
        for item in selection:
            fname = os.path.basename(item['path'])
            dest = os.path.join(output_dir, fname)
            shutil.copy2(item['path'], dest)
            
        print(f"Saved to {output_dir}")

if __name__ == "__main__":
    main()
