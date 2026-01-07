import os
import glob
import pickle
import argparse
import random
import time
import shutil
from datetime import datetime

# Define Solution class structure to match what is expected by the loader
class Solution:
    def __init__(self, set_a, set_b, cut_value=None):
        self.set_a = set_a
        self.set_b = set_b
        self.cut_value = cut_value
    
    def __repr__(self):
        return f"Solution(value={self.cut_value}, len_a={len(self.set_a)}, len_b={len(self.set_b)})"

def parse_solution_file(filepath):
    """Parses a text solution file and returns a Solution object"""
    set_a = set()
    set_b = set()
    cut_value = 0.0
    
    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                if line.startswith("set_a:"):
                    content = line.split(":", 1)[1].strip()
                    if content:
                        set_a = {int(x) - 1 for x in content.split(",")}
                elif line.startswith("set_b:"):
                    content = line.split(":", 1)[1].strip()
                    if content:
                        set_b = {int(x) - 1 for x in content.split(",")}
                elif line.startswith("cut_value:"):
                    cut_value = float(line.split(":", 1)[1].strip())
        
        # Validation
        if not set_a and not set_b:
            return None
            
        return Solution(set_a, set_b, cut_value)
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Convert TXT solutions to PKL elite pool")
    parser.add_argument("source_dir", help="Directory containing search results (e.g. output/max_cut/search_best_result.fast_stop/g67.mc)")
    parser.add_argument("target_pool", help="Target elite pool directory (e.g. output/max_cut/elite_pool/g67.mc)")
    parser.add_argument("-k", "--top_k", type=int, default=20, help="Number of top solutions to keep")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.source_dir):
        print(f"Error: Source directory {args.source_dir} not found.")
        return

    print(f"Scanning {args.source_dir} for solution files...")
    
    # 1. Find all result txt files (final and intermediate)
    # Recursively find all txt files
    candidates = []
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(args.source_dir):
        for file in files:
            if file.endswith(".txt") and ("result" in file):
                full_path = os.path.join(root, file)
                
                # Extract score from filename if possible to avoid parsing everything
                # intermediate_result.6678.exp.run.txt
                score = -1
                try:
                    parts = file.split(".")
                    # Look for the numeric part
                    for p in parts:
                        if p.isdigit() and float(p) > 1000: # heuristic check
                            score = float(p)
                            break
                            
                    # If filename doesn't help, we might need to peek inside (expensive)
                    # For now rely on filename convention from our script
                    if score == -1 and "final_result" in file:
                         # Final result might not have score in name, parse it
                         sol = parse_solution_file(full_path)
                         if sol:
                             score = sol.cut_value
                except:
                    pass
                
                if score > 0:
                    candidates.append((score, full_path))

    if not candidates:
        print("No valid solution files found.")
        return

    # 2. Sort and Pick Top K
    # Sort descending by score
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    top_candidates = candidates[:args.top_k]
    print(f"Found {len(candidates)} solutions. Extracting Top {len(top_candidates)}:")
    for score, path in top_candidates:
        print(f"  - {score}: {os.path.basename(path)}")
        
    # 3. Prepare Target Directory (Time Bucket Sharding)
    # output/max_cut/elite_pool/g67.mc/YYYYMMDD_HH/shard_0/
    timestamp = time.time()
    dt = datetime.fromtimestamp(timestamp)
    bucket_name = dt.strftime("%Y%m%d_%H")
    
    # We put everything in shard_0 for simplicity in this batch import
    target_shard_dir = os.path.join(args.target_pool, bucket_name, "shard_0")
    os.makedirs(target_shard_dir, exist_ok=True)
    print(f"\nTarget Directory: {target_shard_dir}")
    
    # 4. Convert and Save
    count = 0
    for score, path in top_candidates:
        sol = parse_solution_file(path)
        if sol:
            # Filename: sol_{value}_{timestamp}_{worker}_{rand}.pkl
            # Use fake worker_id "imported"
            fname = f"sol_{int(sol.cut_value)}_{int(timestamp)}_{'imported'}_{random.randint(1000, 9999)}.pkl"
            target_path = os.path.join(target_shard_dir, fname)
            
            try:
                with open(target_path, "wb") as f:
                    pickle.dump(sol, f)
                count += 1
                # print(f"  Converted -> {fname}")
            except Exception as e:
                print(f"  Failed to write {fname}: {e}")
    
    print(f"\nSuccessfully imported {count} elite solutions into pool.")

if __name__ == "__main__":
    main()
