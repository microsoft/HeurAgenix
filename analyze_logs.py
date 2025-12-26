import os
import glob
import re
from collections import defaultdict
import datetime

def parse_logs(log_dir):
    log_files = glob.glob(os.path.join(log_dir, "*.log"))
    
    # Structure: dataset -> exp -> run_id -> list of updates
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    rejections = defaultdict(lambda: defaultdict(int))
    
    # Regex
    data_pattern = re.compile(r"Data:(?P<data>[\w\.]+)\tExp:(?P<exp>[\w_]+)\tID:(?P<id>\d+)\tSteps:(?P<steps>\d+)\tSelected:(?P<selected>\d+)\tTotal:(?P<total>\d+)\tInit:(?P<init>[\d\.]+)\tNow:(?P<now>[\d\.]+)\tCurrent best:(?P<curr_best>[\d\.]+)\tBest known:(?P<best_known>[\d\.]+)\tNow:(?P<timestamp>.*?)\tTime cost\(hour\):(?P<time>[\d\.]+)")
    quality_gate_pattern = re.compile(r"Data:(?P<data>[\w\.]+)\tExp:(?P<exp>[\w_]+).*\[Quality Gate\].*Aborting run")

    for log_file in log_files:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                match = data_pattern.search(line)
                if match:
                    data = match.group('data')
                    exp = match.group('exp')
                    run_id = match.group('id')
                    
                    try:
                        dt = datetime.datetime.strptime(match.group('timestamp'), '%Y-%m-%d %H:%M:%S')
                    except:
                        dt = datetime.datetime.now()

                    results[data][exp][run_id].append({
                        'init': float(match.group('init')),
                        'value': float(match.group('now')),
                        'best_known': float(match.group('best_known')),
                        'time_cost': float(match.group('time')),
                        'steps': int(match.group('steps')),
                        'timestamp': dt
                    })
                    continue

                match = quality_gate_pattern.search(line)
                if match:
                    data = match.group('data')
                    exp = match.group('exp')
                    rejections[data][exp] += 1

    return results, rejections

def analyze_results(results, rejections):
    print(f"{'Dataset':<10} | {'Exp (Time)':<15} | {'Runs':<4} | {'Rej':<4} | {'BestKnown':<9} | {'MyBest':<9} | {'Gap%':<6} | {'MaxStep':<7} | {'Stagnation(h)':<13} | {'Status'}")
    print("-" * 115)

    for data_name in sorted(results.keys()):
        exps = results[data_name]
        for exp_name in sorted(exps.keys()):
            runs = exps[exp_name]
            num_runs = len(runs)
            num_rej = rejections[data_name][exp_name]
            
            if num_runs == 0:
                print(f"{data_name:<10} | {exp_name.split('_')[-1]:<15} | {0:<4} | {num_rej:<4} | {'N/A':<9} | {'N/A':<9} | {'N/A':<6} | {'N/A':<7} | {'N/A':<13} | {'Dead'}")
                continue

            # Aggregate stats
            all_updates = []
            for r_id, updates in runs.items():
                all_updates.extend(updates)
            
            # Sort by time
            all_updates.sort(key=lambda x: x['timestamp'])
            
            if not all_updates:
                continue

            best_known = all_updates[0]['best_known']
            my_best = max(u['value'] for u in all_updates)
            gap = (best_known - my_best) / best_known * 100
            max_steps = max(u['steps'] for u in all_updates)
            
            # Find when the global best for this exp was FIRST found
            best_updates = [u for u in all_updates if u['value'] == my_best]
            first_hit_best = min(best_updates, key=lambda x: x['timestamp'])
            
            # Latest activity in this experiment
            latest_activity = all_updates[-1]['timestamp']
            
            # Stagnation: Time since we last found a NEW global best for this experiment
            stagnation_duration = latest_activity - first_hit_best['timestamp']
            stagnation_hours = stagnation_duration.total_seconds() / 3600
            
            status = "Active"
            if stagnation_hours > 4:
                status = "Stagnated"
            if stagnation_hours > 10:
                 status = "Converged?"
            
            # Shorten exp name to just time part if possible
            exp_short = exp_name.split('_')[-1] if '_' in exp_name else exp_name
            
            print(f"{data_name:<10} | {exp_short:<15} | {num_runs:<4} | {num_rej:<4} | {best_known:<9.0f} | {my_best:<9.0f} | {gap:<6.2f} | {max_steps:<7} | {stagnation_hours:<13.2f} | {status}")

if __name__ == "__main__":
    log_dir = r"d:\ORLLM\max_cut\log"
    results, rejections = parse_logs(log_dir)
    analyze_results(results, rejections)
