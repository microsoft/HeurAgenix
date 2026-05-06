with open("src/problems/cvrp/hyper_heuristics/hgs_island.py", "r") as f:
    text = f.read()

import re

new_diversity_check = """
        # Check against existing to maintain strict diversity (radius = 8 edges)
        is_duplicate = False
        for elite in self.elite_pool:
            if abs(elite['value'] - pure_cost) < 1e-4:
                dist = self._get_cvrp_distance(elite['fingerprint'], fingerprint)
                if dist < 8:
                    is_duplicate = True
                    break
            else:
                dist = self._get_cvrp_distance(elite['fingerprint'], fingerprint)
                # If they are practically identical (distance < 10) but different cost
                # Only keep the one with better cost to avoid clone swarms
                if dist < 10:
                    is_duplicate = True
                    if pure_cost < elite['value']:
                        elite['value'] = pure_cost
                        elite['fingerprint'] = fingerprint
                        if hasattr(env, 'current_solution'):
                            elite['routes'] = [list(r) for r in env.current_solution.routes]
                        elite['timestamp'] = time.time()
                        self.elite_pool.sort(key=lambda x: x['value'])
                    break
"""

text = re.sub(
    r'# Check against existing to maintain diversity\s+is_duplicate = False\s+for elite in self\.elite_pool:\s+if abs\(elite\[\'value\'\] - pure_cost\) < 1e-4:\s+dist = self\._get_cvrp_distance\(elite\[\'fingerprint\'\], fingerprint\)\s+if dist < 5: # Highly overlapping edges/structure\s+is_duplicate = True\s+break',
    new_diversity_check.strip('\n'),
    text
)

with open("src/problems/cvrp/hyper_heuristics/hgs_island.py", "w") as f:
    f.write(text)
