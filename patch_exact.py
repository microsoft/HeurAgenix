with open("src/problems/cvrp/env.py", "r") as f:
    text = f.read()

import re
text = re.sub(
    r'for vehicle_index in range\(self\.instance_data\["vehicle_num"\]\):\s+route = solution\.routes\[vehicle_index\]',
    r'for route in solution.routes:',
    text
)

text = re.sub(
    r'expected_len = self\.instance_data\["node_num"\] - 1 \+ self\.instance_data\["vehicle_num"\]\s+actual_len = sum\(len\(route\) for route in solution\.routes\)\s+if actual_len != expected_len:\s+total_current_cost \+= abs\(expected_len - actual_len\) \* 100000\.0',
    r'expected_customers = self.instance_data["node_num"] - 1\n        actual_customers = sum(len(route) - 1 for route in solution.routes if len(route) > 0)\n        if actual_customers != expected_customers:\n            total_current_cost += abs(expected_customers - actual_customers) * 100000.0',
    text
)

with open("src/problems/cvrp/env.py", "w") as f:
    f.write(text)
