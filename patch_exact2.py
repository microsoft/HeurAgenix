with open("src/problems/cvrp/env.py", "r") as f:
    text = f.read()

import re

# Fix where old_actual_len is calculated
text = re.sub(
    r'old_actual_len = sum\(len\(route\) for route in solution\.routes\)',
    r'old_actual_len = sum(len(route) - 1 for route in solution.routes if len(route) > 0)',
    text
)
# Fix expected_len and new_actual_len in update cost
text = re.sub(
    r'expected_len = self\.instance_data\["node_num"\] - 1 \+ self\.instance_data\["vehicle_num"\]\s+new_actual_len = sum\(len\(route\) for route in solution\.routes\)',
    r'expected_len = self.instance_data["node_num"] - 1\n        new_actual_len = sum(len(route) - 1 for route in solution.routes if len(route) > 0)',
    text
)

with open("src/problems/cvrp/env.py", "w") as f:
    f.write(text)
