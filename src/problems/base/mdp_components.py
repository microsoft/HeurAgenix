from src.problems.base.components import BaseSolution, BaseOperator

class Solution(BaseSolution):
    """The solution of MDP env."""
    def __init__(self, actions: list[list] = []):
        self.actions = actions

    def __str__(self) -> str:
        solution_string = ""
        for index, actions in enumerate(self.actions):
            solution_string += f"agent_{index}: " + ", ".join(map(str, actions)) + "\n"
        return solution_string


class ActionOperator(BaseOperator):
    def __init__(self, actions: list):
        self.actions = actions

    def run(self, solution: Solution) -> Solution:
        if solution.actions == []:
            actions = [[] for _ in self.actions]
        else:
            actions = solution.actions
        new_actions = [actions[agent_id][:] + [action] for agent_id, action in enumerate(self.actions)]
        return Solution(new_actions)