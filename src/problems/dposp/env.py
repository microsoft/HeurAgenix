import os
import pandas as pd
import numpy as np
from src.problems.base.env import BaseEnv
from src.problems.dposp.components import Solution


class Env(BaseEnv):
    """DPOSP env that stores the static global data, current solution, dynamic state and provide necessary to support algorithm."""
    def __init__(self, data_name: str, **kwargs):
        super().__init__(data_name, "dposp")
        self.construction_steps = self.instance_data["order_num"]
        self.key_item = "fulfilled_order_num"
        self.compare = lambda x, y: x - y

    @property
    def is_complete_solution(self) -> bool:
        return True

    def load_data(self, data_name: str) -> tuple:
        production_df = pd.read_csv(os.path.join(data_name, "production.tsv"), sep="\t")
        transition_df = pd.read_csv(os.path.join(data_name, "transition.tsv"), sep="\t")
        order_df = pd.read_csv(os.path.join(data_name, "order.tsv"), sep="\t")

        # Determine the number for production lines, products and orders
        production_line_num = max(production_df["ProductionLine"].max(), transition_df["ProductionLine"].max()) + 1
        product_num = max(production_df["Product"].max(), transition_df["SourceProduct"].max(), transition_df["DestinationProduct"].max(), order_df["Product"].max()) + 1
        order_num = len(order_df)

        # Initialize arrays
        production_rate = np.zeros((production_line_num, product_num))
        transition_time = np.zeros((production_line_num, product_num, product_num))
        order_product = np.zeros(len(order_df), dtype=int)
        order_quantity = np.zeros(len(order_df))
        order_deadline = np.zeros(len(order_df))

        # Fill production rates
        for _, row in production_df.iterrows():
            production_rate[row["ProductionLine"], row["Product"]] = row["ProductionRate"]

        # Fill transition times
        for _, row in transition_df.iterrows():
            if row["TransitionTime"] == "Forbidden":
                transition_time[row["ProductionLine"], row["SourceProduct"], row["DestinationProduct"]] = np.inf
            else:
                transition_time[row["ProductionLine"], row["SourceProduct"], row["DestinationProduct"]] = row["TransitionTime"]

        # Fill order details
        for i, row in order_df.iterrows():
            order_product[i] = row["Product"]
            order_quantity[i] = row["Quantity"]
            order_deadline[i] = row["Deadline"]

        return {"production_line_num":production_line_num, 
                "product_num": product_num, 
                "order_num": order_num, 
                "production_rate": production_rate,
                "transition_time": transition_time,
                "order_product": order_product,
                "order_quantity": order_quantity,
                "order_deadline": order_deadline
            }

    def init_solution(self) -> Solution:
        return Solution(production_schedule=[[] for _ in range(self.instance_data["production_line_num"])])

    def get_key_value(self, solution: Solution=None) -> float:
        fulfilled_orders = []
        for schedule in solution.production_schedule:
            if schedule:
                fulfilled_orders.extend(schedule)
        return len(set(fulfilled_orders))

    def helper_function(self) -> dict:
        return {"get_problem_state": self.get_problem_state, "validation_solution": self.validation_solution, "validation_single_production_schedule": self.validation_single_production_schedule, "get_time_cost_delta": self.get_time_cost_delta}

    def validation_single_production_schedule(self, line_id: int, production_schedule: list[int]) -> bool:
        if production_schedule == []:
            return True

        schedule_np = np.array(production_schedule)

        products = self.instance_data["order_product"][schedule_np]
        quantities = self.instance_data["order_quantity"][schedule_np]
        deadlines = self.instance_data["order_deadline"][schedule_np]

        if np.any(self.instance_data["production_rate"][line_id, products] <= 0):
            return False
        production_times = quantities / self.instance_data["production_rate"][line_id, products]

        if len(products) > 1:
            transitions = self.instance_data["transition_time"][line_id, products[:-1], products[1:]]

        end_times = np.cumsum(production_times + np.concatenate(([0], transitions))) if len(products) > 1 else production_times

        for i, order_id in enumerate(production_schedule):
            if end_times[i] > deadlines[i]:
                return False
        
        return True

    def get_time_cost_delta(self, production_line_id: int, order_id: int, position: int, solution: Solution=None) -> float:
        solution = self.current_solution if solution is None else solution
        line_schedule = solution.production_schedule[production_line_id]
        prev_product = None if position == 0 else self.instance_data["order_product"][line_schedule[position - 1]]
        next_product = None if position == len(line_schedule) or position == -1 else self.instance_data["order_product"][line_schedule[position]]
        quantity = self.instance_data["order_quantity"][order_id]
        product = self.instance_data["order_product"][order_id]
        production_time_for_order = quantity / self.instance_data["production_rate"][production_line_id, product]
        transition_time_before = 0 if prev_product is None else self.instance_data["transition_time"][production_line_id, prev_product, product]
        transition_time_after = 0 if next_product is None else self.instance_data["transition_time"][production_line_id, product, next_product]
        original_transition_time = 0 if prev_product is None or next_product is None else self.instance_data["transition_time"][production_line_id, prev_product, next_product]
        delta_time_cost = transition_time_before + production_time_for_order + transition_time_after - original_transition_time
        return delta_time_cost

    def validation_solution(self, solution: Solution=None) -> bool:
        """Check the validation of this solution in the following items:
            1. Order existence: Ensure that the order IDs in Solution's production_schedule are valid (i.e., < self.instance_data["order_num"]).
            2. No duplicate fulfilled orders: Ensure that no order appears more than once in the production schedule.
            3. Deadline fulfillment: Ensure that all orders are fulfilled before their respective deadlines.
        """
        if solution is None:
            solution = self.current_solution

        if not isinstance(solution, Solution) or not isinstance(solution.production_schedule, list):
            return False
        all_orders = [order for sublist in solution.production_schedule for order in sublist]

        # Check order existence
        if not np.all(np.array(all_orders) < self.instance_data["order_num"]):
            return False

        # Check no duplicate fulfilled orders
        if len(all_orders) != len(set(all_orders)):
            return False

        for line_id, schedule in enumerate(solution.production_schedule):
            if schedule:
                # Check order fulfillment
                if not self.validation_single_production_schedule(line_id, schedule):
                    return False
        return True
