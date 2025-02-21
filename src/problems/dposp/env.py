import os
import pandas as pd
import numpy as np
from src.problems.base.env import BaseEnv
from src.problems.dposp.components import Solution


class Env(BaseEnv):
    """DPOSP env that stores the static global data, current solution, dynamic state and provide necessary to support algorithm."""
    def __init__(self, data_name: str, **kwargs):
        super().__init__(data_name, "dposp")
        self.production_line_num, self.product_num, self.order_num, self.production_rate, self.transition_time, self.order_product, self.order_quantity, self.order_deadline = self.data
        self.construction_steps = self.order_num
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

        return production_line_num, product_num, order_num, production_rate, transition_time, order_product, order_quantity, order_deadline

    def init_solution(self) -> Solution:
        return Solution(production_schedule=[[] for _ in range(self.production_line_num)])

    def get_global_data(self) -> dict:
        """Retrieve the global static information data as a dictionary.

        Returns:
            dict: A dictionary containing the global static information data with:
                - "product_num" (int): Total number of unique products.
                - "production_line_num" (int): Total number of production lines.
                - "order_num" (int): Total number of orders to be scheduled.
                - "production_rate" (numpy.array): 2D array of production time for each product on each production line.
                - "transition_time" (numpy.array): 3D array of transition time between products on each production line.
                - "order_product" (numpy.array): 1D array mapping each order to its required product.
                - "order_quantity" (numpy.array): 1D array of the quantity required for each order.
                - "order_deadline" (numpy.array): 1D array of the deadline for each order.
        """

        global_data_dict = {
            "product_num": self.product_num,
            "production_line_num": self.production_line_num,
            "order_num": self.order_num,
            "production_rate": self.production_rate,
            "transition_time": self.transition_time,
            "order_product": self.order_product,
            "order_quantity": self.order_quantity,
            "order_deadline": self.order_deadline,
        }
        return global_data_dict

    def get_state_data(self, solution: Solution=None) -> dict:
        """Retrieve the current dynamic state data as a dictionary.

        Returns:
            dict: A dictionary containing the current dynamic state data with:
                - "current_solution" (Solution): Current scheduling solution.
                - "fulfilled_order_num" (int): Number of fulfilled orders.
                - "fulfilled_orders" (list[int]): List of fulfilled orders.
                - "unfulfilled_orders" (list[int]): List of unfulfilled orders.
                - "total_time_cost_per_production_line" (numpy.array): 1D array of the sum of production and transition times for each production line.
                - "feasible_orders_to_fulfill" (list): The feasible orders that can be fulfilled based on the current solution without delaying other planned orders.
                - "validation_single_production_schedule" (callable): def validation_single_production_schedule(production_line_id: int, production_schedule: list[int]) -> bool: function to check whether the production schedule is valid.
                - "get_time_cost_delta" (callable): def get_time_cost_delta(production_line_id: int, order_id: int, position: int, solution: Solution=None) -> float: function to get the time cost for following order after insert in this solution. solution can be omitted if using current solution.
                - "validation_solution" (callable): def validation_solution(solution: Solution) -> bool: function to check whether new solution is valid.
        """
        if solution is None:
            solution = self.current_solution

        # Go through the production schedule to get the production, transition time for each line and fulfillment time for order.
        production_time_cost = np.zeros(self.production_line_num)
        transition_time_cost = np.zeros(self.production_line_num)
        quantity_produced_per_product = np.zeros(self.product_num)
        fulfilled_orders = []
        order_fulfillment_times = [-1] * self.order_num
        for line_id, schedule in enumerate(solution.production_schedule):
            if schedule:
                schedule_np = np.array(schedule)
                fulfilled_orders.extend(schedule)

                products = self.order_product[schedule_np]
                quantities = self.order_quantity[schedule_np]
                production_times = quantities / self.production_rate[line_id, products]
                production_time_cost[line_id] = np.sum(production_times)

                if len(products) > 1:
                    transitions = self.transition_time[line_id, products[:-1], products[1:]]
                    transition_time_cost[line_id] = np.sum(transitions)

                end_times = np.cumsum(production_times + np.concatenate(([0], transitions))) if len(products) > 1 else production_times

                for i, order_id in enumerate(schedule):
                    order_fulfillment_times[order_id] = end_times[i]
                np.add.at(quantity_produced_per_product, products, quantities)

        # Calculate unfulfilled orders
        unfulfilled_orders = list(set(range(self.order_num)) - set(fulfilled_orders))

        # Calculate remaining quantity for each product
        quantity_remaining_per_product = np.zeros(self.product_num)
        for product in range(self.product_num):
            total_required_quantity = np.sum(self.order_quantity[self.order_product == product])
            quantity_remaining_per_product[product] = total_required_quantity - quantity_produced_per_product[product]

        total_time_cost = production_time_cost + transition_time_cost

        feasible_orders_to_fulfill = []

        for order_id in unfulfilled_orders:
            if order_id in feasible_orders_to_fulfill:
                continue
            product = self.order_product[order_id]
            quantity = self.order_quantity[order_id]
            deadline = self.order_deadline[order_id]

            for line_id in range(self.production_line_num):
                if self.production_rate[line_id, product] == 0:
                    continue
                current_schedule = solution.production_schedule[line_id]

                feasible = False
                for position in range(len(current_schedule) + 1):
                    # Calculate the delta time costs
                    if position == 0:
                        prev_product = None
                    else:
                        prev_product = self.order_product[current_schedule[position - 1]]

                    if position == len(current_schedule):
                        next_product = None
                    else:
                        next_product = self.order_product[current_schedule[position]]

                    production_time_for_order = quantity / self.production_rate[line_id, product]
                    transition_time_before = 0 if prev_product is None else self.transition_time[line_id, prev_product, product]
                    transition_time_after = 0 if next_product is None else self.transition_time[line_id, product, next_product]
                    original_transition_time = 0 if prev_product is None or next_product is None else self.transition_time[line_id, prev_product, next_product]

                    delta_time_cost = transition_time_before + production_time_for_order + transition_time_after - original_transition_time

                    # Check if the order can be fulfilled before its deadline
                    if position == 0:
                        previous_order_fulfillment_time = 0
                    else:
                        previous_order_fulfillment_time = order_fulfillment_times[current_schedule[position - 1]]

                    if previous_order_fulfillment_time + transition_time_before + production_time_for_order > deadline:
                        continue

                    # Check if all subsequent orders can still be fulfilled before their deadlines
                    feasible = True
                    for i in range(position, len(current_schedule)):
                        order_after = current_schedule[i]
                        new_fulfillment_time = order_fulfillment_times[order_after] + delta_time_cost
                        if new_fulfillment_time > self.order_deadline[order_after]:
                            feasible = False
                            break

                    if feasible:
                        feasible_orders_to_fulfill.append(order_id)
                        break
                if feasible:
                    break

        state_data_dict = {
            "current_solution": solution,
            "fulfilled_order_num": len(fulfilled_orders),
            "fulfilled_orders": fulfilled_orders,
            "unfulfilled_orders": unfulfilled_orders,
            "total_time_cost_per_production_line": total_time_cost,
            "feasible_orders_to_fulfill": feasible_orders_to_fulfill,
            "validation_single_production_schedule": self.validation_single_production_schedule,
            "get_time_cost_delta": self.get_time_cost_delta,
            "validation_solution": self.validation_solution
        }
        return state_data_dict

    def validation_single_production_schedule(self, line_id: int, production_schedule: list[int]) -> bool:
        if production_schedule == []:
            return True

        schedule_np = np.array(production_schedule)

        products = self.order_product[schedule_np]
        quantities = self.order_quantity[schedule_np]
        deadlines = self.order_deadline[schedule_np]

        if np.any(self.production_rate[line_id, products] <= 0):
            return False
        production_times = quantities / self.production_rate[line_id, products]

        if len(products) > 1:
            transitions = self.transition_time[line_id, products[:-1], products[1:]]

        end_times = np.cumsum(production_times + np.concatenate(([0], transitions))) if len(products) > 1 else production_times

        for i, order_id in enumerate(production_schedule):
            if end_times[i] > deadlines[i]:
                return False
        
        return True

    def get_time_cost_delta(self, production_line_id: int, order_id: int, position: int, solution: Solution=None) -> float:
        solution = self.current_solution if solution is None else solution
        line_schedule = solution.production_schedule[production_line_id]
        prev_product = None if position == 0 else self.order_product[line_schedule[position - 1]]
        next_product = None if position == len(line_schedule) or position == -1 else self.order_product[line_schedule[position]]
        quantity = self.order_quantity[order_id]
        product = self.order_product[order_id]
        production_time_for_order = quantity / self.production_rate[production_line_id, product]
        transition_time_before = 0 if prev_product is None else self.transition_time[production_line_id, prev_product, product]
        transition_time_after = 0 if next_product is None else self.transition_time[production_line_id, product, next_product]
        original_transition_time = 0 if prev_product is None or next_product is None else self.transition_time[production_line_id, prev_product, next_product]
        delta_time_cost = transition_time_before + production_time_for_order + transition_time_after - original_transition_time
        return delta_time_cost

    def validation_solution(self, solution: Solution=None) -> bool:
        """Check the validation of this solution in the following items:
            1. Order existence: Ensure that the order IDs in Solution's production_schedule are valid (i.e., < self.order_num).
            2. No duplicate fulfilled orders: Ensure that no order appears more than once in the production schedule.
            3. Deadline fulfillment: Ensure that all orders are fulfilled before their respective deadlines.
        """
        if solution is None:
            solution = self.current_solution

        if not isinstance(solution, Solution) or not isinstance(solution.production_schedule, list):
            return False
        all_orders = [order for sublist in solution.production_schedule for order in sublist]

        # Check order existence
        if not np.all(np.array(all_orders) < self.order_num):
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

    def get_observation(self) -> dict:
        return {
            "Total Time Cost For All Production Lines": sum(self.state_data["total_time_cost_per_production_line"]),
            "Feasible Orders To Fulfill Num": sum(self.state_data["feasible_orders_to_fulfill"]),
            "Fulfilled Order Num": self.state_data["fulfilled_order_num"],
        }

    def dump_result(self, dump_trajectory: bool=True, compress_trajectory: bool=False, result_file: str="result.txt") -> str:
        content_dict = {
            "production_line_num": self.production_line_num,
            "product_num": self.product_num,
            "order_num": self.order_num,
        }
        content = super().dump_result(content_dict=content_dict, dump_trajectory=dump_trajectory, compress_trajectory=compress_trajectory, result_file=result_file)
        return content
