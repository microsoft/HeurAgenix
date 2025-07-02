from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np
from sklearn.linear_model import LinearRegression

def cost_minimization_charging_d684(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Dynamic Charging Optimization for EV Fleet with Real-Time Adjustments Based on Demand and SoC.

    Args:
        global_data (dict): Contains global data necessary for the algorithm.
            - fleet_size (int): Number of electric vehicles in the fleet.
            - total_chargers (int): Maximum number of available chargers.
            - customer_arrivals (list[int]): List of customer arrivals at each time step.
            - charging_price (list[float]): List of charging prices at each time step.
        state_data (dict): Contains the current state information.
            - current_step (int): Index of the current time step.
            - operational_status (list[int]): Operational status of each EV (0: idle, 1: serving, 2: charging).
            - time_to_next_availability (list[int]): Lead time until each EV becomes available.
            - battery_soc (list[float]): Battery state of charge for each EV in percentage.
        get_state_data_function (callable): Function to receive the new solution and return the state dictionary for new solution.

    Returns:
        ActionOperator: Operator with actions for the EV fleet, ensuring actions comply with constraints.
        dict: Updated algorithm data or empty dictionary if no update.
    """

    # Extract necessary data
    fleet_size = global_data['fleet_size']
    total_chargers = global_data['total_chargers']
    current_step = state_data['current_step']
    operational_status = state_data['operational_status']
    time_to_next_availability = state_data['time_to_next_availability']
    battery_soc = state_data['battery_soc']
    customer_arrivals = global_data['customer_arrivals']
    charging_price = global_data['charging_price']

    # Prepare data for the predictive model
    X_demand = np.arange(max(0, current_step - 5), current_step + 1).reshape(-1, 1)
    y_demand = np.array(customer_arrivals[max(0, current_step - 5):current_step + 1])

    # Train linear regression model for demand
    demand_model = LinearRegression().fit(X_demand, y_demand)

    # Predict future trends
    future_demand = demand_model.predict(np.array([[current_step + 1]]))[0]

    # Calculate dynamic SoC thresholds based on real-time demand and SoC distribution
    average_soc = np.mean(battery_soc)
    base_charge_lb = max(0.3, average_soc - 0.1 * (future_demand / np.max(customer_arrivals)))
    base_charge_ub = min(0.6, average_soc + 0.1 * (np.max(customer_arrivals) - future_demand) / np.max(customer_arrivals))

    # Initialize actions based on fleet size
    actions = [0] * fleet_size

    # Determine actions for each EV
    for i in range(fleet_size):
        # If EV is serving a trip, it must remain available
        if time_to_next_availability[i] >= 1:
            actions[i] = 0
        # Prioritize charging based on dynamic SoC thresholds and future demand
        elif time_to_next_availability[i] == 0 and battery_soc[i] <= base_charge_lb:
            actions[i] = 1
        elif time_to_next_availability[i] == 0 and battery_soc[i] >= base_charge_ub:
            actions[i] = 0

    # Ensure the sum of actions does not exceed the total number of chargers
    if sum(actions) > total_chargers:
        excess_count = sum(actions) - total_chargers
        charge_indices = [index for index, action in enumerate(actions) if action == 1]
        np.random.shuffle(charge_indices)
        for index in charge_indices[:excess_count]:
            actions[index] = 0

    # Create a new solution with the determined actions
    new_solution = Solution([actions])

    # Return the operator and empty algorithm data
    return ActionOperator(actions), {}