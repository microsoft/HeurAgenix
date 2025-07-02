from src.problems.base.mdp_components import *
import numpy as np
from sklearn.linear_model import LinearRegression

def cost_minimization_charging_f94b(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Dynamic Charging Optimization for EV Fleet with Predictive Modeling.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): Number of electric vehicles in the fleet.
            - total_chargers (int): Maximum number of available chargers.
            - charging_price (list[float]): List of charging prices at each time step.
            - customer_arrivals (list[int]): List of customer arrivals at each time step.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_step (int): Index of the current time step.
            - operational_status (list[int]): Operational status of each EV (0: idle, 1: serving, 2: charging).
            - time_to_next_availability (list[int]): Lead time until each EV becomes available.
            - battery_soc (list[float]): Battery state of charge for each EV in percentage.
        algorithm_data (dict): Optional algorithm-specific data if necessary.
        get_state_data_function (callable): Function to receive the new solution and return the state dictionary for new solution.
        Introduction for hyper parameters in kwargs:
            - charge_lb (float, optional): Base lower bound of SoC for charging decisions, default is 0.25.
            - charge_ub (float, optional): Base upper bound of SoC for staying available, default is 0.4.
            - price_threshold (float, optional): Charging price threshold for prioritizing low SoC, default is 0.35.

    Returns:
        ActionOperator: Operator with actions for the EV fleet, ensuring actions comply with constraints.
        dict: Updated algorithm data or empty dictionary if no update.
    """

    # Set base values for hyper-parameters if not provided
    charge_lb = kwargs.get('charge_lb', 0.25)
    charge_ub = kwargs.get('charge_ub', 0.4)
    price_threshold = kwargs.get('price_threshold', 0.35)

    # Extract necessary data
    fleet_size = global_data['fleet_size']
    total_chargers = global_data['total_chargers']
    current_step = state_data['current_step']
    operational_status = state_data['operational_status']
    time_to_next_availability = state_data['time_to_next_availability']
    battery_soc = state_data['battery_soc']
    charging_price = global_data['charging_price']
    customer_arrivals = global_data['customer_arrivals']

    # Calculate fleet-to-charger ratio
    fleet_to_charger_ratio = fleet_size / total_chargers

    # Calculate average charging price and historical demand patterns
    avg_charging_price = np.mean(charging_price[:current_step+1])
    
    # Predict future demand using a sliding window approach
    window_size = min(5, current_step+1)
    X_demand = np.arange(max(0, current_step - window_size + 1), current_step + 1).reshape(-1, 1)
    y_demand = np.array(customer_arrivals[max(0, current_step - window_size + 1):current_step + 1])
    demand_model = LinearRegression().fit(X_demand, y_demand)
    predicted_demand = demand_model.predict(np.array([[current_step + 1]]))[0]

    # Adjust charge thresholds based on predicted demand fluctuations
    if predicted_demand < np.percentile(customer_arrivals, 25):
        charge_lb -= 0.05
        charge_ub += 0.05
    elif predicted_demand > np.percentile(customer_arrivals, 75):
        charge_lb += 0.05
        charge_ub -= 0.05

    # Refine tiered approach with more granular adjustments based on fleet-to-charger ratio
    if fleet_to_charger_ratio <= 10:
        charge_lb -= 0.05
        charge_ub += 0.05
    elif 11 <= fleet_to_charger_ratio <= 14:
        charge_lb -= 0.02
        charge_ub += 0.02
    elif 15 <= fleet_to_charger_ratio <= 17:
        charge_lb += 0.02
        charge_ub -= 0.02
    elif 18 <= fleet_to_charger_ratio <= 19:
        charge_lb += 0.04
        charge_ub -= 0.04
    elif fleet_to_charger_ratio >= 20:
        charge_lb += 0.06
        charge_ub -= 0.06

    # Initialize actions based on fleet size
    actions = [0] * fleet_size

    # Adaptive mechanism for low-demand periods
    if predicted_demand < np.percentile(customer_arrivals, 25):
        lowest_soc_indices = np.argsort(battery_soc)
        for i in lowest_soc_indices:
            if operational_status[i] == 0 and time_to_next_availability[i] == 0:
                actions[i] = 1
                if sum(actions) >= total_chargers:
                    break

    # Introduce mechanism for zero charging price scenario
    elif avg_charging_price == 0:
        for i in range(fleet_size):
            if operational_status[i] == 0 and time_to_next_availability[i] == 0:
                actions[i] = 1

    # Add logic to prioritize charging EVs with the lowest SoC when there is a price dip
    else:
        lowest_soc_index = np.argmin([soc if operational_status[i] == 0 and time_to_next_availability[i] == 0 else float('inf') for i, soc in enumerate(battery_soc)])
        actions[lowest_soc_index] = 1

    # Determine actions for each EV
    for i in range(fleet_size):
        # If EV is serving a trip, it must remain available
        if time_to_next_availability[i] >= 1:
            actions[i] = 0
        # Prioritize charging for EVs with low time_to_next_availability and battery_soc
        elif time_to_next_availability[i] == 0 and battery_soc[i] <= charge_lb:
            actions[i] = 1
        elif time_to_next_availability[i] == 0 and battery_soc[i] >= charge_ub:
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