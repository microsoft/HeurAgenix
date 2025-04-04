from src.problems.base.mdp_components import Solution, ActionOperator
from sklearn.linear_model import LinearRegression
import numpy as np

def balanced_fleet_management_6168(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Heuristic algorithm for EV Fleet Charging Optimization with dynamic prioritization and machine learning prediction.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): Number of electric vehicles in the fleet.
            - total_chargers (int): Maximum number of available chargers.
            - customer_arrivals (list[int]): List of customer arrivals at each time step.
            - order_price (list[float]): List of payment received per minute when a vehicle is on a ride.
            - charging_price (list[float]): List of charging prices at each time step.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_solution (Solution): Current action trajectory for EVs.
            - current_step (int): Index of the current time step.
            - operational_status (list[int]): Operational status of each EV (0: idle, 1: serving, 2: charging).
            - time_to_next_availability (list[int]): Lead time until each EV becomes available.
            - battery_soc (list[float]): Battery state of charge for each EV in percentage.
        (Optional) algorithm_data (dict): The algorithm dictionary for current algorithm only. In this algorithm, the following items are necessary:
            - demand_model (LinearRegression): Predictive model for future demand.
            - price_model (LinearRegression): Predictive model for future price.
        (Optional) get_state_data_function (callable): The function receives the new solution as input and return the state dictionary for new solution.
        (Optional) kwargs: Hyper-parameters for controlling algorithm behavior.
            - charge_lb (float, optional): Base lower bound of SoC for charging decisions. Default is 0.3.
            - charge_ub (float, optional): Base upper bound of SoC for staying available. Default is 0.4.
            - fleet_to_charger_ratio_threshold (float, optional): Threshold for fleet-to-charger ratio to apply prioritization logic. Default is 5.0.

    Returns:
        An ActionOperator that assigns charging actions to EVs based on dynamic thresholds and predicted demand.
        Updated algorithm data containing any new models or parameters.
    """
    # Set default hyper-parameters if not provided
    charge_lb = kwargs.get('charge_lb', 0.3)
    charge_ub = kwargs.get('charge_ub', 0.4)
    fleet_to_charger_ratio_threshold = kwargs.get('fleet_to_charger_ratio_threshold', 5.0)

    # Extract necessary data
    fleet_size = global_data['fleet_size']
    total_chargers = global_data['total_chargers']
    current_step = state_data['current_step']
    operational_status = state_data['operational_status']
    time_to_next_availability = state_data['time_to_next_availability']
    battery_soc = state_data['battery_soc']
    customer_arrivals = global_data['customer_arrivals']
    order_price = global_data['order_price']
    charging_price = global_data['charging_price']

    # Initialize actions with zeros (remain available)
    actions = [0] * fleet_size

    # Calculate fleet-to-charger ratio
    fleet_to_charger_ratio = fleet_size / total_chargers

    # Train or retrieve predictive models
    if 'demand_model' not in algorithm_data:
        X_demand = np.arange(max(0, current_step - 5), current_step + 1).reshape(-1, 1)
        y_demand = np.array(customer_arrivals[max(0, current_step - 5):current_step + 1])
        demand_model = LinearRegression().fit(X_demand, y_demand)
    else:
        demand_model = algorithm_data['demand_model']

    if 'price_model' not in algorithm_data:
        X_price = np.arange(max(0, current_step - 5), current_step + 1).reshape(-1, 1)
        y_price = np.array(charging_price[max(0, current_step - 5):current_step + 1])
        price_model = LinearRegression().fit(X_price, y_price)
    else:
        price_model = algorithm_data['price_model']

    # Predict future trends
    future_demand = demand_model.predict(np.array([[current_step + 1]]))[0] if current_step + 1 < len(customer_arrivals) else np.mean(customer_arrivals)
    future_price = price_model.predict(np.array([[current_step + 1]]))[0] if current_step + 1 < len(charging_price) else np.mean(charging_price)

    # Adjust thresholds based on predicted trends
    if fleet_to_charger_ratio > fleet_to_charger_ratio_threshold:
        charge_lb = max(charge_lb, 0.35 if future_demand > np.mean(customer_arrivals) and np.min(battery_soc) < 0.35 else charge_lb)
    
    # Determine actions for each EV
    for i in range(fleet_size):
        # EVs on a ride must remain available
        if time_to_next_availability[i] >= 1:
            actions[i] = 0
        # Prioritize charging for idle EVs with low battery SoC during high demand
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

    # Return the operator and updated algorithm data
    return ActionOperator(actions), {"demand_model": demand_model, "price_model": price_model}