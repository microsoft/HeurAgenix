from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def optimal_charge_fit_e6f4(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """OptimalChargeFit heuristic algorithm with advanced machine learning models and real-time adjustments.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "total_chargers" (int): The maximum number of available chargers.
            - "fleet_size" (int): The total number of EVs in the fleet.
            - "customer_arrivals" (list[int]): The number of customer arrivals at each time step.
            - "charging_price" (list[float]): The charging price at each time step.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): An instance of the Solution class representing the current solution.
            - "operational_status" (list[int]): List indicating the operational status of each EV (0: idle, 1: serving a trip, 2: charging).
            - "battery_soc" (list[float]): List of current state of charge (SoC) for each EV in percentage.
            - "time_to_next_availability" (list[int]): List of time remaining for each EV to become available.
        algorithm_data (dict): Can be omitted if not used.
        get_state_data_function (callable): Can be omitted if not used.
        kwargs: Hyper-parameters used to control the algorithm behavior, including model configurations.

    Returns:
        ActionOperator: The operator that assigns EVs to charging stations based on enhanced predictive prioritization logic.
        dict: An empty dictionary as no algorithm-specific data updates are needed.
    """

    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    customer_arrivals = global_data["customer_arrivals"]
    charging_price = global_data["charging_price"]
    operational_status = state_data["operational_status"]
    battery_soc = state_data["battery_soc"]
    time_to_next_availability = state_data["time_to_next_availability"]
    current_step = state_data["current_step"]

    # Advanced machine learning model to predict future demand and pricing
    if current_step > 0:
        X = np.arange(current_step).reshape(-1, 1)
        y_demand = np.array(customer_arrivals[:current_step])
        y_price = np.array(charging_price[:current_step])

        demand_model = RandomForestRegressor(n_estimators=100).fit(X, y_demand)
        price_model = RandomForestRegressor(n_estimators=100).fit(X, y_price)

        predicted_demand = demand_model.predict(np.array(current_step).reshape(-1, 1))[0]
        predicted_price = price_model.predict(np.array(current_step).reshape(-1, 1))[0]
    else:
        predicted_demand = customer_arrivals[current_step]
        predicted_price = charging_price[current_step]

    # Dynamic priority threshold based on predictions and real-time adjustments
    average_soc = np.mean(battery_soc)
    historical_data_factor = np.std(customer_arrivals[:current_step]) / np.mean(customer_arrivals) if current_step > 0 else 1
    predictive_factor = predicted_price / np.mean(charging_price) if current_step > 0 else 1
    priority_threshold = kwargs.get("priority_threshold", average_soc * historical_data_factor * predictive_factor)

    # Initialize actions with zeros
    actions = [0] * fleet_size

    # Enhanced prioritization conditions with predictive analytics
    if (predicted_demand >= np.mean(customer_arrivals) * historical_data_factor or
        predicted_price <= np.mean(charging_price) * predictive_factor):
        
        # Sort EVs based on their SoC, prioritizing those with lower SoC for charging
        ev_indices = sorted(range(fleet_size), key=lambda i: (battery_soc[i] < priority_threshold, battery_soc[i]))
        
        chargers_used = 0
        
        for i in ev_indices:
            # Prioritize charging EVs with low SoC that are available
            if operational_status[i] == 0 and time_to_next_availability[i] == 0:
                if chargers_used < total_chargers:
                    actions[i] = 1  # Assign to charge
                    chargers_used += 1
            # Ensure EVs in service remain available
            if operational_status[i] == 1:
                actions[i] = 0

    # Create and return the ActionOperator
    operator = ActionOperator(actions)
    
    return operator, {}