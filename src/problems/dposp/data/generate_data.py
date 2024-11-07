import os
import random
from math import gcd  
from functools import reduce 

def lcm(a, b):  
    return a * b // gcd(a, b)  

def lcm_for_list(numbers):  
    return reduce(lcm, numbers, 1)  

def build_data(
    production_line_num: int,
    product_num: int,
    order_num: int,
    production_distribution: dict,
    production_rate_distribution: dict,
    transition_distribution: dict,
    order_quantity_rate_distribution: dict,
    order_deadline_distribution: dict,
    output_dir: str
):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Helper function to select an item based on distribution
    def select_based_on_distribution(distribution):
        population, weights = zip(*distribution.items())
        return random.choices(population, weights, k=1)[0]

    # Generate production lines and products
    production_lines = [i for i in range(production_line_num)]
    products = [i for i in range(product_num)]
    orders = [i for i in range(order_num)]

    production_data = [["ProductionLine", "Product", "ProductionRate"]]
    transition_data = [["ProductionLine", "SourceProduct", "DestinationProduct", "TransitionTime"]]
    product_production_rates = {product: [] for product in products}
    for line in production_lines:
        # Build production data
        supported_product_count = select_based_on_distribution(production_distribution)
        supported_products = random.sample(products, supported_product_count)
        for product in supported_products:
            production_rate = select_based_on_distribution(production_rate_distribution)
            production_data.append([line, product, production_rate])
            product_production_rates[product].append(production_rate)

        # Build transition data
        for source_product in supported_products:
            for destination_product in supported_products:
                if source_product != destination_product:
                    transition_time = select_based_on_distribution(transition_distribution)
                    # Convert -1 to "Forbidden"
                    transition_time_str = "Forbidden" if transition_time == -1 else transition_time
                    transition_data.append([line, source_product, destination_product, transition_time_str])

    # Build order data
    order_data = [["Order", "Product", "Quantity", "Deadline"]]
    for order in orders:
        product = random.choice(products)
        product_lcm = lcm_for_list(product_production_rates[product])  
        # Select a quantity rate based on distribution  
        quantity_rate = select_based_on_distribution(order_quantity_rate_distribution)  
        # Calculate the final order quantity  
        quantity = product_lcm * quantity_rate  
        deadline = select_based_on_distribution(order_deadline_distribution)
        order_data.append([order, product, quantity, deadline])

    # Write data to TSV files
    with open(os.path.join(output_dir, "production.tsv"), "w") as f:
        for row in production_data:
            f.write("\t".join(map(str, row)) + "\n")

    with open(os.path.join(output_dir, "transition.tsv"), "w") as f:
        for row in transition_data:
            f.write("\t".join(map(str, row)) + "\n")

    with open(os.path.join(output_dir, "order.tsv"), "w") as f:
        for row in order_data:
            f.write("\t".join(map(str, row)) + "\n")


if __name__ == "__main__":
    output_dir = os.path.join("src", "problems", "dposp", "data", "test_data", "case_1")
    production_line_num=5
    product_num=10
    order_num=10
    production_distribution={1: 0.5, 2: 0.25, 3: 0.25}
    production_rate_distribution={1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}
    transition_distribution={-1: 0.25, 0: 0.25, 1: 0.25, 2: 0.25}
    order_quantity_rate_distribution={1: 0.5, 2: 0.5}
    order_deadline_distribution={12: 0.5, 24: 0.5}
    build_data(
        production_line_num=production_line_num,
        product_num=product_num,
        order_num=order_num,
        production_distribution=production_distribution,
        production_rate_distribution=production_rate_distribution,
        transition_distribution=transition_distribution,
        order_quantity_rate_distribution=order_quantity_rate_distribution,
        order_deadline_distribution=order_deadline_distribution,
        output_dir=output_dir
    )
    print(f"Generated to {output_dir}")