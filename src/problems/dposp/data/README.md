# Data Folder

This data folder contains instances of the Dynamic Production Order Scheduling Problem (DPOSP) modeled after real production scenarios. Each instance represents a production environment with multiple machines, orders, and products, with the goal of maximizing the number of completed orders within their respective deadlines.

## Source

The DPOSP instances are derived from actual production scenarios and are designed to reflect the complexities of dynamic production scheduling in an industrial setting.

## Data Format

- Each instance is a folder named after the instance itself, containing three TSV (tab-separated values) files:

1. `production.tsv`: This file includes production rates for different products on various production lines.
   - `ProductionLine`: The identifier of the production line.
   - `Product`: The product being produced.
   - `ProductionRate`: The rate at which the product is produced (in units per hour).

2. `transition.tsv`: This file details the transition times between products on the same production line.
   - `ProductionLine`: The identifier of the production line.
   - `SourceProduct`: The product from which the transition starts.
   - `DestinationProduct`: The product to which the transition is made.
   - `TransitionTime`: The time required to switch from the source product to the destination product (in hours).

3. `order.tsv`: This file lists all orders that need to be fulfilled.
   - `Order`: The identifier of the order.
   - `Product`: The product required for the order.
   - `Quantity`: The amount of product required.
   - `Deadline`: The latest completion time for the order (in hours from a reference start time).

- In order to ensure the calculation accuracy of OR Solver, we ensure that the completion time, deadline, and product conversion time of each order are integers.

- The identifier starts from 0.

## Dataset

- The train_data is a compressed sample from the source, curated to facilitate heuristic evolution.
- The validation_data is a direct sample from the source, used to refine and filter the evolved heuristics.
- The test_data is a direct sample from the source, intended to evaluate the performance of the heuristics.
- The smoke_data dir is used for a quick check to see if the heuristic can run, where smoke_data in the dir is the test intance, and previous_actions are the precursor actions.

## Problem Definition

The DPOSP involves scheduling production orders on multiple machines with varying production rates and transition times between products. Some transitions may not be allowed. Each order specifies a product, quantity, and deadline, and it is only considered complete if all requirements are met on time. The objective is to maximize the number of completed orders, with each order having equal priority.
