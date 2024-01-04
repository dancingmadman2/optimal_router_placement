

---

# Firefly Optimization for Wireless Router Placement

## Overview
This project implements a Firefly Optimization Algorithm for optimizing the placement of wireless mesh routers. The goal is to enhance network performance by optimizing client coverage and network connectivity. The algorithm is inspired by the behavior of fireflies and utilizes a bio-inspired approach to solve this NP-hard problem.

## Features
- **Optimal Router Placement:** Determines the best positions for mesh routers to optimize client coverage and network connectivity.
- **Bio-inspired Algorithm:** Utilizes the Firefly Optimization Algorithm, a nature-inspired metaheuristic.
- **Customizable Parameters:** Allows adjustment of network area, number of clients, routers, and algorithmic parameters like alpha, beta, and gamma.
- **Visual Representation:** Plots the final router positions and client coverage areas for visual analysis.

## Algorithm Parameters
- `num_clients`: Number of clients in the network
- `num_routers`: Number of routers to be placed
- `coverage_radius`: Radius of coverage for each router
- `area_size`: Size of the area (width, height)
- `max_iter`: Maximum number of iterations for the algorithm
- `num_fireflies`: Number of fireflies (solutions-configuration of routers) used in the algorithm
- `alpha`, `beta_0`, `gamma`: Algorithm-specific parameters controlling randomness, attraction, and absorption

## Functions
- `calculate_coverage`: Computes the coverage of clients by routers.
- `calculate_connectivity`: Calculates the connectivity between routers.
- `calculate_fitness`: Determines the fitness of a solution based on coverage and connectivity.
- `move_fireflies`: Updates the positions of fireflies (solutions) based on their attractiveness.

## Results
After running the simulation, the algorithm outputs:
- Maximum coverage, connectivity, and fitness scores achieved.
- Final solution for router placements.
- A plot illustrating the optimal placement of routers and coverage.


## Example Visual Illustration
![Figure_100_clients](https://github.com/dancingmadman2/cmp4503/assets/88443368/297e9578-f56d-4c7c-9329-93ece0dd6d83)

---
