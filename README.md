

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

The Firefly Optimization Algorithm (FA), as applied in the paper "Placement optimization of wireless mesh routers using the firefly optimization algorithm", is inspired by the flashing and communication behavior of fireflies. In this algorithm, the attractiveness between fireflies plays a crucial role. Let's explore the formulas and explanations for attractiveness and distance as presented in the paper:

### Distance Calculation
The distance between any two fireflies i and j is calculated using the Euclidean distance formula. This is represented as:

$$ r_{ij} = \sqrt{\sum_{k=1}^{n}(x_{ik} - x_{jk})^2} $$

where $r_ij$ is the distance between firefly i and firefly j, $x_i$ and $x_j$ are the coordinates of fireflies i and j respectively, and n is the dimensionality of the problem space.

### Attractiveness Calculation
The attractiveness of a firefly is directly proportional to its light intensity, which in turn is associated with the objective function value of the solution it represents. However, the attractiveness decreases with increasing distance due to light absorption. The attractiveness beta of a firefly at a distance $r$ is calculated using the formula:

$$ \beta(r) = \beta_0 e^{-\gamma r^2} $$

Here, $beta_0$ is the attractiveness at $r = 0$ meaning at distance 0, and $γ$ is the light absorption coefficient. The value $beta_0$ represents the base attractiveness, and the term $e^{-\gamma r^2}$
 signifies the decrease in attractiveness with increasing distance due to absorption. The parameter $gamma$ controls how quickly the attractiveness decreases with distance; a higher gamma means faster decrease.

### Firefly Movement
The movement of a firefly i towards a more attractive firefly j is influenced by the attractiveness and is given by:

$$ x_i = x_i + \beta_0 e^{-\gamma r_{ij}^2} (x_j - x_i) + \alpha (\text{rand} - 0.5) $$

Here, x_i and x_j are the positions of fireflies i and j, and alpha represents the randomness parameter. The term $\text{rand} - 0.5$
 introduces a randomization factor to the movement, allowing the fireflies to explore the search space beyond immediate attractiveness gradients.

### Interpretation in the Context of Router Placement
In the context of the router placement problem, each firefly represents a potential solution, i.e., a configuration of router positions. The light intensity of a firefly corresponds to the quality of this configuration, measured in terms of network coverage and connectivity. The attractiveness mechanism drives the fireflies (solutions) to move towards better solutions, thereby exploring the search space for an optimal router placement configuration.

## Results
After running the simulation, the algorithm outputs:
- Maximum coverage, connectivity, and fitness scores achieved.
- Final solution for router placements.
- A plot illustrating the optimal placement of routers and coverage.


## Example Visual Illustration
![Figure_100_clients](https://github.com/dancingmadman2/cmp4503/assets/88443368/297e9578-f56d-4c7c-9329-93ece0dd6d83)

---
