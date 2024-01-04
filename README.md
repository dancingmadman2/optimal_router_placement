

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

### Distance Calculation
The distance between any two fireflies i and j is calculated using the Euclidean distance formula. This is represented as:

$$ r_{ij} = \sqrt{\sum_{k=1}^{n}(x_{ik} - x_{jk})^2} $$

where $(r_ij)$ is the distance between firefly i and firefly j, $(x_i)$ and $(x_j)$ are the coordinates of fireflies i and j respectively, and $(n)$ is the dimensionality of the problem space.

### Attractiveness Calculation
The attractiveness of a firefly is directly proportional to its light intensity, which in turn is associated with the objective function value of the solution it represents. However, the attractiveness decreases with increasing distance due to light absorption. The attractiveness $(β)$ of a firefly at a distance $(r)$ is calculated using the formula:

$$ \beta(r) = \beta_0 e^{-\gamma r^2} $$

Here, $(β_0)$ is the attractiveness at $(r = 0)$ meaning at distance 0, and $(γ)$ is the light absorption coefficient. The value $(β_0)$ represents the base attractiveness, and the term $(e^{-\gamma r^2})$
 signifies the decrease in attractiveness with increasing distance due to absorption. The parameter $(γ)$ controls how quickly the attractiveness decreases with distance; a higher $(γ)$ means faster decrease.

### Firefly Movement
The movement of a firefly i towards a more attractive firefly j is influenced by the attractiveness and is given by:

$$ x_i = x_i + \beta_0 e^{-\gamma r_{ij}^2} (x_j - x_i) + \alpha (\text{rand} - 0.5) $$

Here, $x_i$ and $x_j$ are the positions of fireflies i and j, and $α$ represents the randomness parameter. The term $\text{rand} - 0.5$
 introduces a randomization factor to the movement, allowing the fireflies to explore the search space beyond immediate attractiveness gradients.


Certainly! Continuing from where we left off, let's add the objective function used in the Firefly Optimization Algorithm:

### Objective Function
The objective function in the Firefly Algorithm is crucial as it determines the light intensity or brightness of each firefly, which in turn influences the attractiveness. In the context of the router placement problem, the objective function evaluates the quality of a router configuration in terms of coverage and connectivity. It is typically formulated to maximize these parameters. The objective function can be represented as follows:

$$ F(X) = \gamma \times \frac{C(X)}{N} + (1 - \gamma) \times \frac{G(X)}{N + M} $$

where:
- $F$ is the fitness value or the objective function.
- $C$ represents the coverage, i.e., the number of clients covered by the routers.
- $G$ is the connectivity, indicating how well the routers are connected to each other.
- $N$ is the total number of clients.
- $M$ is the total number of routers.
- $γ$ is a parameter that balances the importance of coverage and connectivity in the solution. It ranges from 0 to 1.

This objective function combines coverage and connectivity to assess the overall network performance. The term $\( \frac{C}{N} \)$ calculates the proportion of clients covered, while $\( \frac{G}{N + M} \)$ assesses the connectivity relative to the total number of network nodes (clients and routers). The parameter $(γ)$ allows for adjusting the relative importance of coverage and connectivity in the fitness evaluation.

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
