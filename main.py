import numpy as np
import matplotlib.pyplot as plt

# Adjusted parameters based on the document
num_clients = 100  # Number of clients
num_routers = 20   # Number of routers
coverage_radius = 200  # Radius of router coverage in units
area_size = (2000, 2000)  # Width and Height of the area in units
max_iter = 1000  # Number of iterations
num_fireflies = 10  # Number of fireflies
alpha = 0.5  # Randomness strength
beta_0 = 1  # Attraction coefficient base
gamma = 1  # Absorption coefficient

# Generate random positions for clients
client_positions = np.random.rand(num_clients, 2) * area_size

# Function to calculate coverage
def calculate_coverage(firefly_position):
    coverage = 0
    for cp in client_positions:
        distance = np.sqrt(np.sum((firefly_position - cp) ** 2, axis=1))
        coverage += np.sum(distance < coverage_radius)
    return coverage

# Initialize fireflies (routers)
fireflies = np.random.rand(num_fireflies, num_routers, 2) * area_size

# Variables to keep track of the highest, lowest, and sum of fitness values
lowest_fitness = np.inf  # Initialize to maximum possible value
sum_fitness = 0

# Firefly Algorithm
best_solution = None
best_fitness = -np.inf

for iter in range(max_iter):
    fitness = np.array([calculate_coverage(f) for f in fireflies])

     # Update highest and lowest fitness
    lowest_fitness = min(lowest_fitness, np.min(fitness))
    sum_fitness += np.sum(fitness)

    # Update best solution
    if np.max(fitness) > best_fitness:
        best_fitness = np.max(fitness)
        best_solution = fireflies[np.argmax(fitness)]

    # Move fireflies
    for i in range(num_fireflies):
        for j in range(num_fireflies):
            if fitness[i] < fitness[j]:  # Move firefly i towards j
                beta = beta_0 * np.exp(-gamma * np.linalg.norm(fireflies[i] - fireflies[j])**2)
                fireflies[i] += beta * (fireflies[j] - fireflies[i]) + alpha * (np.random.rand(num_routers, 2) - 0.5)
                fireflies[i] = np.clip(fireflies[i], 0, area_size[0])  # Keep within bounds

# Calculating the average fitness
average_fitness = sum_fitness / (num_fireflies * max_iter)

# Print results
print("Highest Fitness (Maximum Clients Covered):", best_fitness)
print("Lowest Fitness (Minimum Clients Covered):", lowest_fitness)
print("Average Fitness (Average Clients Covered):", average_fitness)
print("Number of Iterations:",max_iter)

# Plotting the best solution for representing this algorithm
plt.figure(figsize=(8, 8))
plt.scatter(client_positions[:, 0], client_positions[:, 1], color='blue', label='Clients')
plt.scatter(best_solution[:, 0], best_solution[:, 1], color='red', marker='*', s=200, label='Routers')
plt.title('Optimal Router Positions with Firefly Algorithm')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.grid(True)
plt.show()
