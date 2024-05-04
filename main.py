import numpy as np
import matplotlib.pyplot as plt
from config import *
from utils import calculate_coverage, calculate_connectivity
from firefly_algorithm import move_fireflies, calculate_fitness

client_positions = np.random.rand(num_clients, 2) * area_size
fireflies = np.random.rand(num_fireflies, num_routers, 2) * area_size

def main():
    best_coverage = -np.inf
    best_connectivity = -np.inf
    best_fitness = -np.inf
    
    for iter in range(max_iter):
        connectivity = np.array([calculate_connectivity(f, client_positions) for f in fireflies])
        coverage = np.array([calculate_coverage(f, client_positions) for f in fireflies])
        fitness = np.array([calculate_fitness(cov, conn, num_clients, num_routers, gamma) for cov, conn in zip(coverage, connectivity)])

        fireflies = move_fireflies(fireflies, fitness, alpha, beta_0, gamma, area_size)

        # Optional: Print intermediate results
        # print(f"Iteration {iter}, Best Fitness: {np.max(fitness)}")

    best_index = np.argmax(fitness)
    print("Optimal Router Configuration:", fireflies[best_index])
    plot_results(client_positions, fireflies[best_index])

def plot_results(client_positions, optimal_solution):
    plt.figure(figsize=(8, 8))
    plt.scatter(client_positions[:, 0], client_positions[:, 1], color='blue', label='Clients')
    plt.scatter(optimal_solution[:, 0], optimal_solution[:, 1], color='red', marker='*', s=200, label='Routers')
    plt.title('Optimal Router Positions with Firefly Algorithm')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
