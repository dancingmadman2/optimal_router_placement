import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_clients = 150  # Number of clients
num_routers = 20   # Number of routers
coverage_radius = 200  # Radius of router coverage in units
area_size = (2000, 2000)  # Width and Height of the area in units
max_iter = 100  # Number of iterations
num_fireflies = 50  # Number of fireflies
alpha = 0.5  # Randomness strength
beta_0 = 1  # Attraction coefficient base
gamma = 0.5  # Absorption coefficient [0,1]



# Function to calculate coverage
def calculate_coverage(firefly_position):
    coverage = 0
    for cp in client_positions:
        distance = np.sqrt(np.sum((firefly_position - cp) ** 2, axis=1))
        coverage += np.sum(distance < coverage_radius)
    return coverage

#  Function to calculate connectivity
def calculate_connectivity(router_positions):
    connected_clients = set()
    for cp in client_positions:
        for rp in router_positions:
            distance = np.sqrt(np.sum((rp - cp) ** 2))
            if distance < coverage_radius:
                connected_clients.add(tuple(cp))
                break
    return len(connected_clients)

def calculate_fitness(C, G, N, M, alpha):
    '''
    Parameters:
    C : Coverage
    G : Connectivity
    N : Clients
    M : Routers
    alpha : Parameter to balance the importance of coverage and connectivity (0 <= alpha <= 1).
    '''
    fitness = alpha * (C / N) + (1 - alpha) * (G / (N + M))
    return fitness

# Adjusting alpha value linearly to see different results
def adjust_alpha(iteration, max_iterations):
   
    start_alpha = 1.0
    end_alpha = 0.1
    return start_alpha - (start_alpha - end_alpha) * (iteration / max_iterations)

# Initialize fireflies (routers)
fireflies = np.random.rand(num_fireflies, num_routers, 2) * area_size

# Firefly Algorithm
best_solution = None
best_coverage = -np.inf
lowest_coverage = np.inf
sum_coverage = 0
best_connectivity = -np.inf
lowest_connectivity=np.inf
sum_connectivity=0

coverage_solution = []
connectivity_solution = []


    
for iter in range(max_iter):
    
 #  Generate random positions for clients
    client_positions = np.random.rand(num_clients, 2) * area_size

    # Adjust alpha linearly for each iter
    # alpha = adjust_alpha(iter, max_iter)

    # Adjust alpha nonlinearly between [0,1]
    # alpha = np.random.rand()
    
    
    connectivity = np.array([calculate_connectivity(f) for f in fireflies])
    coverage = np.array([calculate_coverage(f) for f in fireflies])
    
    print("alpha: ",alpha)
    print("coverage:" ,coverage)
    print("connectivity:" ,connectivity)
 

   

    # Move fireflies
    for i in range(num_fireflies):
        for j in range(num_fireflies):
            if coverage[i] < coverage[j]:  # Move firefly i towards j
                beta = beta_0 * np.exp(-gamma * np.linalg.norm(fireflies[i] - fireflies[j])**2)
                fireflies[i] += beta * (fireflies[j] - fireflies[i]) + alpha * (np.random.rand(num_routers, 2) - 0.5)
                fireflies[i] = np.clip(fireflies[i], 0, area_size[0])  # Keep within bounds
                
     # Update best solution
    if np.max(coverage) > best_coverage:
        best_coverage = np.max(coverage)
        best_solution = fireflies[np.argmax(coverage)]
        
        
       
    if np.max(connectivity) > best_connectivity:
        best_connectivity = np.max(connectivity)
        best_solution = fireflies[np.argmax(connectivity)]
        
  
    coverage_solution.append(np.max(coverage))
    connectivity_solution.append(np.max(connectivity))
   # print("solution",connectivity_solution)


# Calculating the average values
sum_coverage += np.sum(coverage_solution)
sum_connectivity+=np.sum(connectivity_solution)
average_coverage = sum_coverage / ( max_iter)
average_connectivity = sum_connectivity/(max_iter)

# Calculating lowest values
lowest_connectivity = min(lowest_connectivity, np.min(connectivity_solution))
lowest_coverage = min(lowest_coverage, np.min(coverage_solution))
    
# Calculate the fitness core of the best solution
fitness_score = calculate_fitness(best_coverage, best_connectivity, num_clients, num_routers, alpha)


# Print results
print("Maximum Clients Covered:", best_coverage)
print("Minimum Clients Covered:", lowest_coverage)
print("Average Clients Covered:", average_coverage)

print("\nMaximum Connectivity:", best_connectivity)
print("Lowest Connectivity:", lowest_connectivity)
print("Average Connectivity:", average_connectivity)

print("\nFitness:",fitness_score)
print("\nNumber of Iterations:", max_iter)


    
# Plotting best solution
plt.figure(figsize=(8, 8))
plt.scatter(client_positions[:, 0], client_positions[:, 1], color='blue', label='Clients')
plt.scatter(best_solution[:, 0], best_solution[:, 1], color='red', marker='*', s=200, label='Routers')

# Showing coverage radius of each router
"""
for router in best_solution:
    coverage_circle = plt.Circle((router[0], router[1]), coverage_radius, color='green', alpha=0.1, edgecolor='black')
    plt.gca().add_artist(coverage_circle) 
    """
plt.title('Optimal Router Positions with Firefly Algorithm')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.grid(True)
plt.show()
