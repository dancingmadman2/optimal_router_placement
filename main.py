import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_clients = 100 # Number of clients
num_routers = 20   # Number of routers
coverage_radius = 200  # Radius of router coverage in units
area_size = (2000, 2000)  # Width and Height of the area in units
max_iter = 100  # Number of iterations
num_fireflies = 15  # Number of fireflies
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



def calculate_fitness(C, G, N, M, gamma):
    '''
    Parameters:
    C : Coverage
    G : Connectivity
    N : Clients
    M : Routers
    gamma : Parameter to balance the importance of coverage and connectivity (0 <= alpha <= 1).
    '''
    fitness = gamma * (C / N) + (1 - gamma) * (G / (N + M))
    return fitness

# Adjusting alpha value linearly to see different results
def adjust_alpha(iteration, max_iterations):
   
    start_alpha = 1.0
    end_alpha = 0.1
    return start_alpha - (start_alpha - end_alpha) * (iteration / max_iterations)

def move_fireflies(fireflies, fitness, alpha, beta_0, gamma, area_size):
   # Move fireflies
    for i in range(num_fireflies):
        for j in range(num_fireflies):
            if fitness[i] < fitness[j]:  # Move firefly i towards j
                r = np.linalg.norm(fireflies[i] - fireflies[j])
                beta = beta_0 * np.exp(-gamma * r ** 2)
                random_step = alpha * (np.random.rand(num_routers, 2) - 0.5) * area_size
                fireflies[i] += beta * (fireflies[j] - fireflies[i]) + random_step
                fireflies[i] = np.clip(fireflies[i], 0, area_size[0])  # Keep within bounds
              
                
    return fireflies



optimal_solution = None

best_coverage = -np.inf

sum_coverage = 0

best_connectivity = -np.inf

sum_connectivity=0

best_fitness = -np.inf

sum_fitness = 0


coverage_solution = []
connectivity_solution = []
fitness_solution=[]


 # Initialize fireflies (routers)
fireflies = np.random.rand(num_fireflies, num_routers, 2) * area_size
    
 #  Generate random positions for clients
client_positions = np.random.rand(num_clients, 2) * area_size
    
for iter in range(max_iter):    
  
    
    # Adjust alpha linearly for each iter
    #alpha = adjust_alpha(iter, max_iter)

    # Adjust alpha nonlinearly between [0,1]
    # alpha = np.random.rand()
    
    connectivity = np.array([calculate_connectivity(f) for f in fireflies])
    coverage = np.array([calculate_coverage(f) for f in fireflies])
    fitness = np.array([calculate_fitness(calculate_coverage(f), calculate_connectivity(f), num_clients, num_routers, gamma) for f in fireflies])

  

    # Move fireflies based on fitness
    fireflies = move_fireflies(fireflies, fitness, alpha, beta_0, gamma, area_size)
    
    print("alpha:" ,alpha)
    print("coverage:" ,coverage)
    print("connectivity:" ,connectivity)
    # print("fitness:" ,fitness)
    print("average coverage:",np.sum(coverage)/num_fireflies)
    print("iteration:", iter)
    print("\n")
    
        # Update best solution
    if np.max(coverage) > best_coverage:
        best_coverage = np.max(coverage)   
    if np.max(connectivity) > best_connectivity:
        best_connectivity = np.max(connectivity)
    if np.max(fitness) > best_fitness:
        best_fitness = np.max(fitness)
        optimal_solution = fireflies[np.argmax(fitness)]
 
    '''
    # Move fireflies
    for i in range(num_fireflies):
        for j in range(num_fireflies):
            if coverage[i] < coverage[j]:  # Move firefly i towards j
                beta = beta_0 * np.exp(-gamma * np.linalg.norm(fireflies[i] - fireflies[j])**2)
                fireflies[i] = fireflies[i]+ beta * (fireflies[j] - fireflies[i]) + alpha * (np.random.rand() - 0.5) #(np.random.rand(num_routers, 2
                fireflies[i] = np.clip(fireflies[i], 0, area_size[0]) # Preventing fireflies from going out of bounds    
    
    connectivity = np.array([calculate_connectivity(f) for f in fireflies])
    coverage = np.array([calculate_coverage(f) for f in fireflies])
    fitness = np.array([calculate_fitness(calculate_coverage(f), calculate_connectivity(f), num_clients, num_routers, gamma) for f in fireflies])
    '''
    # Summing up all the values for later averaging
    sum_coverage += np.sum(coverage)
    sum_connectivity+=np.sum(connectivity)
    sum_fitness+=np.sum(fitness)
    

    # Adding the highest value of each iteration to its array
    coverage_solution.append(np.max(coverage))
    connectivity_solution.append(np.max(connectivity))
    fitness_solution.append(np.max(fitness))


# Calculating the average values

average_coverage = sum_coverage /  (num_fireflies * max_iter)
average_connectivity = sum_connectivity/(num_fireflies * max_iter)
average_fitness= sum_fitness/(num_fireflies * max_iter)


# Calculate the fitness core of the best solution
#fitness_score = calculate_fitness(best_coverage, best_connectivity, num_clients, num_routers, gamma)


# Print results

print("Maximum Clients Covered:", best_coverage)

print("Average Clients Covered:", average_coverage)

print("\nMaximum Connectivity:", best_connectivity)

print("Average Connectivity:", average_connectivity)

print("\nFitness:",average_fitness)


print("\nNumber of Iterations:", max_iter)

print("best solution: ", optimal_solution)
# Plotting best solution
plt.figure(figsize=(8, 8))
plt.scatter(client_positions[:, 0], client_positions[:, 1], color='blue', label='Clients')
plt.scatter(optimal_solution[:, 0], optimal_solution[:, 1], color='red', marker='*', s=200, label='Routers')


# Showing coverage radius of each router
for router in optimal_solution:
    coverage_circle = plt.Circle((router[0], router[1]), coverage_radius, color='green', alpha=0.1, edgecolor='black')
    plt.gca().add_artist(coverage_circle) 
  
plt.title('Optimal Router Positions with Firefly Algorithm')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.grid(True)
plt.show()
