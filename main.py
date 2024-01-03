import numpy as np
import matplotlib.pyplot as plt


# Parameters
num_clients = 100 # Number of clients
num_routers = 20   # Number of routers
coverage_radius = 200  # Radius of router coverage in units
area_size = (2000, 2000)  # Width and Height of the area in units
max_iter = 250 # Number of iterations
num_fireflies = 20  # Number of fireflies
alpha = 0.25  # Randomness strength
beta_0 = 1  # Attraction coefficient base
gamma = 0.0001  # Absorption coefficient [0,1]



scaled_r= None

# Function to calculate coverage

def calculate_coverage(firefly_position):
    coverage = 0
    for cp in client_positions:
        distance = np.sqrt(np.sum((firefly_position - cp) ** 2, axis=1))
        if np.any(distance < coverage_radius):  # Check if any router covers this client
            coverage += 1  # Count the client as covered
    return coverage

def calculate_connectivity(router_positions):

    connected_routers = set()
    
    # First, determine which routers are interconnected to form a network
    for i, rp_i in enumerate(router_positions):
        for j, rp_j in enumerate(router_positions):
            if i != j and np.linalg.norm(rp_i - rp_j) < coverage_radius:
                connected_routers.add(tuple(rp_i))
                connected_routers.add(tuple(rp_j))
    
    # Now, check client connectivity based on these connected routers
    connected_clients = set()
    for cp in client_positions:
        for cr in connected_routers:
            if np.linalg.norm(np.array(cr) - cp) < coverage_radius:
                connected_clients.add(tuple(cp))
                break

    return len(connected_clients)



# Calculating objective function
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


def calculate_firefly_distance(firefly1, firefly2):
    '''
    Calculate the Euclidean distance between the centroids of two fireflies (router configurations).
    '''
    # Calculate the centroid of each firefly
    centroid1 = np.mean(firefly1, axis=0)
    centroid2 = np.mean(firefly2, axis=0)


    # Calculate the Euclidean distance between the two centroids
    distance = np.linalg.norm(centroid1 - centroid2)

    return distance


def move_fireflies(fireflies, fitness, alpha, beta_0, gamma, area_size):

   # Move fireflies
    for i in range(num_fireflies):
        for j in range(num_fireflies):
            if fitness[i] < fitness[j]:  # Move firefly i towards j
               
               
                r=calculate_firefly_distance(fireflies[i],fireflies[j])
                scaled_r=r/13
              
                #print("r:",scaled_r)
                
                beta = beta_0 * np.exp(-gamma * scaled_r ** 2) 
                
                #print("beta:",beta)
                #random_vector = alpha * (np.random.rand() - 0.5)
                random_vector = alpha * (np.random.rand(fireflies[i].shape[0], fireflies[i].shape[1]) - 0.5)
                fireflies[i] += beta * (fireflies[j]-fireflies[i]) + random_vector
                fireflies[i] = np.clip(fireflies[i], 0, area_size[0])  # Keep within bounds
     
                return fireflies
             



optimal_solution = None

best_coverage = -np.inf
sum_coverage = 0

best_connectivity = -np.inf
sum_connectivity=0

best_fitness = -np.inf
sum_fitness = 0

final_coverage=0
final_connectivity=0
final_fitness=0

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
    alpha = np.random.rand()
    

    
    connectivity = np.array([calculate_connectivity(f) if f is not None else 0 for f in fireflies])
    coverage = np.array([calculate_coverage(f) if f is not None else 0 for f in fireflies])
    fitness = np.array([calculate_fitness(calculate_coverage(f), calculate_connectivity(f), num_clients, num_routers, gamma) for f in fireflies])
    
    # Handling cases where the coverage and connectivity values between all fireflies are equal
    if np.all(coverage == coverage[0]) and np.all(connectivity == connectivity[0]):
        break   
    
    
    
    
    # Move fireflies based on fitness
    fireflies = move_fireflies(fireflies, fitness, alpha, beta_0, gamma, area_size)
    
    
    
    print("alpha:" ,alpha)
    print("coverage:" ,coverage)
    print("connectivity:" ,connectivity)
    #print("\nfitness:" ,np.round(fitness,2))
    print("average coverage per iter:",np.sum(coverage)/num_fireflies)
    print("iteration:", iter)
    print("\n")
    
    
    final_coverage=np.max(coverage)
    final_connectivity=np.max(connectivity)
    final_fitness=calculate_fitness(final_coverage,final_connectivity,num_clients,num_routers,gamma)
    
        # Update best results
    if np.max(coverage) > best_coverage:
        best_coverage = np.max(coverage)   
    if np.max(connectivity) > best_connectivity:
        best_connectivity = np.max(connectivity)
    if np.max(fitness) > best_fitness:
        best_fitness = np.max(fitness)
        

    # Adding the highest value of each iteration to its array 
    coverage_solution.append(np.max(coverage))
    connectivity_solution.append(np.max(connectivity))
    fitness_solution.append(np.max(fitness))


# Calculating the best solution
optimal_solution = fireflies[np.argmax(final_fitness)]


# Print results

print("Maximum Clients Covered:", best_coverage)

print("Maximum Connectivity:", best_connectivity)

print("Maximum Fitness:", best_fitness)

print("\nFinal Coverage: ",final_coverage)
print("Final Connectivity: ",final_connectivity)
print("Final Fitness: ",final_fitness)

print("\nNumber of Iterations:", max_iter)

print("final solution: ", optimal_solution)
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
