import numpy as np
from config import *
from utils import calculate_coverage, calculate_connectivity

def calculate_firefly_distance(firefly1, firefly2):
    centroid1 = np.mean(firefly1, axis=0)
    centroid2 = np.mean(firefly2, axis=0)
    return np.linalg.norm(centroid1 - centroid2)

def move_fireflies(fireflies, fitness, alpha, beta_0, gamma, area_size):
    for i in range(num_fireflies):
        for j in range(num_fireflies):
            if fitness[i] < fitness[j]:
                r = calculate_firefly_distance(fireflies[i], fireflies[j])
                scaled_r = r / 13
                beta = beta_0 * np.exp(-gamma * scaled_r ** 2)
                random_vector = alpha * (np.random.rand(fireflies[i].shape[0], fireflies[i].shape[1]) - 0.5)
                fireflies[i] += beta * (fireflies[j] - fireflies[i]) + random_vector
                fireflies[i] = np.clip(fireflies[i], 0, area_size[0])  # Keep within bounds
    return fireflies

def calculate_fitness(C, G, N, M, gamma):
        '''
    Parameters:
    C : Coverage
    G : Connectivity
    N : Clients
    M : Routers
    gamma : Parameter to balance the importance of coverage and connectivity (0 < gamma <= 1).
    '''
        return gamma * (C / N) + (1 - gamma) * (G / (N + M))
