import numpy as np
from config import coverage_radius

def calculate_coverage(firefly_position, client_positions):
    coverage = 0
    for cp in client_positions:
        distance = np.sqrt(np.sum((firefly_position - cp) ** 2, axis=1))
        if np.any(distance < coverage_radius):  # Check if any router covers this client
            coverage += 1
    return coverage

def calculate_connectivity(router_positions, client_positions):
    connected_routers = set()
    for i, rp_i in enumerate(router_positions):
        for j, rp_j in enumerate(router_positions):
            if i != j and np.linalg.norm(rp_i - rp_j) < coverage_radius:
                connected_routers.add(tuple(rp_i))
                connected_routers.add(tuple(rp_j))
    
    connected_clients = set()
    for cp in client_positions:
        for cr in connected_routers:
            if np.linalg.norm(np.array(cr) - cp) < coverage_radius:
                connected_clients.add(tuple(cp))
                break
    return len(connected_clients)
