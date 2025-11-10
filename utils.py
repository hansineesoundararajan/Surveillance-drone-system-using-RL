from collections import deque
import numpy as np

def is_valid_pos(x, y, grid_size):
    return 0 <= x < grid_size and 0 <= y < grid_size

def find_shortest_path_bfs(start_pos, goal_pos, grid_size, obstacle_locations):
    """
    Finds the shortest path from start_pos to goal_pos using BFS, avoiding obstacles.
    Returns a list of (x, y) coordinates representing the path, excluding the start.
    """
    queue = deque([start_pos])
    visited = {start_pos}
    parent = {start_pos: None}
    
    # Convert obstacle locations from index to (x, y) tuple for easy checking
    obstacles = {divmod(idx, grid_size) for idx in obstacle_locations}

    # BFS search
    while queue:
        current = queue.popleft()
        if current == goal_pos:
            break

        x, y = current
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dx, dy in directions:
            next_pos = (x + dx, y + dy)
            nx, ny = next_pos

            if is_valid_pos(nx, ny, grid_size) and next_pos not in visited and next_pos not in obstacles:
                visited.add(next_pos)
                parent[next_pos] = current
                queue.append(next_pos)
    
    # Reconstruct path
    path = deque()
    curr = goal_pos
    if curr not in parent:
        return []

    while parent[curr] is not None:
        path.appendleft(curr)
        curr = parent[curr]
        
    return list(path)