import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

def draw_grid(env, total_reward=0):
    grid_size = env.grid_size
    grid = np.zeros((grid_size, grid_size))
    
    # 1. Background elements
    for x, y in env.visited:
        grid[x, y] = 1 # Light Gray (Visited)
    for idx in env.obstacle_locations:
        x, y = divmod(idx, grid_size)
        grid[x, y] = 4 # Black (Obstacle)
    for idx in env.charger_locations:
        x, y = divmod(idx, grid_size)
        grid[x, y] = 5 # Green (Charger)
    if (0, 0) not in env.obstacle_locations:
        grid[0, 0] = 6 # Dark Green (Base)
        
    # 2. Dynamic/Critical elements
    for idx in env.intrusion_locations:
        x, y = divmod(idx, grid_size)
        grid[x, y] = 2 # Red (Intrusion)
        
    # 3. Drone Position
    x, y = env.drone_pos
    if env.battery <= env.battery_critical and (x,y) != (0,0):
        grid[x, y] = 7 # Orange/Yellow (Critical Battery)
    else:
        grid[x, y] = 3 # Blue (Normal)

    # Set up colormap
    cmap = colors.ListedColormap([
        'white',     # 0: empty
        'lightgray', # 1: visited
        'red',       # 2: intrusion
        'blue',      # 3: drone (Normal)
        'black',     # 4: obstacle
        'green',     # 5: secondary charger
        'darkgreen', # 6: base charger
        'orange'     # 7: drone (Critical battery)
    ])
    bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    plt.clf()
    plt.imshow(grid, cmap=cmap, norm=norm)
    plt.xticks(np.arange(grid_size))
    plt.yticks(np.arange(grid_size))
    plt.grid(True, color='gray', linestyle='-', linewidth=0.5)

    # Title with key information
    plt.title(
        f"Drone: {env.drone_pos} | Battery: {env.battery}% | Reward: {total_reward:.2f}", 
        fontsize=12, 
        color='black'
    )

    # Status box
    status_text = (
        f"STATUS: {'AUTO-RETURNING ' if env.auto_returning else 'EXPLORING'}\n"
        f"Cells Visited: {len(env.visited)}/{env.grid_size*env.grid_size - len(env.obstacle_locations)}\n"
        f"Intrusions Left: {len(env.intrusion_locations)}\n"
    )

    # UPDATED REWARD LEGEND 
    reward_text = (
        "REWARD LEGEND:\n"
        "New cell visited: +500.0\n"
        "Detect intrusion: +500.0\n"
        "Revisit cell: 0.0\n"
        "Hit obstacle/Hover: -0.01\n"
        "Stuck Penalty (Scaled): -50.0 *\n"
        "Battery depleted: -50.0"
    )

    plt.subplots_adjust(right=0.75, top=0.85)
    plt.gcf().text(0.78, 0.8, status_text, fontsize=10, va='top', ha='left', bbox=dict(facecolor='lightblue', alpha=0.6, boxstyle='round,pad=0.5'))
    plt.gcf().text(0.78, 0.5, reward_text, fontsize=9, va='center', ha='left', bbox=dict(facecolor='lightgray', alpha=0.5, boxstyle='round,pad=0.5'))
    plt.draw()