# Surveillance Drone using Reinforcement Learning

## Overview
This project implements an **Autonomous Surveillance Drone Navigation System** using **Q-Learning** — a Reinforcement Learning (RL) algorithm that enables the drone to learn from experience without a pre-built model.  
The system trains a drone agent to efficiently explore a grid-based environment, detect intrusions, avoid obstacles, and manage its energy (battery) for sustained operation.

## Core Idea
The drone agent learns optimal navigation policies through interaction with its environment.  
Using an **ε-greedy exploration strategy** combined with a **Breadth-First Search (BFS) heuristic**, the drone efficiently locates intrusions while ensuring safe returns to its base when the battery is low.

## System Architecture
The project is modularized into the following components:

| Module | Description |
|--------|-------------|
| **Environment (`environment.py`)** | Simulates an 8×8 grid world containing obstacles, intrusion points, and charging stations. Handles rewards, penalties, and state transitions. |
| **Q-Learning Agent (`q_learning.py`)** | Implements the Q-Learning algorithm that updates a Q-table to learn optimal navigation policies. |
| **Drone Controller (`drone_controller.py`)** | Interprets agent outputs, applies learned actions, manages stuck detection and auto-return mechanisms. |
| **Utility Functions (`utils.py`)** | Includes BFS-based shortest pathfinding and helper functions for grid validation. |
| **Visualization (`visualization.py`)** | Uses Matplotlib to visualize environment, drone paths, and learning progress. |
| **Graphical Interface (`gui.py`)** | Tkinter-based GUI to visualize drone operations and control simulation. |
| **Main (`main.py`)** | Orchestrates training, evaluation, and visualization across modules. |

## Algorithm Used
**Q-Learning Algorithm**

The Q-values are updated using the Bellman Equation:

\[
Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]
\]

- **State (s):** Drone’s position, nearby obstacles, and battery level  
- **Action (a):** Move (up/down/left/right), detect intrusion, return to base  
- **Reward (r):** Based on exploration success, intrusion detection, and efficient energy use  

## Key Features
- Autonomous drone navigation in a simulated grid environment  
- Obstacle detection and avoidance  
- Intrusion detection and tracking  
- Dynamic battery management with auto-return  
- BFS fallback for safe recovery from low-battery or deadlock situations  
- Tkinter GUI and Matplotlib visual feedback  
- Reward-based adaptive learning system  

## Technology Stack

| Category | Tool / Library |
|-----------|----------------|
| Language | Python 3.x |
| Simulation Framework | Gymnasium |
| RL Algorithm | Custom Q-Learning |
| Visualization | Tkinter, Matplotlib |
| Data Handling | NumPy |
| Utility | `collections.deque`, `random` |
