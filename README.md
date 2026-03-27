# Advanced Traffic Intersection Simulator

## Overview
A real-time traffic intersection simulator built with pygame that models a 4-way intersection with intelligent traffic signal control. Vehicles follow realistic traffic rules and respond to traffic lights.

## Features

### Vehicle Behavior
- **Realistic Traffic Rules**: Vehicles stop at red lights and move when signals are green
- **Individual Vehicles**: Each vehicle has its own state (approaching, waiting, crossing, exiting)
- **Turning Support**: Vehicles can go straight or turn at the intersection
- **Lane Types**: Separate straight and turning lanes for organized traffic flow

### Traffic Signal Control
- **4-Phase System**: 
  - Phase 0: North-South straight traffic
  - Phase 1: North-South turns
  - Phase 2: East-West straight traffic
  - Phase 3: East-West turns
- **Yellow Lights**: Vehicles in intersection can proceed through yellow
- **Minimum Green Duration**: Prevents rapid signal switching

### Simulation
- **Poisson Arrivals**: Realistic random vehicle generation
- **Continuous Operation**: Environment runs indefinitely (no episode resets)
- **Real-time Visualization**: 600x600 pygame window showing intersection and vehicles
- **Interactive Controls**: Pause, resume, and quit simulation in real-time

## Traffic Control Policies

Three different policies are available for controlling traffic signals:

1. **Greedy** - Prioritizes direction with longer queue
2. **Random** - Randomly switches between phases
3. **Fixed-Time** - Uses predetermined phase durations

## Usage

### Basic Usage
```bash
python main.py
```
Runs with default settings (Greedy policy, Medium traffic)

### Custom Configuration
```bash
python main.py --policy greedy --traffic heavy
python main.py --policy random --traffic light
python main.py --policy fixed --traffic medium
```

### Command-Line Arguments
- `--policy {greedy, random, fixed}` - Traffic control policy (default: greedy)
- `--traffic {light, medium, heavy}` - Traffic intensity (default: medium)

### Controls
- **SPACE** - Pause/Resume simulation
- **R** - Reset episode
- **Q** - Quit application

## Color Legend
- **Blue** - Vehicles going straight
- **Cyan** - Vehicles turning
- **Green** - Traffic light is green
- **Yellow** - Traffic light is yellow
- **Red** - Traffic light is red

## Configuration
Edit `env_config.py` to change:
- Arrival rate (vehicles per step)
- Turn ratio (fraction of turning vehicles)
- Yellow light duration
- Minimum green time
- Queue capacity limits

## Example Run
```bash
python main.py --policy greedy --traffic heavy
```
This runs a heavy traffic simulation with the greedy traffic control policy.