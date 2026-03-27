# Adaptive Traffic Light Control (DDQN)

## Overview
Simulates a 4-phase traffic signal using reinforcement learning.

## Changes
Improvements.md

## Features
- Poisson vehicle arrivals
- Queue-based modeling
- Waiting time penalty
- Yellow phase enforcement

## Actions
0: NS Through  
1: EW Through  
2: NS Left  
3: EW Left  

## Reward
r = -(0.7 * Queue + 0.3 * Wait)

## Run
python test_env.py