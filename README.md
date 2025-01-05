# Lilly Trotters

## Project description

Abstract:

We present a neural network architecture for learning state representations and decision-making in Lux AI Season 3, a multi-agent resource management game. Our approach adapts OpenAI Five's architecture by combining a spatial-temporal encoder with an action selection mechanism specifically designed for partially observable, multi-unit control scenarios. The architecture consists of three key components:
* CNN-based spatial encoder that processes the 24x24 game board and unit states
* LSTM(?) network
* Hierarchical action network


To address the challenge of initial training without action labels, we implement a self-supervised pretraining approach that predicts future game states, unit energy levels, and position changes. Using a dataset of 500 games (approximately 250,000 state transitions), our model learns meaningful state representations before fine-tuning for action selection. Empirical results show that this pretraining approach significantly improves the model's understanding of game dynamics and spatial relationships, providing a foundation for subsequent reinforcement learning or imitation learning phases.

The architecture's modular design allows for independent optimization of state representation and action selection, while the temporal component captures crucial sequential dependencies in unit coordination and resource management. This work provides a framework for developing AI agents in complex, multi-unit strategy games with partial observability.

## [Game explained](https://github.com/Lux-AI-Challenge/Lux-Design-S3/blob/main/docs/specs.md)

Lux AI Season 3 is a competitive multi-agent resource management game played on a 24x24 grid-based map where two players compete to accumulate the most resources within 100 timesteps. The game environment features multiple terrain types including empty spaces, asteroids containing mineable resources, nebulas that affect movement costs, and energy cells that provide power.

Each player controls up to four units simultaneously, with units capable of moving in five directions (center, up, right, down, left) and performing resource collection actions through a 'sap' mechanism with a maximum range of 7 tiles. Units maintain individual energy levels which are depleted by movement and sap actions, introducing a critical resource management component.

The game implements partial observability through a fog-of-war system, requiring players to strategically explore and maintain visibility of key resource locations.

Victory is achieved by either accumulating more resources than the opponent or by eliminating enemy units through tactical energy management. The game's complexity arises from the need to balance multiple objectives: efficient resource collection, strategic positioning, energy conservation, and opponent counterplay, all while operating under imperfect information conditions. Map generation includes procedurally generated terrain layouts, ensuring each match presents unique strategic challenges and preventing memorized strategies from dominating gameplay. This is very similar to League.

## Dynamic Environment Complexities:

Map Properties:
* 24x24 2D grid
* Contains multiple tile types: Empty, Asteroid (impassable), Nebula (affects vision/energy)
* Has Energy Nodes that emit harvestable energy fields
* Features Relic Nodes that generate points when ships are nearby
* Map state persists between matches in a game

Game Structure:
* Best of 5 matches
* Each match lasts 100 time steps
* Features fog of war (limited vision)
* Parameters are randomized at start of each game but remain constant across matches

Unit Properties:
* Units are ships that can move in 5 directions (up, down, left, right, center)
* Start with 100 energy, maximum 400 energy
* Can perform sap actions to reduce enemy unit energy
* Have vision/sensor range affected by nebula tiles
* Can stack with friendly units
* Generate energy void fields affecting adjacent enemy units

Key Mechanics:
* Fog of war with vision power system
* Unit collision resolution based on aggregate energy
* Energy management (collection vs consumption)
* Point scoring through relic nodes
* Symmetric map features and movement

Strategic Elements:
* Balance between exploration and exploitation
* Energy management
* Unit positioning and stacking
* Vision control through nebula tiles
* Risk management with sap actions

Randomized Parameters:
* Unit move cost (1-5)
* Unit sensor range (2-4)
* Nebula vision reduction (0-3)
* Sap action costs and ranges
* Various drift speeds for map features

## Exploitable environment parts

Use params from [here](https://github.com/Lux-AI-Challenge/Lux-Design-S3/blob/main/src/luxai_s3/params.py)
Use env.py from [here](https://github.com/Lux-AI-Challenge/Lux-Design-S3/blob/main/src/luxai_s3/env.py)

**Energy Void Field Mechanics:**

* Units create void fields in adjacent tiles
* We can position units to maximize void field damage to enemies
* Stack units strategically to increase void field strength - This might not be too good

**Predictable Movement Patterns:**
* Nebulas move northeast consistently
* We can predict and position units accordingly
* Use this to plan resource collection routes

**Spawn Behavior:**
- Units spawn at fixed intervals
- Can time aggressive moves with spawn cycles
- Exploit spawn positions (corners)

**Collision Resolution**
- Units survive if they have more total energy
- Can stack units for stronger positions
- Energy management crucial for survival

**Sap Actions**
- Sap affects both target tile and adjacents
- Can maximize damage with proper positioning
- Adjacent tiles receive reduced damage

**Unit Energy Clipping**
* Energy has max/min bounds
* Can force enemy units into negative energy states
* Stack energy gains efficiently

**Scoring Mechanisms**
- Only needs one unit per relic for points
- Can spread units thin for maximum coverage
- Don't need to defend relics with multiple units

**Spawn capping**
- Units spawn in corners
- Can camp enemy spawn points
- Time attacks with enemy spawns

**Vision**
* Vision is additive from multiple units
* Can create vision walls
* Nebulas reduce vision

**All others**
* Nebula movement is deterministic - can setup pincer movements
* Energy calculations happen before collision resolution
* Unit stacking affects void field distribution
* Vision power is cumulative from multiple units
* Relic scoring only checks for presence, not duration


## V0 Architecture

This is getting too long, so i am linking the page here.

### Why the name ?

As many of you know, I have a deep love for birds. All my project names are inspired by birds observed near Chilika, Odisha.

This project is named after the graceful ପାନି କୁଳିକା (Pāni Kuḷikā), more charmingly known as lily trotters. These delightful birds are nature’s acrobats, dancing effortlessly across floating lilies and water plants.

Just like them, I hope this project treads gracefully in its domain. 🌿✨

![Lily Trotters](https://upload.wikimedia.org/wikipedia/commons/4/4e/Irediparra_gallinacea_-_Comb-crested_Jacana.jpg)
