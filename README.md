# PokémonRL

A machine-learning Pokémon battle agent trained to learn from human gameplay and eventually improve through reinforcement learning and self-play.

The project uses **PyTorch**, **poke-env**, and a local **Pokémon Showdown** server to build an end-to-end pipeline from raw battle replays to an autonomous battling agent.

## Overview

Pokémon battles are a difficult reinforcement learning environment: the agent must reason about partial information, team composition, switching, type matchups, status effects, field conditions, and long-term strategy while choosing from a changing set of legal actions.

Rather than training an RL agent entirely from scratch, PokémonRL uses a two-stage approach:

1. **Behavior Cloning** — learn an initial policy from millions of human battle decisions.
2. **Reinforcement Learning / Self-Play** — use the pretrained policy as a starting point and improve it through simulated battles.

The goal is to eventually deploy the trained agent in a **human-vs-AI battle platform** where players can directly challenge the model.

## Current Progress

### Human Replay Dataset Pipeline

The supervised training pipeline processes **2.5M+ state-action pairs** derived from the Metamon Pokémon replay dataset.

Raw battle data is:

* Downloaded and extracted from the replay dataset
* Parsed into individual battle states and player decisions
* Cleaned and encoded into model-ready representations
* Split into sharded files for efficient streaming during training
* Loaded incrementally to support training under limited-memory environments such as Google Colab

This allows the model to train on a dataset substantially larger than available system memory.

### Behavior Cloning

The first version of the agent is trained using **behavior cloning**, treating Pokémon battle decisions as a supervised classification problem.

Given an encoded battle state, the policy network learns to predict the action selected by a human player.

This provides the agent with an initial understanding of concepts such as:

* Move selection
* Pokémon switching
* Type matchups
* Team state
* Opponent information
* HP and status conditions
* Weather and field effects
* Legal action constraints

A trained PyTorch policy checkpoint is included in the project under:

`bot/training/best_bc_policy.pt`

### Pokémon Showdown Integration

Battles are executed through **poke-env**, which provides a Python interface to Pokémon Showdown.

The project currently supports interaction with a local Showdown server and exposes battle information including:

* Current Pokémon
* Known opponent Pokémon
* HP
* Status conditions
* Stat boosts
* Moves
* Items and abilities
* Weather
* Field effects
* Side conditions
* Available moves
* Available switches

This interface will serve as the environment layer for reinforcement learning.

## Reinforcement Learning Roadmap

The next stage of the project is transitioning the behavior-cloned model into an online RL agent.

The planned training pipeline is:

```text
Human Replays
      |
      v
Data Processing
      |
      v
Behavior Cloning
      |
      v
Pretrained Policy
      |
      v
Pokemon Showdown Environment
      |
      v
Self-Play / Reinforcement Learning
      |
      v
Improved Battle Policy
      |
      v
Human vs AI Web Platform
```

Instead of immediately training over the full Pokémon battle space, the RL environment will use constrained battle formats to reduce complexity and accelerate learning.

Planned environments include:

* Modified Generation 9 Random Battles
* Fixed-level singles battles
* Restricted team / ruleset environments
* Progressive self-play against increasingly strong opponents

## Repository Structure

```text
PokeMonRL/
├── bot/
│   ├── main.py
│   └── training/
│       ├── BC_train.py
│       └── best_bc_policy.pt
│
├── data/
│   ├── download.py
│   ├── extract.py
│   ├── splitData.py
│   ├── parseData.ipynb
│   └── checkData.ipynb
│
├── showdown-server/
│
├── tests/
│
├── pyproject.toml
└── README.md
```

## Tech Stack

**Machine Learning**

* Python
* PyTorch
* scikit-learn
* WebDataset

**Battle Environment**

* poke-env
* Pokémon Showdown
* Asyncio

**Data Processing**

* pandas
* LZ4
* Hugging Face
* Sharded datasets

## Project Goals

The long-term goal is to build an agent capable of learning increasingly sophisticated Pokémon strategy rather than relying on manually programmed battle heuristics.

Key milestones:

* [x] Build replay ingestion pipeline
* [x] Parse and encode human battle trajectories
* [x] Process 2.5M+ state-action pairs
* [x] Implement streaming dataset training
* [x] Train initial behavior-cloned policy
* [x] Integrate with Pokémon Showdown through poke-env
* [x] Connect pretrained policy to live battle decisions
* [ ] Implement reinforcement learning training loop
* [ ] Implement self-play
* [ ] Evaluate against baseline agents
* [ ] Develop human-vs-AI battle API
* [ ] Build web battle interface
* [ ] Deploy the trained agent

## Why Behavior Cloning First?

Training purely through reinforcement learning would require the agent to discover basic Pokémon strategy through trial and error.

Instead, behavior cloning provides the agent with a strong initial policy learned from human games.

The intended progression is:

**Human knowledge → imitation → exploration → self-play → stronger policy**

This reduces the amount of exploration required during early RL training and allows reinforcement learning to focus on improving strategy rather than discovering basic gameplay from scratch.

## Status

**Work in progress — September 2026**

The supervised pretraining pipeline and initial behavior-cloned model are implemented. Current development is focused on connecting the learned policy to live Pokémon Showdown battles and building the reinforcement-learning/self-play training system.
