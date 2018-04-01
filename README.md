# Smart Navigation Simulations

In this project we teach an agent to navigate to a target using reinforcement learning.
The project is designed so that the trained agent can easilly be applied to a real control task

## Environment

The learning environment is designed as follows:

* An agent is randomly spawned at some location in continuous space
* The agent must move to a target somewhere in the space
* The agent must avoid penalty tiles in the environment
* The agent must avoid other agents in the environment
* The simualtion ends when the agent reaches the target or has a collision

## Learning algorithms

Several different learning algorithms are tested.

### DDPG

Currently we have only implimented DDPG.
See the original implimentation here: [https://github.com/songrotek/DDPG](https://github.com/songrotek/DDPG)

### A3C

To be implimented

## Prerequisites

Conda is used for package management. To install conda on ubuntu:
```sh
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
bash Miniconda3-latest-Linux-x86_64.sh \
source /home/ubuntu/.bashrc \
source ubuntu.sh # Install ubuntu deps
```

Create a new environment with required dependencies:
```sh
git clone https://github.com/maxkferg/smart-navigation && cd smart-navigation
conda env create -f environment.yml
```

## Training

All the code was written with Python 3.6. The code should work with all recent versions of Python3.
Test the environment manually
```sh
python -m environments.hospital.environment
```

Run the training process. Change epsilon to a high value before training
the first time.
```sh
python run_ppo_hospital.py --train=True
```

Run using MPI
```sh
mpirun -n 70 python run_ppo_hospital.py --train=True
```

## License

* environments: Copyright Max Ferguson 2017
* learners: MIT
