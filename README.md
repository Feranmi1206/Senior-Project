# Optimizing Bipedal Locomotion
## Overview

This repository contains a reinforcement learning (RL) training framework implemented using Isaac Gym and `rl_games`. It is designed to demonstrate the integration of an Isaac Gym environment for training a bipedal humanoid model to walk leveraging the highly optimized `rl_games` library.

## Installation
### IsaacGym

An installation of IsaacGym is required to run this project. Request access to the framework at https://developer.nvidia.com/isaac-gym and follow the installation instructions to get the environment properly setup. I suggest utilising a conda environment for the installation if possible as this makes it easier to manage the required dependencies.

### IsaacGymEnvs

For ease-of-use, this IsaacGym implementation makes use of IsaacGymEnvs, an API for creating preset vectorized environments. The source code for this API can be found at https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/tree/main. Clone this repository into the file created for your IsaacGym installation to get started

### Setting up the custom task

This project implements a new instance of the `VecTask` class as outlined in `docs/framework.md` in the IsaacGymEnvs repository. In order to properly run this implementation, you must add the new task to the imports and `isaacgym_task_map` dict in the `isaacgymenvs/tasks/__init__.py` file.

You will also need to add config files for task and training, which will be passed in dictionary form to the first config argument of your task. The task config, which goes in the corresponding config folder, is the `MyHumanoid.yaml` file in this repository and must be added as a new file in the path `isaacgymenvs/cfg/task/`. 

You also need a train config specifying RL Games arguments. This is the `MyHumanoidPPO.yaml` in the repository be added as a new file in the path `isaacgymenvs/cfg/train/`.

## Running the code

To train your first policy, run this line:

```bash
python train.py task=MyHumanoid
```

Note that by default, this will show a preview window, which will slow down training. You 
can use the `v` key while running to disable viewer updates and allow training to proceed 
faster. Hit the `v` key again to resume viewing to check in on the progress.

Use the `esc` key or close the viewer window to stop training early.

Alternatively, you can train headlessly, as follows:

```bash
python train.py task=MyHumanoid headless=True
```

### Loading trained models // Checkpoints

Checkpoints are saved in the folder `runs`. To load a trained checkpoint and continue training, use the `checkpoint` argument:

```bash
python train.py task=MyHumanoid checkpoint=runs/path_to_checkpoint
```

To load a trained checkpoint and only perform inference (no training), pass `test=True` 
as an argument, along with the checkpoint name. To avoid rendering overhead, you may 
also want to run with fewer environments using `num_envs=64`:

```bash
python train.py task=MyHumanoid checkpoint=runs/path_to_checkpoint test=True num_envs=64
```
