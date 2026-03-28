# ECE567_RL_Project4

<!--
## Changes made to the environment

The CORA Github repository was last updated in June 2023. Since then, OpenAI Gym was replaced by Gymnasium (November 2023) and NumPy was updated to NumPy 2.0 (June 2024). This means that some packages in CORA are different to use in present day. 

We update the environment by updating the dependencies and code to reflect more modern versions.

1. Replace ```atari-py``` with ```ale-py```
2. Replace ```gym[atari]``` with ```gymnasium[atari]```

We update the code to work with more recent package versions.

In Gymnasium, the ```env.step()``` function returns both a ```terminated``` and ```truncated``` value rather than just a single ```done``` value, so we combine both ```terminated``` and ```truncated``` into one ```done``` value to minimize code edits. The ```env.reset()``` funcation returns a ```(obs, info)``` tuple, where we only take and use the first ```obs``` value. 

In NumPy 2.0, ```np.float``` is deprecated, so we change that type to the new ```np.float64```. 

Previous environments in Gym should still exist under the same name in Gymnasium, so no further edits are needed. All previous code is commented out for record keeping. 

-->

## Environment setup

We recommend using Pip to set up the environment, as the ```environment.yml``` file is old and has a lot of conflicts.

First create and activate a virtual environment:

```
python3.9 -m venv <env name>
source <env name>/bin/activate
```

Then install Pytorch and the project + dependencies:

```
pip install torch torchvision
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117 numpy==1.23.5 gym[atari]==0.25.2 atari-py==0.2.5
pip install torch==2.2.0 torchvision==0.17.0 numpy==1.23.5 gym[atari]==0.25.2 atari-py==0.2.5
pip install -e .
```

### On Great Lakes

On the Great Lakes HPC, there is no Python 3.9 version, we recommend creating a conda environment, .

```
conda create -n <env name> python=3.9 -y
conda activate <env name>
```

Then install the packages using pip.

Now you should be ready to run experiments. 

