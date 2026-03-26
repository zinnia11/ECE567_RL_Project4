# ECE567_RL_Project4

## Changes made to the environment

The CORA Github repository was last updated in June 2023. Since then, OpenAI Gym was replaced by Gymnasium (November 2023) and NumPy was updated to NumPy 2.0 (June 2024). This means that some packages in CORA are difficult to use in present day. 

We update the environment by updating the dependencies and code to reflect more modern versions.

1. Replace ```atari-py``` with ```ale-py```
2. Replace ```gym[atari]``` with ```gymnasium[atari]```

We update the code to work with more recent package versions.

In Gymnasium, the ```env.step()``` function returns both a ```terminated``` and ```truncated``` value rather than just a single ```done``` value, so we combine both ```terminated``` and ```truncated``` into one ```done``` value to minimize code edits. The ```env.reset()``` funcation returns a ```(obs, info)``` tuple, where we only take and use the first ```obs``` value. 

In NumPy 2.0, ```np.float``` is deprecated, so we change that type to the new ```np.float64```. 

Previous environments in Gym should still exist under the same name in Gymnasium, so no further edits are needed. All previous code is commented out for record keeping. 

