# Progress
This document will log the progress of the project.

## TODOS:
- make configurable
- make static version of original camera
- fix randomness seed


### 8/29/20
- Done with initial CEM implementation from AAP project. Tested on FetchReach and seems to work well.
- Copying pusher env from resetrl project. Need to change the ob and action space to Gym format, and remove resetrl specific settings

### 8/30/20
- lifted Fetch code from Gym for customization
- changed push env initialization for robot and block to be near each other
- robot learns to shoot the block instead of pushing, but whatever. that's because of the cost function only caring about the block.

### 9/1/20
- Implemented initial version of goal sampling for fetch reach.
- Changed cost function for toy task to include robot gripper distance, for better comparison against MSE pixel cost
- Run initial experiments, record success rate between Baseline CEM vs Pixel CEM

10 trials. Need to fix the goal sampling though so they're the same for both.
Baseline CEM: 0.9
Pixel CEM: 0 lol

### 9/2/20
- got the initial version of mujoco segmentation.
- got the weighted cost function with alpha weighting
- running initial weighted pixel CEM vs pixel CEM

### 9/4/20
-