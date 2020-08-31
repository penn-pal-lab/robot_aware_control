# Progress
This document will log the progress of the project.

## TODOS:
- Test pixel observation, ground truth dynamics, and MSE pixel cost function on FetchReach, and FetchPush

### 8/29/20
- Done with initial CEM implementation from AAP project. Tested on FetchReach and seems to work well.
- Copying pusher env from resetrl project. Need to change the ob and action space to Gym format, and remove resetrl specific settings

### 8/30/20
- lifted Fetch code from Gym for customization
- changed push env initialization for robot and block to be near each other
- robot learns to shoot the block instead of pushing, but whatever. that's because of the cost function only caring about the block.
