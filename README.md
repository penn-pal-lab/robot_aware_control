# Robot aware cost and dynamics
Use python 3.8.1

- Implement a block pushing environment, and get CEM running on ground truth state, dynamics, and cost function
- Then test pixel observation, ground truth dynamics, and MSE pixel cost function

## Troubleshooting

Ran into C++ compilation error for `mujoco-py`. Had to correctly symlink GCC-6 to gcc
command for it to compile since it was using the system gcc, which was gcc-11.
