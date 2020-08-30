# Robot aware cost and dynamics
Use python 3.8.1
```
pip install -r requirements.txt
```

## Troubleshooting

Ran into C++ compilation error for `mujoco-py`. Had to correctly symlink GCC-6 to gcc
command for it to compile since it was using the system gcc, which was gcc-11.
