"""Rename the files in dataset
"""

import os
from tqdm import tqdm

dir = "/home/ed/Robonet/sudri0_c1_hdf5"
# append c0 suffix to all strings
for file_name in tqdm(os.listdir(dir)):
    parts = file_name.split(".")
    parts[0] += "_c1"
    new_name = ".".join(parts)
    old_file = os.path.join(dir, file_name)
    new_file = os.path.join(dir, new_name)
    os.rename(old_file, new_file)
