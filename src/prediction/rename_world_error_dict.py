import pickle
import os

with open("baxter_left_world_error.pkl.backup", "rb") as f:
    d = pickle.load(f)

old_root = "/scratch/edward/Robonet/new_hdf5"
new_root = "/scratch/edward/Robonet/baxter_views/left_c0"
new_dict = {}
for k, v in d.items():
    file_name = k.split("/")[-1]
    file_name = file_name.split(".")[0] + "_c0.hdf5"
    new_key = os.path.join(new_root, file_name)
    new_dict[new_key] = v

with open("baxter_left_world_error_new.pkl", "wb") as f:
    pickle.dump(new_dict, f)