import pickle
from ipdb import set_trace
import pandas as pd

def load_robonet(metadata_path):
    df = pd.read_pickle(metadata_path, compression='gzip')


if __name__ == "__main__":
    metadata_path = "/media/ed/hdd/Datasets/Robonet/hdf5/meta_data.pkl"
    load_robonet(metadata_path)