import numpy as np
import tensorflow as tf
import h5py
import matplotlib.pyplot as plt

from robonet.datasets import load_metadata
from robonet.datasets.robonet_dataset import RoboNetDataset

def load_hdf5(path):
    demo = {}
    hf =  h5py.File(path, "r")
    demo["cam0_video"] = hf["env"]['cam0_video']['frames']
    demo["cam1_video"] = hf["env"]['cam1_video']['frames']
    demo["cam2_video"] = hf["env"]['cam2_video']['frames']
    demo["cam3_video"] = hf["env"]['cam3_video']['frames']
    demo["cam4_video"] = hf["env"]['cam4_video']['frames']

    return demo

if __name__ == "__main__":
    data_dir = 'hdf5/'
    all_robonet = load_metadata(data_dir)                               # path to folder you unzipped in step (1)
    sawyer_data = all_robonet[all_robonet['robot'] == 'sawyer']         # pythonic filtering supported
    sawyer_files = sawyer_data.get_shuffled_files()  # gets shuffled list of sawyer files

    sawyer_data.get_file_metadata(sawyer_files[0])
    f1 = h5py.File(sawyer_files[0], 'r')
    # f1.keys(): 
    #   <KeysViewHDF5 ['env', 'file_version', 'metadata', 'misc', 'policy']>
    # f1['env'].keys(): 
    #   <KeysViewHDF5 ['cam0_video', 'cam1_video', 'cam2_video', 'cam3_video', 'cam4_video', 
    #   'finger_sensors', 'high_bound', 'low_bound', 'qpos', 'qvel', 'state']>
    # f1['env']['qpos']:
    #   <HDF5 dataset "qpos": shape (31, 7), type "<f8">

    demo = load_hdf5(sawyer_files[0])

    import pdb; pdb.set_trace()


    data = RoboNetDataset(batch_size=1, dataset_files_or_metadata=sawyer_data)
    train_images = data['images']  # images, states, and actions are from paired
    train_actions = data['actions']
    train_states = data['states']

    # s = tf.Session()
    # imgs = s.run(train_images, feed_dict=data.build_feed_dict('train'))

    mode = 'train'
    tensors = [data[x, mode] for x in ['images', 'states', 'actions']]
    s = tf.Session()
    out_tensors = s.run(tensors, feed_dict=data.build_feed_dict(mode))

    import pdb
    pdb.set_trace()

    plt.figure()
    plt.imshow(imgs[0, 0, 0])
    plt.show()
