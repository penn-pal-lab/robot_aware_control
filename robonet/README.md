# RoboNet

Code for loading and manipulating the RoboNet dataset, as well as for training supervised inverse models and video prediction models on the dataset.

Please refer to the [project wiki](https://github.com/SudeepDasari/RoboNet/wiki) for more detailed documentation.

If you find the codebase or dataset useful please consider citing our paper.

```text
@inproceedings{dasari2019robonet,
    title={RoboNet: Large-Scale Multi-Robot Learning},
    author={Sudeep Dasari and Frederik Ebert and Stephen Tian and Suraj Nair and Bernadette Bucher and Karl Schmeckpeper and Siddharth Singh and Sergey Levine and Chelsea Finn},
    year={2019},
    eprint={1910.11215},
    archivePrefix={arXiv},
    primaryClass={cs.RO},
    booktitle={CoRL 2019: Volume 100 Proceedings of Machine Learning Research}
}
```

## Downloading the Dataset

You can find instructions for downloading the dataset on the [project wiki](https://github.com/SudeepDasari/RoboNet/wiki/Getting-Started) as well. All data is provided under the [Creative Commons BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode) license.

For our project, follow the instruction on the [project wiki](https://github.com/SudeepDasari/RoboNet/wiki/Getting-Started), then download the small subset of RoboNet and put them under `hdf5` folder.

## Dataset Spec

State: 5D, [(x, y, z, rotation)?, gripper]

## Dataset Visualization

For visualizing specific hdf5 file:

```bash
mkdir images
python dataset_visualizer.py hdf5/experiment_name.hdf5
```

For collecting and visualizing experiments on specific robot and viewpoint, run

```bash
python data_collector.py hdf5/ berkeley_sawyer
```

For calibrating a specific viewpoint of a robot, run

```bash
python camera_calib/robot_viewpoint_calib.py berkeley_sawyer sudri0 3 
```

Candidate Robot Names: sawyer, kuka, R3, widowx, baxter, fetch, franka

### Missing information

action: google, stanford fetch

qpos: stanford franka corr

## Camera Calibration

Penn uses Logitech C920 camera, and the camera intrinsic is `[641.5, 0, 320.0, 0, 641.5, 240.0, 0, 0, 1]`.

1. Run `dataset_visualizer.py` on at least 3 experiments. The corresponding data are stored in `images/`
2. Change `robonet_calibration.py` around line 10 `use_for_calibration` to the desired experiment names
3. `python camera_calib/robonet_calibration.py`
4. Click the end effector in the image, and press `space` to proceed to the next image.

## Workspace Dimension

```text
"baxter_right": [[ 0.40, -0.67 ,  -0.15, 15.0,  0.0 ], [ 0.75, -0.20,  -0.05,  3.2e+02,  1.0e+02]],
"baxter_left": [[ 0.45,  0.15, -0.15, 15.0,  0.0 ], [  0.75,  0.59, -0.05,  3.2e+02,  1.0e+02]],
"vestri": [[0.47, -0.2, 0.176, 1.5707963267948966, -1], [0.81, 0.2, 0.292, 4.625122517784973, 1]],
"vestri_table": [[0.43, -0.34, 0.176, 1.5707963267948966, -1], [0.89, 0.32, 0.292, 4.625122517784973, 1]],
"vestri_table_default": [[0.43, -0.34, 0.176, 1.5707963267948966, -1], [0.89, 0.32, 0.292, 4.625122517784973, 1]],
"sudri": [[0.45, -0.18, 0.176, 1.5707963267948966, -1], [0.79, 0.22, 0.292, 4.625122517784973, 1]],
"test": [[0.47, -0.2, 0.1587, 1.5707963267948966, -1], [0.81, 0.2, 0.2747, 4.625122517784973, 1]],
"baxter": [[ 0.6, -0.5 , -0.18, 15.0,  0.0 ], [ 8.0e-01, -5.0e-02,  -0.0316,  3.2e+02,  1.0e+02]]
```
