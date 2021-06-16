import numpy as np

# holds the transform from camera space to world space.
# most notably, the translation and rotation parts of this matrix correspond
# to the camera's position and rotation in the world space.
camera_to_world_dict = {
    "baxter_left_c0": np.array(
        [
            [0.05010049, 0.5098481, -0.85880432, 1.70268951],
            [0.99850135, -0.00660876, 0.05432662, 0.26953027],
            [0.02202269, -0.86023906, -0.50941512, 0.48536055],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "baxter_right_c0": np.array(
        [
            [0.59474902, -0.48560866, 0.64066983, 0.00593267],
            [-0.80250365, -0.40577623, 0.4374169, -0.84046503],
            [0.04755516, -0.77429315, -0.63103774, 0.45875102],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "sawyer_sudri0_c0": np.array(
        [
            [-0.01290487, 0.62117762, -0.78356355, 1.21061856],
            [1, 0.00660994, -0.01122798, 0.01680913],
            [-0.00179526, -0.78364193, -0.62121019, 0.47401633],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "sawyer_sudri0_c1": np.array(
        [
            [0.9975901, 0.0691292, 0.00592799, 0.60620359],
            [0.04619134, -0.72546495, 0.68670734, -0.42756365],
            [0.05177208, -0.68477862, -0.72690982, 0.53600216],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "sawyer_sudri0_c2": np.array(
        [
            [-0.35527701, 0.41521095, -0.8374832, 1.12403976],
            [0.9189123, -0.00914706, -0.39435582, 0.24057687],
            [-0.17140136, -0.90967917, -0.37829271, 0.29666432],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "sawyer_sudri2_c0": np.array(
        [
            [-0.20352987, 0.64259509, -0.73867932, 1.17506129],
            [0.9567336, -0.02969794, -0.28944578, 0.19938629],
            [-0.20793369, -0.76563018, -0.6087479, 0.46536255],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "sawyer_sudri2_c1": np.array(
        [
            [0.99706184, 0.07581474, 0.01094559, 0.55393717],
            [0.04626195, -0.7098712, 0.70281058, -0.4425706],
            [0.06105336, -0.70023925, -0.71129282, 0.52610051],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "sawyer_sudri2_c2": np.array(
        [
            [-0.39771899, 0.36153698, -0.84327375, 1.14520489],
            [0.89713902, -0.03934587, -0.4399926, 0.30102312],
            [-0.19225293, -0.9315272, -0.30870033, 0.28974425],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "sawyer_vestri_table2_c0": np.array(
        [
            [-0.01183555, 0.58241102, -0.8128083, 1.31055191],
            [0.99973558, -0.00913481, -0.02110293, 0.0089173],
            [-0.01971543, -0.81284313, -0.5821489, 0.50151772],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "sawyer_vestri_table2_c1": np.array(
        [
            [0.99962747, 0.01402494, -0.02341411, 0.65820915],
            [0.0265253, -0.70128186, 0.71239046, -0.47751281],
            [-0.00642866, -0.71274614, -0.70139263, 0.56862831],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "sawyer_vestri_table2_c2": np.array(
        [
            [-0.06536258, 0.43301436, -0.89901407, 1.24390769],
            [0.99785944, 0.02649836, -0.05978605, 0.0647729],
            [-0.00206582, -0.90099745, -0.43381947, 0.36955964],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "widowx_widowx1_c0": np.array(
        [
            [-0.17251765, 0.5984481, -0.78236663, 0.37869496],
            [-0.98499368, -0.10885336, 0.13393427, -0.04712975],
            [-0.00501052, 0.79373221, 0.60824672, 0.15596613],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "locobot_c0": np.array(
        [
            [0.10142061, 0.72632463, -0.67386291, 0.78975893],
            [0.98958408, -0.08242317, 0.06193354, -0.03911564],
            [-0.00928995, -0.68100839, -0.72849251, 0.64767807],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),

    "locobot_c1": np.array(
        [
            [0.10142061, 0.72632463, -0.67386291, 0.78975893],
            [0.98958408, -0.08242317, 0.06193354, -0.03911564],
            [-0.00928995, -0.68100839, -0.72849251, 0.64767807],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "locobot_c2": np.array(
        [
            [0.10142061, 0.72632463, -0.67386291, 0.78975893],
            [0.98958408, -0.08242317, 0.06193354, -0.03911564],
            [-0.00928995, -0.68100839, -0.72849251, 0.64767807],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "locobot_c3": np.array(
        [
            [0.10142061, 0.72632463, -0.67386291, 0.78975893],
            [0.98958408, -0.08242317, 0.06193354, -0.03911564],
            [-0.00928995, -0.68100839, -0.72849251, 0.64767807],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),

    "locobot_modified_c0": np.array(
        [
            [0.0452768, 0.73303716, -0.67868, 0.79116035],
            [0.99869241, -0.01707084, 0.04818772, -0.00249282 - 0.015],
            [0.02373775, -0.67997435, -0.73285156, 0.64026054 + 0.0125],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
}

world_to_camera_dict = {k: np.linalg.inv(v) for k, v in camera_to_world_dict.items()}

cam_intrinsics_dict = {
    # captured 320 x 240 images in robonet
    "logitech_c420": np.array([[320.75, 0, 160], [0, 320.75, 120], [0, 0, 1]]),
    # captured 640 x 480 images for locobot
    "intel_realsense_d435": np.array(
        [[612.45, 0, 330.55], [612.56, 0, 248.61], [0, 0, 1]]
    ),
}
