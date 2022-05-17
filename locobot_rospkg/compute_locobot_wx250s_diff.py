import numpy as np

"""
----------------
|  robot
|           right
|
|        center
|  left
"""

locobot_points = np.array([
    [0.27, 0.12], # right
    [0.42, -0.04], # center
    [0.27, -0.18] # left
])

wx250s_points = np.array([
    [0.4, 0.13], # right
    [0.54, -0.03], # center
    [0.41, -0.17], # left
])


print(np.mean( locobot_points - wx250s_points, axis=0))