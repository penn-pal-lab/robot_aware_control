import numpy as np

def gaus2d(x=0, y=0, mx=0, my=0, sx=1, sy=1, height=1):
    return height / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))