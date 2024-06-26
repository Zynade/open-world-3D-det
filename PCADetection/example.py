import PCADet
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
from mpl_toolkits.mplot3d import Axes3D

# Generating random points and finding the MOBB
x = np.linspace(3, 8, 100) + np.random.normal(0, 0.2, 100)
y = np.linspace(3, 8, 100) + np.random.normal(0, 0.2, 100)
z = np.linspace(1, 3, 100) + np.random.normal(0, 0.2, 100)
# add an element to x
x = np.append(x, -100)
y = np.append(y, -100)
z = np.append(z, -100)
points = np.array([x, y, z])
pca = PCADet.PCADetection(points)
yaw = pca.compute_yaw()
print(f"Yaw: {yaw}")
pca.plot()