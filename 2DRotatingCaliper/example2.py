import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import rotating_caliper
import point2D as p2D
import yaw 

# Generating random points and finding the MOBB
x = np.linspace(3, 8, 100) + np.random.normal(0, 0.2, 100)
y = np.linspace(4, 7, 100) + np.random.normal(0, 0.2, 100)
# z = np.linspace(1, 3, 100) + np.random.normal(0, 0.2, 100)
z = np.zeros(100)
points = np.array([x, y, z]).T
yaw = yaw.yawCompute(points)
print("Yaw: ", yaw)
