import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import rotating_caliper
import point2D as p2D

# Generating random points and finding the MOBB
x = np.linspace(3, 8, 100) + np.random.normal(0, 0.2, 100)
y = np.linspace(4, 7, 100) + np.random.normal(0, 0.2, 100)
# z = np.linspace(1, 3, 100) + np.random.normal(0, 0.2, 100)
z = np.zeros(100)
points = np.array([x, y, z]).T
# obtain the coordinates of the MOBB
rectangle = rotating_caliper.rotatingCaliper(points)

rectangle_vertices = np.array([[point.x, point.y] for point in rectangle.vertices])

# Plotting the points and the MOBB
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2])
# Plotting the MOBB
for i in range(len(rectangle_vertices)):
    ax.plot([rectangle_vertices[i][0], rectangle_vertices[(i+1)%4][0]], [rectangle_vertices[i][1], rectangle_vertices[(i+1)%4][1]], zs=0, zdir='z')
ax.plot(rectangle_vertices[:, 0], rectangle_vertices[:, 1], zs=0, zdir='z')
plt.show()
# Calculating the yaw of the MOBB
p1_p2 = np.array([rectangle_vertices[1][0] - rectangle_vertices[0][0], rectangle_vertices[1][1] - rectangle_vertices[0][1]])
yaw = np.arctan2(p1_p2[1], p1_p2[0])
yaw = np.degrees(yaw)
print("Yaw: ", yaw)