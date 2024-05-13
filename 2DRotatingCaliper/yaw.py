import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import rotating_caliper
import point2D as p2D

def yawCompute(points):
    rectangle = rotating_caliper.rotatingCaliper(points)
    rectangle_vertices = np.array([[point.x, point.y] for point in rectangle.vertices])
    p1_p2 = np.array([rectangle_vertices[1][0] - rectangle_vertices[0][0], rectangle_vertices[1][1] - rectangle_vertices[0][1]])
    yaw = np.arctan2(p1_p2[1], p1_p2[0])
    yaw = np.degrees(yaw)
    return yaw