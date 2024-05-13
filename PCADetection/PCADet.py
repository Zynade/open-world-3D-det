import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
from mpl_toolkits.mplot3d import Axes3D

class PCADetection: 
    def __init__(self, points):
        self.points = points
    def compute_rotation(self):
        self.mean = np.mean(self.points, axis = 1)
        cov_matrix = np.cov(self.points)
        eigenvalues, eigenvectors = LA.eig(cov_matrix)
        self.centered_points = self.points - self.mean[:, np.newaxis]
        self.rotation_matrix = eigenvectors

        if LA.det(self.rotation_matrix) < 0:
            self.rotation_matrix = -self.rotation_matrix
        print(self.rotation_matrix)
        aligned_points = self.rotation_matrix.T @ self.centered_points
        return aligned_points

    def corner_points(self):
        aligned_points = self.compute_rotation()
        xmin, xmax, ymin, ymax, zmin, zmax = np.min(aligned_points[0, :]), np.max(aligned_points[0, :]), np.min(aligned_points[1, :]), np.max(aligned_points[1, :]), np.min(aligned_points[2, :]), np.max(aligned_points[2, :])
        box_coordinates = lambda x1, y1, z1, x2, y2, z2: np.array([[x1, x1, x2, x2, x1, x1, x2, x2], 
                                                                    [y1, y2, y2, y1, y1, y2, y2, y1], 
                                                                    [z1, z1, z1, z1, z2, z2, z2, z2]])
        self.extremes = [xmin, xmax, ymin, ymax, zmin, zmax]
        self.realigned_points = self.rotation_matrix @ aligned_points
        self.realigned_points = self.realigned_points + self.mean[:, np.newaxis]
        self.corners = self.rotation_matrix @ box_coordinates(xmin, ymin, zmin, xmax, ymax, zmax)
        self.corners += self.mean[:, np.newaxis]
    
    def get_extents_and_centroid(self):
        self.corner_points()
        xmin, xmax, ymin, ymax, zmin, zmax = self.extremes
        length = xmax - xmin
        width = ymax - ymin
        height = zmax - zmin
        print(zmax - zmin)

        centroid = np.mean(self.corners, axis=1)

        return (width, length, height), centroid
        
    def compute_yaw(self):
        self.corner_points()
        self.yaw = np.arctan2(self.rotation_matrix[1, 0], self.rotation_matrix[0, 0])
        self.yaw = np.degrees(self.yaw)
        # self.pitch = np.arctan2(-self.rotation_matrix[2, 0], np.sqrt(self.rotation_matrix[2, 1]**2 + self.rotation_matrix[2, 2]**2))
        # self.roll = np.arctan2(self.rotation_matrix[2, 1], self.rotation_matrix[2, 2])
        return self.yaw
    
    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.realigned_points[0, :], self.realigned_points[1, :], self.realigned_points[2, :])
        # Plots for the edges of the MOBB
        ax.plot(self.corners[0, 0:2], self.corners[1, 0:2], self.corners[2, 0:2], color='r')
        ax.plot(self.corners[0, 1:3], self.corners[1, 1:3], self.corners[2, 1:3], color='r')
        ax.plot(self.corners[0, 2:4], self.corners[1, 2:4], self.corners[2, 2:4], color='r')
        ax.plot(self.corners[0, [3, 0]], self.corners[1, [3, 0]], self.corners[2, [3, 0]], color='r')
        ax.plot(self.corners[0, 4:6], self.corners[1, 4:6], self.corners[2, 4:6], color='r')
        ax.plot(self.corners[0, 5:7], self.corners[1, 5:7], self.corners[2, 5:7], color='r')
        ax.plot(self.corners[0, 6:], self.corners[1, 6:], self.corners[2, 6:], color='r')
        ax.plot(self.corners[0, [7, 4]], self.corners[1, [7, 4]], self.corners[2, [7, 4]], color='r')
        ax.plot(self.corners[0, [0, 4]], self.corners[1, [0, 4]], self.corners[2, [0, 4]], color='r')
        ax.plot(self.corners[0, [1, 5]], self.corners[1, [1, 5]], self.corners[2, [1, 5]], color='r')
        ax.plot(self.corners[0, [2, 6]], self.corners[1, [2, 6]], self.corners[2, [2, 6]], color='r')
        ax.plot(self.corners[0, [3, 7]], self.corners[1, [3, 7]], self.corners[2, [3, 7]], color='r')
        ax.scatter(self.corners[0, :], self.corners[1, :], self.corners[2, :], color='b')
        self.center = np.mean(self.corners, axis=1)
        ax.plot([self.center[0], self.center[0] + 1*self.rotation_matrix[0, 0]], [self.center[1], self.center[1] + 1*self.rotation_matrix[1, 0]], [self.center[2], self.center[2] + 1*self.rotation_matrix[2, 0]], color='magenta')
        ax.plot([self.center[0], self.center[0] + 1*self.rotation_matrix[0, 1]], [self.center[1], self.center[1] + 1*self.rotation_matrix[1, 1]], [self.center[2], self.center[2] + 1*self.rotation_matrix[2, 1]], color='cyan')
        ax.plot([self.center[0], self.center[0] + 1*self.rotation_matrix[0, 2]], [self.center[1], self.center[1] + 1*self.rotation_matrix[1, 2]], [self.center[2], self.center[2] + 1*self.rotation_matrix[2, 2]], color='yellow')
        plt.show()
