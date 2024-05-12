Takes a set of N points in 3D space and computes the Minimum Oriented Bounding Box (MOBB) that encloses the points. The yaw of the bounding box is computed and returned in degrees. 

To run an example, execute the following command:
```python example.py```
This will output the yaw value of the MOBB that encloses the points in the example point cloud.

Additionally, the MOBB can be visualized by adding this line to the example.py file:
```pca.plot()```