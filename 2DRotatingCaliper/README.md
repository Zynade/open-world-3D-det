Takes a set of N points in 3D space, clips the z-coordinate and computes the Minimum Oriented Bounding Box (MOBB) that encloses the points by computing the Convex Hull of the points using Graham's algorithm and rotating calipers. The yaw of the bounding box is computed and returned in degrees. 

To run an example, execute the following command:
```python example.py```
This will output the yaw value of the MOBB that encloses the points in the example point cloud along with the MOBB's dimensions.

Another example without the plots can be run by executing the following command:
```python example2.py```  