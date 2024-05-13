import point2D as p2D
import convexHull as ch
import rectangle as r

def edgeeGeneration(points):
    '''
    Returns all the edges of the points in a sequence. 
    '''
    edges = []
    for i in range(1, len(points)):
        edges.append((points[i-1], points[i]))
    edges.append((points[-1], points[0]))
    return edges

def rotatingCaliper(points):
    '''
    Returns the minimum oriented bounding box of the points. For a given set of points, it computes the convex hull of the points and then computes the minimum oriented bounding box of the convex hull. The vertices of the minimum oriented bounding box are returned.
    Parameters:
    points: list of Point2D objects
    Returns: 
    minRectangle: Rectangle object of the minimum oriented bounding box
    '''
    points = [p2D.Point2D(point) for point in points]
    convexHullPoints = ch.ConvexHull.hullCompute(points)
    edges = edgeeGeneration(convexHullPoints)
    directionVectors, edgeLengths, normalVectors, perpendicularVectors = [], [], [], []
    for edge in edges: 
        directionVector = p2D.Point2D([edge[1].x - edge[0].x, edge[1].y - edge[0].y])
        directionVectors.append(directionVector)
        edgeLengths.append((directionVector.x ** 2 + directionVector.y ** 2) ** 0.5)
    for i in range(len(directionVectors)):
        normalVectors.append(p2D.Point2D([directionVectors[i].y/edgeLengths[i], -directionVectors[i].x/edgeLengths[i]]))
    for i in range(len(normalVectors)):
        perpendicularVectors.append(p2D.Point2D([-normalVectors[i].y, normalVectors[i].x]))
    minX, maxX = [], []
    for i in range(len(normalVectors)):
        minXVal = float('inf')
        maxXVal = float('-inf')
        for j in range(len(convexHullPoints)):
            temp = convexHullPoints[j].x * normalVectors[i].x + convexHullPoints[j].y * normalVectors[i].y
            maxXVal = max(maxXVal, temp)
            minXVal = min(minXVal, temp)
        minX.append(minXVal)
        maxX.append(maxXVal)
    minY, maxY = [], []
    for i in range(len(perpendicularVectors)):
        minYVal = float('inf')
        maxYVal = float('-inf')
        for j in range(len(convexHullPoints)):
            temp = convexHullPoints[j].x * perpendicularVectors[i].x + convexHullPoints[j].y * perpendicularVectors[i].y
            maxYVal = max(maxYVal, temp)
            minYVal = min(minYVal, temp)
        minY.append(minYVal)
        maxY.append(maxYVal)
    minResult = float('inf')
    minIndex = -1
    for i in range(len(minX)):
        result = abs((maxX[i] - minX[i]) * (maxY[i] - minY[i]))
        if result < minResult:
            minResult = result
            minIndex = i
    minRectangle = r.Rectangle(p2D.Point2D((minX[minIndex], minY[minIndex])), 
                               p2D.Point2D((maxX[minIndex], minY[minIndex])), 
                               p2D.Point2D((maxX[minIndex], maxY[minIndex])),
                               p2D.Point2D((minX[minIndex], maxY[minIndex])))
    for i in range(len(minRectangle.vertices)):
        minRectangle.vertices[i] = [minRectangle.vertices[i].x * normalVectors[minIndex].x + minRectangle.vertices[i].y * perpendicularVectors[minIndex].x, 
                                    minRectangle.vertices[i].x * normalVectors[minIndex].y + minRectangle.vertices[i].y * perpendicularVectors[minIndex].y]
        minRectangle.vertices[i] = p2D.Point2D(minRectangle.vertices[i])
    return minRectangle
