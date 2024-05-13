import copy
import functools
from collections import deque

class ConvexHull:
    @staticmethod
    def determinant(p, q, r):
        '''
        Returns the determinant of the matrix formed by the vectors p, q, and r.
        Parameters:
        p: Point2D
        q: Point2D
        r: Point2D
        '''
        return (p.x - r.x) * (q.y - r.y) - (q.x - r.x) * (p.y - r.y)
    
    def distance(self, pt, p, q):
        '''
        Returns the distance between the point pt and the line formed by the points p and q
        Parameters:
        pt: Point2D
        p: Point2D
        q: Point2D
        '''
        pt_p = (p.x - pt.x, p.y - pt.y)
        pt_q = (q.x - pt.x, q.y - pt.y)
        pt_p_mod = (pt_p[0] ** 2 + pt_p[1] ** 2)**(0.5)
        pt_q_mod = (pt_q[0] ** 2 + pt_q[1] ** 2)**(0.5)
        if pt_p_mod > pt_q_mod:
            return 1
        return -1
    
    @classmethod
    def calcOrientation(self, pt, p, q, delta = 0): 
        '''
        Returns the orientation of the points pt, p, and q
        If the orientation is counterclockwise, then it returns -1. If the orientation is clockwise, then it return 1. If the points are collinear, then it returns 0.'
        Parameters:
        pt: Point2D
        p: Point2D
        q: Point2D
        delta: float
        '''
        det = self.determinant(pt, p, q)
        if det > delta: 
            return -1
        elif det < -delta: 
            return 1
        return 0

    @classmethod 
    def compare(self, pt, p, q, delta = 0):
        '''
        Returns the orientation of the points pt, p, and q. If the orientation is counterclockwise, then it returns -1. If the orientation is clockwise, then it return 1. If the points are collinear, then it returns 0.'
        Parameters:
        pt: Point2D
        p: Point2D
        q: Point2D
        '''
        orientation = self.calcOrientation(pt, p, q)
        if (orientation == 0):
            return self.distance(pt, p, q)
        return orientation

    @classmethod
    def hullCompute(self, points):
        '''
        Returns the convex hull of the list of 2D points. The convex hull is computed using the Graham's Scan algorithm. 
        Parameters: 
        points: list of Point2D objects
        '''
        if (len(points) == 0): 
            return []
        pointsCopy = copy.deepcopy(points)
        minimumPoint = min(pointsCopy, key = lambda p: (p.y, p.x))
        pointsCopy.remove(minimumPoint)
        sortedPoints = [minimumPoint] + sorted(pointsCopy, key = functools.cmp_to_key(lambda A, B: self.compare(minimumPoint, A, B)))
        convex_hull = deque() 
        if (len(pointsCopy) > 0): 
            convex_hull.append(sortedPoints[0])
        else: 
            return list(convex_hull)
        if (len(pointsCopy) > 1):
            convex_hull.append(sortedPoints[1])
        else:
            return list(convex_hull)
        if (len(pointsCopy) > 2): 
            convex_hull.append(sortedPoints[2])
        else:
            return list(convex_hull)

        if self.calcOrientation(convex_hull[-3], convex_hull[-2], convex_hull[-1], 0) == 0:
            convex_hull.pop()
        index = 3
        while (index < len(sortedPoints)):
            p = convex_hull[-2]
            q = convex_hull[-1]
            r = sortedPoints[index]
            if self.calcOrientation(p, q, r) == 1: 
                convex_hull.pop()
            else:
                if self.calcOrientation(p, q, r) == 0:
                    convex_hull.pop()
                convex_hull.append(r)
                index += 1
        convex_hull = list(convex_hull)
        return convex_hull