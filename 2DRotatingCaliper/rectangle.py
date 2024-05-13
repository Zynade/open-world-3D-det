class Rectangle: 
    def __init__(self, A, B, C, D):
        self.vertices = [A, B, C, D]

    def area(self):
        '''
        Returns the area of the rectangle
        '''
        return abs((self.vertices[1].x - self.vertices[0].x) * (self.vertices[3].y - self.vertices[0].y))
    
    def perimeter(self):
        '''
        Returns the perimeter of the rectangle
        '''
        return 2 * ((self.vertices[1].x - self.vertices[0].x) + (self.vertices[3].y - self.vertices[0].y))
    
    def getEdges(self):
        '''
        Returns the edges of the rectangle
        '''
        edges = []
        for i in range(1, len(self.vertices)):
            edges.append((self.vertices[i-1], self.vertices[i]))
        edges.append((self.vertices[-1], self.vertices[0]))
        return edges
    