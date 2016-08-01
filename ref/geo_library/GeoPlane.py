from GeoVector import *

class GeoPlane(object):
    
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d    

    @staticmethod
    def Create(p0, p1, p2): # p0, p1, p2: GeoPoint
        
        v = GeoVector(p0, p1)
        u = GeoVector(p0, p2)

        # normal vector
        n = u * v;

        a = n.x
        b = n.y
        c = n.z
        d = - (a * p0.x + b * p0.y + c * p0.z)

        return GeoPlane(a, b, c, d)

    def __neg__(self):
        return GeoPlane(-self.a, -self.b, -self.c, -self.d)
    
    def __mul__(self, pt): # pt: GeoPoint
        return (self.a * pt.x + self.b * pt.y + self.c * pt.z + self.d)