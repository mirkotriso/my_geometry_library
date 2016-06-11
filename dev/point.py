class Point2D(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.coords = (x, y)
    def __add__(self, p):
        return Point2D(self.x + p.x, self.y + p.y)


class Point3D(object):
    def __init__(self, x, y, z):
	    self.x = x
	    self.y = y
	    self.z = z
	    self.coords = (x, y, z)
    def __len__(self):
        return len(self.coords)
    def __add__(self, p):
        return Point3D(self.x + p.x, self.y + p.y, self.z + p.z)

		
#def __repr__(self):
#    return 'Point2D(%s, %s)' % self.coords
#def __repr__(self):
#    return 'Point3D(%s, %s, %s)' % self.coords