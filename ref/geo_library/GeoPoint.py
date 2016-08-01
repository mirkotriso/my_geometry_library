class GeoPoint(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    def __add__(self, p):
        return GeoPoint(self.x + p.x, self.y + p.y, self.z + p.z)