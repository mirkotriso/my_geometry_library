class Point(object):
    def __init__(self, *args):
        self.__x = args[0]
        self.__y = args[1]
        if len(args) > 2:
            self.__z = args[2]
        self.__coords = (args)

    @property
    def x(self):
        return self.__x
    
    @property
    def y(self):
        return self.__y

    @property
    def z(self):
        return self.__z

    @property
    def coords(self):
        return self.__coords

    def __len__(self):
        return len(self.coords)

    def __add__(self, p):
        return Point(*[a + b for a, b in zip(self.coords, p.coords)])
        
    def __sub__(self, other):
        """Subtract a vector or number
        """
        return self.__add__(other * -1)

    def __repr__(self):
        if len(self.coords) == 2:
            return 'Point(%s, %s)' % self.coords
        elif len(self.coords) == 3:
            return 'Point(%s, %s, %s)' % self.coords

    def __iter__(self):
        """For iterating, the vectors coordinates are represented as a tuple."""
        return self.coords.__iter__()

    def distance_to(self, other):
        """Find the distance between this point and another.
        """
        #if isinstance(other, Plane):
         #   # get distance between point and plane
         #   pass
        #elif isinstance(other, Line):
            # get distance between point and line
         #   pass
        #else:
        return (other - self).l
