import numbers
import numpy as np


class Point(object):
    def __init__(self, *args):
        if isinstance(args[0], list) or isinstance(args[0], tuple):
            temp = args[0]
            if isinstance(temp, list):
                temp = tuple(temp)
            self.__x, self.__y = temp[:2]
            if len(temp) > 2:
                self.__z = temp[2]
            self.__coords = (temp)
        else:
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
        
    def __hash__(self):
        return self.coords.__hash__()

    def __getitem__(self, key):
        """Treats the vector as a tuple or dict for indexes and slicing.
        """
        if key in ('x','y','z'):
            return self.asDict()[key]
        else:
            return self.coords.__getitem__(key)

    def __add__(self, other):
        if isinstance(other, numbers.Number):
            # then add to the length of the vector
            # multiply the number by the normalized self, and then
            # add the multiplied vector to self
            return self.__class__(*[a + other for a in self.coords])
        elif isinstance(other, self.__class__):
           return self.__class__(*[a + b for a, b in zip(self.coords, other.coords)])
        else:
            raise TypeError(
                    "unsupported operand (+/-) for types %s and %s" % (
                        self.__class__, type(other)))

    def __sub__(self, other):
        return self.__add__(other * -1)
        
    def __mul__(self, other):
        """if with a number, then scalar multiplication of the vector,
            if with a Vector, then dot product, I guess for now, because
            the asterisk looks more like a dot than an X.
            >>> v2 = Vector3d(-4.0, 1.2, 3.5)
            >>> v1 = Vector3d(2.0, 1.1, 0.0)
            >>> v2 * 1.25
            Vector3d(-5.0, 1.5, 4.375)
            >>> v2 * v1 #dot product
            -6.6799999999999997
        """
        if isinstance(other, numbers.Number):
            # scalar multiplication for numbers
            new_point = [x * other for x in self.coords]
            return self.__class__(new_point)

    def __repr__(self):
        if len(self.coords) == 2:
            return 'Point(%s, %s)' % self.coords
        elif len(self.coords) == 3:
            return 'Point(%s, %s, %s)' % self.coords

    def __iter__(self):
        """For iterating, the vectors coordinates are represented as a tuple."""
        return self.coords.__iter__()

    def distance_to(self, other):
        if isinstance(other, self.__class__):
            return np.sqrt(np.sum([(abs(a - b))**2 for a, b in zip(self.coords, other.coords)]))
        elif isinstance(other, Line):
            return 0
        elif isinstance(other, Plane):
            return 0
            
    def belongs_to(self, other):
        #if isinstance(other, line.Line):
        """va connesso a distance_to"""
        print other, other.p1, other.p2
        v1 = Vector(other.p2 - other.p1)
        v2 = Vector(other.p1 - self)
        v3 = Vector(other.p2 - other.p1)
        v = v1.cross(v2)
        return v.length / v3.length

          
class PointCollection(object):
    def __init__(self, data, *args):
        self.points = []
        for p in data:
            if isinstance(p, Point):
                self.points.append(p)
            elif isinstance(p, list) or isinstance(p, tuple):
                self.points.append(Point(p))
    def __len__(self):
        return len(self.points)
        
    def __eq__(self, other):
        """I am allowing these to compared to tuples, and to say that yes, they
        are equal. the idea here is that a Vector3d _is_ a tuple of floats, but
        with some extra methods.
        """
        return self.points == other.points
		
    def __iter__(self):
        """For iterating, the vectors coordinates are represented as a tuple."""
        return self.points.__iter__()
            
    def __hash__(self):
        return self.points.__hash__()

    def __getitem__(self, key):
        """Treats the vector as a tuple or dict for indexes and slicing.
        """
        return self.points.__getitem__(key)
        
    def __add__(self, other):
        return self.__class__(other.points + self.points)
        
    def distance_to(self, other):
        if isinstance(other, Point):
            return [p.distance_to(other) for p in self.points]
        elif isinstance(other, self.__class__):
            if len(other) != len(self):
                raise ValueError("The vector collections need to have the same dimension")
            return [p.distance_to(x) for p, x in zip(self.points, other.points)]
