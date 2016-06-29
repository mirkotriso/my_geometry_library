# -*- coding: utf-8 -*-
import math
import numbers
from point import Point

class Vector(object):
    def __init__(self, p1, p0=Point(0,0,0)):
        self.p0 = p0  # read-write
        self.p1 = p1  # read-write
        self.x = p1.x - p0.x  # read only
        self.y = p1.y - p0.y  # read only
        if len(p0) == 2:
            self.coords = (self.x, self.y)  # read only
        if len(p0) == 3:
            self.z = p1.z - p0.z  # read only
            self.coords = (self.x, self.y, self.z)  # read only
        self.l = self.__length()

    def __hash__(self):
        return self.coords.__hash__()

    def __getitem__(self, key):
        """Treats the vector as a tuple or dict for indexes and slicing.
        """
        if key in ('x','y','z'):
            return self.asDict()[key]
        else:
            return self.coords.__getitem__(key)

    def __repr__(self):
        if len(self.coords) == 2:
            return 'Vector(%s, %s)' % self.coords
        elif len(self.coords) == 3:
            return 'Vector(%s, %s, %s)' % self.coords

    def __eq__(self, other):
        """I am allowing these to compared to tuples, and to say that yes, they
        are equal. the idea here is that a Vector3d _is_ a tuple of floats, but
        with some extra methods.
        """
        return self.coords == other.coords
		
    def __iter__(self):
        """For iterating, the vectors coordinates are represented as a tuple."""
        return self.coords.__iter__()

    def __add__(self, other):
        """we want to add single numbers as a way of changing the length of the
        vector, while it would be nice to be able to do vector addition with
        other vectors.
        """
        if isinstance(other, numbers.Number):
            # then add to the length of the vector
            # multiply the number by the normalized self, and then
            # add the multiplied vector to self
            return self.__class__(Point(*(self.normalized() * other + self).coords), self.p0)
        elif isinstance(other, self.__class__):
            # add all the coordinates together
            # there are probably more efficient ways to do this
            return self.__class__(Point(*(sum(p) for p in zip(self.p1, other.coords))), self.p0)
        else:
            raise TypeError(
                    "unsupported operand (+/-) for types %s and %s" % (
                        self.__class__, type(other)))

    def __sub__(self, other):
        """Subtract a vector or number
        """
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
            new_point = Point(*[x * other for x in self.coords])
            return self.__class__(new_point)

        elif isinstance(other, self.__class__):
            # dot product for other vectors
            return self.dot(other)
        else:
            raise TypeError(
                    "unsupported operand (multiply/divide) for types %s and %s" % (
                        self.__class__, type(other)))
		
    def __length(self):
        """get the vector length / amplitude
        """
        # only calculate the length if asked to.
        return math.sqrt(sum(n**2 for n in self.coords))
        
    def __move(self, new_p0):
        """move the application point p0"""
        return 
		
    def normalized(self):
        """just returns the normalized version of self without editing self in
        place.
        """
        return self * (1 / self.__l)  # define length or use it as function
		
    def dot(self, other):
        """Gets the dot product of this vector and another.
        """
        return sum((a * b) for a, b in zip(self.coords, other.coords))
        
    def asDict(self):
        """return dictionary representation of the vector"""
        return dict( zip( list('xyz'), self.coords ) )

    def cross(self, other):
        """Gets the cross product between two vectors
        """
        x = (self[1] * other[2]) - (self[2] * other[1])
        y = (self[2] * other[0]) - (self[0] * other[2])
        z = (self[0] * other[1]) - (self[1] * other[0])
        return self.__class__(Point(x, y, z))