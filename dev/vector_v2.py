# -*- coding: utf-8 -*-
import numbers
import numpy as np
from point import Point
        
        
class Vector(object):
    def __init__(self, p1, p0=Point(0,0,0)):
        self.__p0 = p0  # read-write
        self.__p1 = p1  # read-write
        self.__x = p1.x - p0.x  # read only
        self.__y = p1.y - p0.y  # read only
        if len(p1) == 2:
            self.__coords = (self.__x, self.__y)
        if len(p1) == 3:
            self.__z = p1.z - p0.z  # read only
            self.__coords = (self.__x, self.__y, self.__z)
        self.__length = self.__get_length()

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
    def p0(self):
        return self.__p0
    @p0.setter
    def p0(self, value):
        self.__p0 = value
        self.__init__(self.p1, self.p0)

    @property
    def p1(self):
        return self.__p1
    @p1.setter
    def p1(self, value):
        self.__p1 = value
        self.__init__(self.p1, self.p0)

    @property
    def coords(self):
        return self.__coords
        
    @property
    def length(self):
        return self.__length
        
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
		
    def __get_length(self):
        """get the vector length / amplitude
        """
        # only calculate the length if asked to.
        return np.sqrt(sum(n**2 for n in self.coords))
        
    def __move(self, new_p0):
        """move the application point p0"""
        return self.__class__(self.p1 + new_p0, new_p0)
		
    def normalized(self):
        """just returns the normalized version of self without editing self in
        place.
        """
        return self * (1 / self.length)  # define length or use it as function

    def to_length(self, number):
        """Get a parallel vector with the input amplitude."""
        # depends on normalized() and __mul__
        # create a vector as long as the number
        return self.normalized() * number
		
    def dot(self, other):
        """Gets the dot product of this vector and another.
        """
        return sum((a * b) for a, b in zip(self.coords, other.coords))  # sum(self.coords * other.coords)
        
    def asDict(self):
        """return dictionary representation of the vector"""
        return dict( zip( list('xyz'), self.coords ) )

    def cross(self, other):
        """
        Gets the cross product between two vectors.
        The new point is applied to the origin of the axis.
        """
        x = (self[1] * other[2]) - (self[2] * other[1])
        y = (self[2] * other[0]) - (self[0] * other[2])
        z = (self[0] * other[1]) - (self[1] * other[0])
        return self.__class__(Point(x, y, z))
        
    def angle_to(self, other):
        """computes the angle between two vectors
            cos theta = (n * m) / (n.length * m.length)
        """
        arc = self.dot(other) / self.length / other.length
        if abs(arc - 1) <= 1e-6 or abs(arc + 1) <= 1e-6:
            arc = 1
        return np.acos(arc)


class VectorCollection(object):
    def __init__(self, data, *args):
        if args:
            self.vectors = [Vector(Point(p1),
                                   Point(p0)) for p0, p1 in zip(data, args[0])]
        else:
            self.vectors = []
            for v in data:
                if isinstance(v, Vector):
                    self.vectors.append(v)
                elif isinstance(v, Point):
                    self.vectors.append(Vector(v))
                elif isinstance(v, list) or isinstance(v, tuple):
                    self.vectors.append(Vector(Point(v)))
            
    def __len__(self):
        return len(self.vectors)
        
    def __eq__(self, other):
        """I am allowing these to compared to tuples, and to say that yes, they
        are equal. the idea here is that a Vector3d _is_ a tuple of floats, but
        with some extra methods.
        """
        return self.vectors == other.vectors
		
    def __iter__(self):
        """For iterating, the vectors coordinates are represented as a tuple."""
        return self.vectors.__iter__()
            
    def __hash__(self):
        return self.vectors.__hash__()

    def __getitem__(self, key):
        """Treats the vector as a tuple or dict for indexes and slicing.
        """
        return self.vectors.__getitem__(key)
        
    def __add__(self, other):
        return self.__class__(other.vectors + self.vectors)
        
    def normalized(self):
        return self.__class__([v.normalized().coords for v in self.vectors])
        
    def angle_to(self, other):
        if isinstance(other, Vector):
            return [v.angle_to(other) for v in self.vectors]
        elif isinstance(other, self.__class__):
            if len(other) != len(self):
                raise ValueError("The vector collections need to have the same dimension")
            return [v.angle_to(x) for v, x in zip(self.vectors, other.vectors)]

    def dot(self, other):
        if isinstance(other, Vector):
            return [v.dot(other) for v in self.vectors]
        elif isinstance(other, self.__class__):
            if len(other) != len(self):
                raise ValueError("The vector collections need to have the same dimension")
            return [v.dot(x) for v, x in zip(self.vectors, other.vectors)]

    def cross(self, other):
        if isinstance(other, Vector):
            return [v.cross(other) for v in self.vectors]
        elif isinstance(other, self.__class__):
            if len(other) != len(self):
                raise ValueError("The vector collections need to have the same dimension")
            return [v.cross(x) for v, x in zip(self.vectors, other.vectors)]

        