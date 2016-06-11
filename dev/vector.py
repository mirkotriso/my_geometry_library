import math
import numbers

class Vector(object):
    def __init__(self, p0, p1):
        self.__p0 = p0  # read-write
        self.__p1 = p1  # read-write
        if len(p0) == 2:
            self.__x = p1.x - p0.x  # read only
            self.__y = p1.y - p0.y  # read only
            self.__coords = (self.x, self.y)  # read only
        if len(p0) == 3:
            self.__x = p1.x - p0.x  # read only
            self.__y = p1.y - p0.y  # read only
            self.__z = p1.z - p0.z  # read only
            self.__coords = (self.x, self.y, self.z)  # read only

    @property
    def p0(self):
        return self.__p0
    @p0.setter
    def p0(self, p0):
        self.__p0 = p0
        if len(p0) == 2:
            self.__x = self.p1.x - p0.x
            self.__y = self.p1.y - p0.y
        if len(p0) == 3:
            self.__x = self.p1.x - p0.x
            self.__y = self.p1.y - p0.y
            self.__z = self.p1.z - p0.z

    @property
    def p1(self):
        return self.__p1
    @p1.setter
    def p1(self, p1):
        self.__p1 = p1
        if len(p1) == 2:
            self.__x = self.p1.x - p0.x
            self.__y = self.p1.y - p0.y
        if len(p1) == 3:
            self.__x = self.p1.x - p0.x
            self.__y = self.p1.y - p0.y
            self.__z = self.p1.z - p0.z

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

    def __eq__(self, other):
        """I am allowing these to compared to tuples, and to say that yes, they
        are equal. the idea here is that a Vector3d _is_ a tuple of floats, but
        with some extra methods.
        """
        return self.coords == other.coords
		
    def __iter__(self):
        """For iterating, the vectors coordinates are represented as a tuple."""
        return self.coords.__iter__()
		
    def length(self):
        """get the vector length / amplitude
        """
        # only calculate the length if asked to.
        return math.sqrt(sum(n**2 for n in self.coords))
		
    def normalized(self):
        """just returns the normalized version of self without editing self in
        place.
        """
        return self * (1 / self.length)  # define length or use it as function
		
    def dot(self, other):
        """Gets the dot product of this vector and another.
        """
        return sum((a * b) for a, b in zip(self.coords, other.coords))
		
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
            return self.__class__( *((n * other) for n in self))

        elif isinstance(other, self.__class__):
            # dot product for other vectors
            return self.dot(other)
        else:
            raise TypeError(
                    "unsupported operand (multiply/divide) for types %s and %s" % (
                        self.__class__, type(other)))
			