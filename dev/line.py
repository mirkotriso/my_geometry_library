#from point import Point
from vector_v2 import Vector

class Line(object):
    def __init__(self, p1, *args, **kw):
        if isinstance(args[0], Point):
            self.__p1 = p1
        else:
            raise TypeError()
        if isinstance(args[0], Point):
            self.__p2 = args[0]
            self.__direction = Vector(self.__p2, self.__p1).normalized()
        elif isinstance(args[0], Vector):
            self.__direction = args[0]
        else:
            raise TypeError()			

    @property
    def p1(self):
        return self.__p1

    @property
    def p2(self):
        return self.__p2
        
    @property
    def direction(self):
        return self.__direction


class Segment(Line):
    def __init__(self, *args, **kw):
        super(Segment, self).__init__(*args, **kw)
        self.length = self.__length()
        
    def __length(self, p1=None, p2=None):
        if self.p1 and self.p2:
            return self.p1.distance_to(self.p2)
        #return						   
							   