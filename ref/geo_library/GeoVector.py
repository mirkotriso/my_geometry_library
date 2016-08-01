from GeoPoint import *

class GeoVector(object):

    def __init__(self, p0, p1): # GeoPoint p0, p1        
        self.__p0 = p0 # read write (get set)
        self.__p1 = p1 # read write (get set)
        self.__x = p1.x - p0.x # read only
        self.__y = p1.y - p0.y # read only       
        self.__z = p1.z - p0.z # read only
        
    @property
    def p0(self):        
        return self.__p0
    @p0.setter
    def p0(self, p0):        
        self.__p0 = p0
        self.__x = self.p1.x - p0.x
        self.__y = self.p1.y - p0.y
        self.__z = self.p1.z - p0.z
        
    @property    
    def p1(self):        
        return self.__p1
    @p1.setter
    def p1(self, p1):        
        self.__p1 = p1
        self.__x = p1.x - self.p0.x
        self.__y = p1.y - self.p0.y
        self.__z = p1.z - self.p0.z

    @property     
    def x(self):                
        return self.__x
    
    @property
    def y(self):
        return self.__y
    
    @property
    def z(self):
        return self.__z
    
    def __mul__(self, v): # v: GeoVector        
        x = self.y * v.z - self.z * v.y
        y = self.z * v.x - self.x * v.z
        z = self.x * v.y - self.y * v.x
        p0 = v.p0
        p1 = p0 + GeoPoint(x, y, z)
        return GeoVector(p0, p1)