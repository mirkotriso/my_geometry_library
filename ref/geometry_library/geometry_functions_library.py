# -*- coding: utf-8 -*-
import numpy as np


def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. Euclidean norm, along axis.

    >>> v0 = numpy.random.random(3)
    >>> v1 = unit_vector(v0)
    >>> numpy.allclose(v1, v0 / numpy.linalg.norm(v0))
    True
    >>> v0 = numpy.random.rand(5, 4, 3)
    >>> v1 = unit_vector(v0, axis=-1)
    >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=2)), 2)
    >>> numpy.allclose(v1, v2)
    True
    >>> v1 = unit_vector(v0, axis=1)
    >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=1)), 1)
    >>> numpy.allclose(v1, v2)
    True
    >>> v1 = numpy.empty((5, 4, 3))
    >>> unit_vector(v0, axis=1, out=v1)
    >>> numpy.allclose(v1, v2)
    True
    >>> list(unit_vector([]))
    []
    >>> list(unit_vector([1]))
    [1.0]

    """
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= np.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data*data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


def rotation_matrix(angle, direction, point=None):
    """Return matrix to rotate about axis defined by point and direction.

    >>> R = rotation_matrix(math.pi/2, [0, 0, 1], [1, 0, 0])
    >>> numpy.allclose(numpy.dot(R, [0, 0, 0, 1]), [1, -1, 0, 1])
    True
    >>> angle = (random.random() - 0.5) * (2*math.pi)
    >>> direc = numpy.random.random(3) - 0.5
    >>> point = numpy.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(-angle, -direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> I = numpy.identity(4, numpy.float64)
    >>> numpy.allclose(I, rotation_matrix(math.pi*2, direc))
    True
    >>> numpy.allclose(2, numpy.trace(rotation_matrix(math.pi/2,
    ...                                               direc, point)))
    True

    """
    sina = np.sin(angle)
    cosa = np.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array([[0.0, -direction[2], direction[1]],
                   [direction[2], 0.0, -direction[0]],
                   [-direction[1], direction[0], 0.0]])
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M


def translation_matrix(direction):
    """Return matrix to translate by direction vector.

    >>> v = numpy.random.random(3) - 0.5
    >>> numpy.allclose(v, translation_matrix(v)[:3, 3])
    True

    """
    M = np.identity(4)
    M[:3, 3] = direction[:3]
    return M


def unit_normal(a, b, c):
    x = np.linalg.det([[1, a[1], a[2]],
                       [1, b[1], b[2]],
                       [1, c[1], c[2]]])
    y = np.linalg.det([[a[0], 1, a[2]],
                       [b[0], 1, b[2]],
                       [c[0], 1, c[2]]])
    z = np.linalg.det([[a[0], a[1], 1],
                       [b[0], b[1], 1],
                       [c[0], c[1], 1]])
    magnitude = (x**2 + y**2 + z**2)**.5
    return (x/magnitude, y/magnitude, z/magnitude)


def normalize_v3(arr):
    """
    Normalize a numpy array of 3 component vectors with shape = (n,3)

    INPUTS:
    - arr: an array with dimensions (n,3)

    OUTPUTS:
    - arr: the normalised input array
    """
    lens = np.sqrt(arr[:, 0]**2 + arr[:, 1]**2 + arr[:, 2]**2)
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens

    return arr


def Normals(tris):
    """Calculate the normal for all the triangles by taking the cross product
    of the vectors v1-v0, and v2-v0 in each triangle"""
    ni = - np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    n = normalize_v3(ni)

    return n


def panels_area(tris):
    """
    The function returns the area of each panel comprising the panelisation.
    The normal to each panel is computed; the area of the triangle corresponds 
    to the half of the module of the normal.
    
    Parameters
    ----------
    tris:     
        triplets explicitly representing the coordinates of the vertices
        of each triangular face as obtained from the Panelisation function
                
    Returns
    -------
    Area:     
        array containing the area of each face in the panelisation.
        
    Info
    -----
    Author: 
        Mirko Trisolini     
    Change Log: 
        - Docstring creation: 02/10/15  
    """
    Ni = np.cross( tris[::,1 ] - tris[::,0]  , tris[::,2 ] - tris[::,0] )
    Area = np.sqrt(Ni[:,0]**2 + Ni[:,1]**2 + Ni[:,2]**2) * 0.5
    
    return Area
 
   
def panels_area_alternative(tris):
    """
    The function returns the area of each panel of the panelisation.
    The function distinguish between panelisation made with triangles or 
    rectangles.
    
    Parameters
    ----------
    tris:     
        triplets explicitly representing the coordinates of the vertices
        of each triangular face as obtained from the Panelisation function
                
    Returns
    -------
    Area:     
        array containing the area of each face in the panelisation.
        
    Info
    -----
    Author: 
        Mirko Trisolini     
    Change Log: 
        - Docstring creation: 02/10/15  
    """
    
    if len(tris[0]) is 3:
        Ni = np.cross( tris[::,1 ] - tris[::,0]  , tris[::,2 ] - tris[::,0] )
        Area = np.sqrt(Ni[:,0]**2 + Ni[:,1]**2 + Ni[:,2]**2) * 0.5
    elif len(tris[0]) is 4:
        Area = []
        for i in xrange(len(tris)):
            a = ((tris[i][0][0] - tris[i][1][0])**2 + (tris[i][0][1] - \
                tris[i][1][1])**2 + (tris[i][0][2] - tris[i][1][2])**2)**0.5
            b = ((tris[i][1][0] - tris[i][2][0])**2 + (tris[i][1][1] - \
                tris[i][2][1])**2 + (tris[i][1][2] - tris[i][2][2])**2)**0.5
            c = ((tris[i][0][0] - tris[i][2][0])**2 + (tris[i][0][1] - \
                tris[i][2][1])**2 + (tris[i][0][2] - tris[i][2][2])**2)**0.5
            d = ((tris[i][2][0] - tris[i][3][0])**2 + (tris[i][2][1] - \
                tris[i][3][1])**2 + (tris[i][2][2] - tris[i][3][2])**2)**0.5
            if d <= 1e-5:
                p = 0.5 *(a + b + c)
                area = np.sqrt(p *(p-a)*(p-b)*(p-c)) 
            else:
                area = a * b  
            Area.append(area)
    
    return Area


class Line(object):
    def __init__(self, point_1, point_2):
        self.__x1 = point_1[0]
        self.__y1 = point_1[1]
        self.__z1 = point_1[2]
        self.__x2 = point_2[0]
        self.__y2 = point_2[1]
        self.__z2 = point_2[2]
        self.length = np.sqrt((self.__x2 - self.__x1)**2
                              + (self.__y2 - self.__y1)**2
                              + (self.__z2 - self.__z1)**2)
        self.direction = np.array([(self.__x2 - self.__x1),
                                   (self.__y2 - self.__y1),
                                   (self.__z2 - self.__z1)]) / self.length
        
        
class Plane(object):
    def __init__(self, point, normal):
        self.a = normal[0]
        self.b = normal[1]
        self.c = normal[2]
        self.n = normal
        self.d = np.dot(normal, point)

    def intersection(self, point, direction):
        if np.dot(self.n, direction) == 0:
            return []
        else:
            t = (self.d - np.dot(self.n, point)) / np.dot(self.n, direction)
            if t < 0:
                return []
        r = point + t * direction
        return r

    def transform_matrix(self):
        '''computes intersection between plane and plane z=0'''
        if np.dot(self.n, [0, 0, 1]) == 1 or np.dot(self.n, [0, 0, 1]) == -1:
            return translation_matrix(-self.d*self.n)
        else:
            if self.a == 0:
                d = [1, 0, 0]
                P = [0, float(self.d) / self.b, 0]
            elif self.b == 0:
                d = [0, 1, 0]
                P = [float(self.d) / self.a, 0, 0]
            else:
                d = [-float(self.b) / self.a, 1, 0]
                P = [float(self.d) / self.a, 0, 0]
            rot_angle = np.pi - np.arccos(np.dot([0, 0, 1], self.n))
            return rotation_matrix(rot_angle, d, point=P)

    def is_in_plane(self, point):
        if self.a*point[0] + self.b*point[1] + self.c*point[2] - self.d == 0:
            return True
        else:
            return False

    def point_distance(self, point):
        d = abs(self.a * point[0] + self.b * point[1] + self.c * point[2] -
                self.d) / np.sqrt(self.a**2 + self.b**2 + self.c**2)
        return d
