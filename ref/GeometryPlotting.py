# -*- coding: utf-8 -*-
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
import matplotlib.colors as colors
import matplotlib.cm as cmx

class Segment:
    def __init__(self, angle, prev_angle, wrap_around):
        self.angle = angle
        self.length = abs(angle - prev_angle + \
                          (2*math.pi if wrap_around else 0))
        self.num_points = 0

    def sub_length(self):
        return self.length / (self.num_points + 1)

    def next_sub_length(self):
        return self.length / (self.num_points + 2)

    def add_point(self):
        self.num_points += 1

def distribute(angles, n):
    # No points given? Evenly distribute them around the circle
    if len(angles) == 0:
        return [2*math.pi / n * i - math.pi for i in xrange(n)]

    # Sort the angles and split the circle into segments
    s, pi, ret = sorted(angles), math.pi, []
    segments = [Segment(s[i], s[i-1], i == 0) for i in xrange(len(s))]

    # Calculate the length of all subsegments if the point
    # would be added; take the largest to add the point to
    for _ in xrange(n):
        max(segments, key = lambda x: x.next_sub_length()).add_point()

    # Split all segments and return angles of the points
    for seg in segments:
        for k in xrange(seg.num_points):
            a = seg.angle - seg.sub_length() * (k + 1)
            # Make sure all returned values are between -pi and +pi
            ret.append(a - 2*pi if a > pi else a + 2*pi if a < -pi else a)

    return ret

def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. Euclidean norm, along axis.

    >>> v0 = np.random.random(3)
    >>> v1 = unit_vector(v0)
    >>> np.allclose(v1, v0 / np.linalg.norm(v0))
    True
    >>> v0 = np.random.rand(5, 4, 3)
    >>> v1 = unit_vector(v0, axis=-1)
    >>> v2 = v0 / np.expand_dims(np.sqrt(np.sum(v0*v0, axis=2)), 2)
    >>> np.allclose(v1, v2)
    True
    >>> v1 = unit_vector(v0, axis=1)
    >>> v2 = v0 / np.expand_dims(np.sqrt(np.sum(v0*v0, axis=1)), 1)
    >>> np.allclose(v1, v2)
    True
    >>> v1 = np.empty((5, 4, 3))
    >>> unit_vector(v0, axis=1, out=v1)
    >>> np.allclose(v1, v2)
    True
    >>> list(unit_vector([]))
    []
    >>> list(unit_vector([1]))
    [1.0]

    """
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
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
    >>> np.allclose(np.dot(R, [0, 0, 0, 1]), [1, -1, 0, 1])
    True
    >>> angle = (random.random() - 0.5) * (2*math.pi)
    >>> direc = np.random.random(3) - 0.5
    >>> point = np.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(-angle, -direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> I = np.identity(4, np.float64)
    >>> np.allclose(I, rotation_matrix(math.pi*2, direc))
    True
    >>> np.allclose(2, np.trace(rotation_matrix(math.pi/2,
    ...                                               direc, point)))
    True

    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array([[ 0.0,         -direction[2],  direction[1]],
                      [ direction[2], 0.0,          -direction[0]],
                      [-direction[1], direction[0],  0.0]])
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M

def translation_matrix(direction):
    """Return matrix to translate by direction vector.

    >>> v = np.random.random(3) - 0.5
    >>> np.allclose(v, translation_matrix(v)[:3, 3])
    True

    """
    M = np.identity(4)
    M[:3, 3] = direction[:3]
    return M


def CreateBox(center, l, w, h):
    
    xc, yc, zc = center
    x1 = xc - 0.5 * l
    x2 = xc + 0.5 * l
    y1 = yc - 0.5 * w
    y2 = yc + 0.5 * w
    z1 = zc - 0.5 * h
    z2 = zc + 0.5 * h
    
    points = np.array([[x1,y1,z1], \
                       [x2,y1,z1], \
                       [x2,y2,z1], \
                       [x1,y2,z1], \
                       [x1,y1,z2], \
                       [x2,y1,z2], \
                       [x2,y2,z2], \
                       [x1,y2,z2]])
      
    bars = np.array([[0,1], [1,2], [2,3], \
                    [3,0], [0,4], [1,5], \
                    [2,6], [3,7], [4,5], \
                    [5,6], [6,7], [7,4]])
    
    faces = np.array([[0,1,2,3], \
                      [0,4,5,1], \
                      [2,1,5,6], \
                      [2,6,7,3], \
                      [0,3,7,4], \
                      [4,7,6,5]])
                      
    return points, bars, faces
    

def PlotPenProb(points, faces, Pp, newFig, Title=None, colormap='autumn', saveFig='on', name=None, path=None):        
    
    
    cm = plt.get_cmap(colormap)    
    y_cmap = cm(np.arange(256))
    x_cmap = np.linspace(np.log(min(Pp)), 0, len(y_cmap))
    cNorm_cb  = colors.LogNorm(vmin=min(Pp), vmax=1)
    scalarMap_cb = cmx.ScalarMappable(norm=cNorm_cb, cmap=cm)    
    
    
    if newFig[0] is True:
        size = (10, 8)
        fig = plt.figure(figsize=size, dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_aspect("auto")
        ax.set_autoscale_on(True)
        ax.set_axis_off()
        plt.tight_layout(pad=2)
        ax.set_xlim3d([min(points[:,0])-1, max(points[:,0])+1])
        ax.set_ylim3d([min(points[:,1])-1, max(points[:,1])+1])
        ax.set_zlim3d([min(points[:,2])-1, max(points[:,2])+1])
    
    elif newFig[0] is False:
        ax = newFig[1]
    else:
        raise ValueError("Only True or False value accepted")
    
    if Title is None:
        plt.title('', fontsize=16)
    else:
        plt.title(Title[0], fontsize=Title[1])    

    plt.tick_params(axis='both', which='major', labelsize=14)
    
    for j in xrange(len(faces)):
    
        xk = 0
        
        for i in xrange(len(x_cmap)-1):    
            if np.log(Pp[j]) >= x_cmap[i] and np.log(Pp[j]) <= x_cmap[i+1]:
                xk = i
                break
            elif Pp[j] == x_cmap[i+1]:
                xk = len(x_cmap)-1
                break 
              
        if xk is len(x_cmap)-1:
            R = y_cmap[xk][0]
            G = y_cmap[xk][1]
            B = y_cmap[xk][2]
        else:
            R = ((np.log(Pp[j]) - x_cmap[xk]) / (x_cmap[xk+1]-x_cmap[xk]) * (y_cmap[xk+1][0]-y_cmap[xk][0])) + y_cmap[xk][0]
            G = ((np.log(Pp[j]) - x_cmap[xk]) / (x_cmap[xk+1]-x_cmap[xk]) * (y_cmap[xk+1][1]-y_cmap[xk][1])) + y_cmap[xk][1]
            B = ((np.log(Pp[j]) - x_cmap[xk]) / (x_cmap[xk+1]-x_cmap[xk]) * (y_cmap[xk+1][2]-y_cmap[xk][2])) + y_cmap[xk][2]
        

        color_face_j = points[faces[j]]
        if np.size(color_face_j, 1) > 3:
            color_face_j = np.delete(color_face_j, 3, 1)
        else:
            pass
        side_j = art3d.Poly3DCollection([color_face_j])
        side_j.set_color([R,G,B,1])
        ax.add_collection3d(side_j)
    
    scalarMap_cb._A = []
    cb = plt.colorbar(scalarMap_cb, shrink=.65, pad=.002, aspect=15)
    cb.set_label('Penetration Probability', fontsize=16, labelpad=10)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(14)
    
    if saveFig == 'on':
        plt.savefig(path+name+".png", format="png", bbox_inches='tight')
        plt.savefig(path+name+".eps", format="eps", bbox_inches='tight')
    elif saveFig == 'off': 
        pass
    else:
        raise ValueError("multiPlot is a boolean variable. Only accepts 'on' and 'off' as inputs")
    
    return ax


def PlotGeometry(points, bars, newFig, color='k', label='label_1'):
    
    if newFig[0] is True:
        size = (10, 10)
        fig = plt.figure(figsize=size, dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_aspect("auto")
        ax.set_autoscale_on(True)
        #ax.set_axis_off()
        plt.tight_layout(pad=2)
        ax.set_xlim3d([min(points[:,0])-1, max(points[:,0])+1])
        ax.set_ylim3d([min(points[:,1])-1, max(points[:,1])+1])
        ax.set_zlim3d([min(points[:,2])-1, max(points[:,2])+1])
    elif newFig[0] is False:
        ax = newFig[1]
    else:
        raise ValueError("Only True or False value accepted")
        
    Xk = []
    Yk = []
    Zk = []

    for i in xrange(len(bars)):
        xk = [points[bars[i,0], 0], points[bars[i,1], 0]]
        yk = [points[bars[i,0], 1], points[bars[i,1], 1]]
        zk = [points[bars[i,0], 2], points[bars[i,1], 2]]
        Xk.append(xk)
        Yk.append(yk)
        Zk.append(zk)


    for xb, yb, zb in zip(Xk, Yk, Zk):
        ax.plot(xb, yb, zb, color, linewidth=1.5, label=label)   
      
    return ax


def moveGeometry(points, rot_angles, rot_dir, rot_points, tr_vector, faces, bars):
    from numpy import radians, dot, array
    new_points = []
    M = []
    for j in xrange(len(rot_angles)):
        Mj = rotation_matrix(radians(rot_angles[j]), rot_dir[j], point=rot_points[j])
        M.append(Mj)
    
    T = translation_matrix(tr_vector)
    
    for i in xrange(len(points)):
        points_i = list(points[i])
        points_i.append(1)
        
        for k in xrange(len(M)):
            points_temp = dot(M[k], points_i)
            points_i = points_temp
        
        points_i = dot(T, points_i)
        points_i = list(points_i)
        del points_i[-1]
        new_points.append(points_i)
        
    new_points = array(new_points)
    
    return new_points, faces, bars  # MODIFICARE
    
    
def createCylinder(center, l, r, n):
    
    c_center1 = [(center[0], center[1], center[2] - 0.5*l)]
    c_center2 = [(center[0], center[1], center[2] + 0.5*l)]
    
    x1 = r*np.cos(distribute([], n)) + center[0]
    y1 = r*np.sin(distribute([], n)) + center[1]
    z1 = [c_center1[0][2]]*n

    x2 = r*np.cos(distribute([], n)) + center[0]
    y2 = r*np.sin(distribute([], n)) + center[1]
    z2 = [c_center2[0][2]] * n
    
    p1 = zip(x1,y1,z1)
    p2 = zip(x2,y2,z2)
    
    c_points = np.array(c_center1 + p1 + c_center2 + p2)
    
    c_bars = []
    c_faces = []
    for i in xrange(len(c_points)):
        
        l = len(c_points)
        if i == 0 and i == l*0.5:
            pass
        elif i == 1:
            bi = [i, i+1]
            bii = [i, i+0.5*l]
            biii = [i, 0]
            c_bars.append(bi)
            c_bars.append(bii)
            c_bars.append(biii)

            fi = [i, i+1, 0, 0]
            fii = [i, 0.5*l-1, l-1, i+0.5*l]
            c_faces.append(fi)
            c_faces.append(fii)

        elif i < l*0.5-1 and i > 0:
            bi = [i, i+1]
            bii = [i, i+0.5*l]
            biii = [i, 0]
            c_bars.append(bi)
            c_bars.append(bii)
            c_bars.append(biii)

            fi = [i, i+1, 0, 0]
            fii = [i, i-1, i-1+0.5*l, i+0.5*l]
            c_faces.append(fi)
            c_faces.append(fii)

        elif i == l*0.5-1:
            bi = [i, 1]
            bii = [i, i+0.5*len(c_points)]
            biii = [i, 0]
            c_bars.append(bi)
            c_bars.append(bii)
            c_bars.append(biii)

            fi = [i, 1, 0, 0]
            fii = [i, i-1, i-1+0.5*l, i+0.5*l]
            c_faces.append(fi)
            c_faces.append(fii)

        elif i > l*0.5 and i < l-1:
            bi = [i, i+1]
            bii = [i, l*0.5]
            c_bars.append(bi)
            c_bars.append(bii)

            fi = [i, l*0.5, i+1, i+1]
            c_faces.append(fi)

        elif i == l-1:
            bi = [i, 0.5*l+1]
            biii = [i, 0.5*l]
            c_bars.append(bi)
            c_bars.append(biii)

            fi = [i, l*0.5, 0.5*l+1, 0.5*l+1]
            c_faces.append(fi)
            
    c_bars = np.array(c_bars, dtype=np.int8)
    c_faces = np.array(c_faces, dtype=np.int8)
    
    return c_points, c_bars, c_faces
    
    
from meshpy.tet import MeshInfo, build
from meshpy.geometry import GeometryBuilder, generate_surface_of_revolution


def createSphere(center, R, m, n, mv=1):
    """
    The function computes the panelisation of a sphere using the predefined
    tools available in the Python module meshpy.
    
    INPUTS:
    - R:    radius of the sphere (m)
    - m:    number of radial subdivisions of the mesh (integer)
    - n:    number of circumferential subdivisions of the mesh (integer)
    - mv:   maxium valume of an element in the thetrahedral mesh 
            (the mesh is actually 3D but we only extract the surface panels)
    
    OUTPUTS:
    - verts:    coordinates of each vertices of the panelisation
    - faces:    triplets of indices representing the vertices of each triangular panel
    - tris:     triplets explicitly representing the coordinates of the vertices
                of each triangular face.
    """
    
    rz = []
    rz.append((0, R))
    for i in xrange(1, n):
        rz.append((R * np.sin(i * np.pi / n), R * np.cos(i * np.pi / n)))
    rz.append((0, -R))

    geob = GeometryBuilder() 
    geob.add_geometry(*generate_surface_of_revolution(rz, radial_subdiv=m)) 

    mesh_info = MeshInfo() 
    geob.set(mesh_info) 

    mesh = build(mesh_info, max_volume=mv) 

    verts = np.array(mesh.points)
    faces = np.array(mesh.faces)
    tris = verts[faces]
    bars = []
    return verts, bars, faces  # , tris
    
if __name__ == '__main__':
    
    '''Some examples on how to use the functions'''

    print "\nSome plotting examples\n"
    print "-----------------------------------------------------------------\n"     
    
    p, b, f = CreateBox([0,0,0], 4, 4, 4)
    p2, b2, f2 = CreateBox([0,0,0], 2, 2, 1)
    axis = PlotGeometry(p, b, color='g', newFig=[True,0])
    PlotGeometry(p2, b2, color='k', newFig=[False, axis])
    
    p_r, f_r, b_r = moveGeometry(p2, [180], [[0,1,0]], [[0.5,0,0]], [0,0,0], f2, b2)
    
    print "\nPlotting some wireframe boxes\n"    
    PlotGeometry(p_r, b_r, color='r', newFig=[False, axis])
    plt.show()
    # sample penetration probability
    # the probability is assigned to the face
    
    Pp = [1, 0.8, 0.6, 0.4, 0.2, 0.1]
    print "Plotting a box with faces colors dependent on a penetration probability distribution\n"
    PlotPenProb(p_r, f_r, Pp, newFig=[True,0], colormap='autumn', saveFig='off', name=None, path=None)
    plt.show()
    
    prova_p, prova_b, prova_f = createCylinder([0,0,0], 0.4, 0.1, 8)
    
    print "\nPlotting a wirefram cylinder\n"
    PlotGeometry(prova_p, prova_b, newFig=[True,0], color='b')
    plt.show()
    
    print "Plotting a cylinder with faces colors dependent on a penetration probability distribution\n"
    Pp = 5 * [0.5] + 5 * [0.2] + 10 * [0.8] + 5 * [1.0] + 5* [0.35]    
    PlotPenProb(prova_p, prova_f, Pp, newFig=[True,0], colormap='winter', saveFig='off', name=None, path=None)
    plt.show()
