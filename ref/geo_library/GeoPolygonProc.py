from Utility import *
from GeoPlane import *
from GeoPoint import *
from GeoFace import *

class GeoPolygonProc(object):

    def __init__(self, polygonInst): # polygonInst: GeoPolygon

        self.__MaxUnitMeasureError = 0.001;

        # Set boundary
        self.__Set3DPolygonBoundary(polygonInst)

        # Set maximum point to face plane distance error,
        self.__Set3DPolygonUnitError()

        # Set faces and face planes
        self.__SetConvex3DFaces(polygonInst)

    @property
    def x0(self):
        return self.__x0
    @property
    def x1(self):
        return self.__x1
    @property
    def y0(self):
        return self.__y0
    @property
    def y1(self):
        return self.__y1
    @property
    def z0(self):
        return self.__z0
    @property
    def z1(self):
        return self.__z1
    @property
    def NumberOfFaces(self):
        return self.__NumberOfFaces
    @property
    def FacePlanes(self):
        return self.__FacePlanes
    @property
    def Faces(self):
        return self.__Faces

    def __Set3DPolygonBoundary(self, polygon): # polygon: GeoPolygon
    
        # List<GeoPoint>
        vertices = polygon.v;

        n = polygon.n;

        xmin = vertices[0].x
        xmax = vertices[0].x
        ymin = vertices[0].y
        ymax = vertices[0].y
        zmin = vertices[0].z
        zmax = vertices[0].z

        for i in range(1, n):
        
            if (vertices[i].x < xmin): xmin = vertices[i].x
            if (vertices[i].y < ymin): ymin = vertices[i].y
            if (vertices[i].z < zmin): zmin = vertices[i].z
            if (vertices[i].x > xmax): xmax = vertices[i].x
            if (vertices[i].y > ymax): ymax = vertices[i].y
            if (vertices[i].z > zmax): zmax = vertices[i].z
                
        self.__x0 = xmin
        self.__x1 = xmax
        self.__y0 = ymin
        self.__y1 = ymax
        self.__z0 = zmin
        self.__z1 = zmax

    def __Set3DPolygonUnitError(self):
        self.MaxDisError = ((abs(self.__x0) + abs(self.__x1) +
                             abs(self.__y0) + abs(self.__y1) +
                             abs(self.__z0) + abs(self.__z1)) / 6 * self.__MaxUnitMeasureError)


    def __SetConvex3DFaces(self, polygon): # polygon: GeoPolygon                             
        
        # vertices indexes for all faces, 2d list                
        faceVerticeIndex = []         

        # face planes for all faces, 1d list       
        fpOutward = []
        
        vertices = polygon.v  
        n = polygon.n
        
        for i in range(0, n):
                    
            # triangle point 1
            
            p0 = vertices[i]
          
            for  j in range(i + 1, n):
                  
                # triangle p             
                p1 = vertices[j]
             
                for k in range(j + 1, n):
                     
                    # triangle point 3
                    
                    p2 = vertices[k]
                    
                    trianglePlane = GeoPlane.Create(p0, p1, p2)
            
                    onLeftCount = 0       
                    onRightCount = 0
                                
                    # indexes of points that is in same plane with face triangle plane           
                    pointInSamePlaneIndex = []           
               
                    for l in range(0, n):
                    
                        # any point other than the 3 triangle points
                        if(l != i and l != j and l != k):
                                               
                            pt = vertices[l]
                                                 
                            dis = trianglePlane * pt
                          
                            # next point is in the triangle plane
                            if(abs(dis) < self.MaxDisError):                                
                                pointInSamePlaneIndex.append(l)                          
                            else:                          
                                if(dis < 0):                             
                                    onLeftCount = onLeftCount + 1                             
                                else:                        
                                    onRightCount = onRightCount + 1                                                

                    # This is a face for a CONVEX 3d polygon.
                    # For a CONCAVE 3d polygon, this maybe not a face.
                    if(onLeftCount == 0 or onRightCount == 0):
                                        
                        verticeIndexInOneFace = [];                                       

                        # add 3 triangle plane vertices
                        verticeIndexInOneFace.append(i);
                        verticeIndexInOneFace.append(j);
                        verticeIndexInOneFace.append(k);

                        m = len(pointInSamePlaneIndex);
                        
                        # add other vertices in the same triangle plane
                        if(m > 0):
                            for p in range(0, m):                      
                                verticeIndexInOneFace.append(pointInSamePlaneIndex[p])
                                                                     
                        # if verticeIndexInOneFace is a new face               
                        if ( not Utility.ContainsList(faceVerticeIndex, verticeIndexInOneFace)):

                            #print(verticeIndexInOneFace)
                            
                            # add it in the faceVerticeIndex list
                            faceVerticeIndex.append(verticeIndexInOneFace)

                            # add the trianglePlane in the face plane list fpOutward.
                            if (onRightCount == 0):                      
                                fpOutward.append(trianglePlane)                  
                            else:
                                if (onLeftCount == 0):                      
                                    fpOutward.append(-trianglePlane)
                        #else:

                            # possible reasons :
                            # 1. the plane is not a face of a convex 3d polygon,
                            #    it is a plane crossing the convex 3d polygon.
                            # 2. the plane is a face of a concave 3d polygon

        # number of faces           
        self.__NumberOfFaces = len(faceVerticeIndex)
        # face list
        self.__Faces = []
        # face planes list
        self.__FacePlanes = []

        for  i in range(0, self.__NumberOfFaces):
        
            self.__FacePlanes.append(GeoPlane(fpOutward[i].a, fpOutward[i].b,
                                              fpOutward[i].c, fpOutward[i].d))

            # face vertices
            v = []
            # face vertices indexes
            idx = []     

            # number of vertices of the face
            count = len(faceVerticeIndex[i])

            for j in range(0, count):
          
                idx.append(faceVerticeIndex[i][j])
                v.append(GeoPoint(vertices[ idx[j] ].x,
                                  vertices[ idx[j] ].y,
                                  vertices[ idx[j] ].z))
          
            self.__Faces.append(GeoFace(v, idx))

    def PointInside3DPolygon(self, x, y, z):
        
        pt = GeoPoint(x, y, z)
        
        for  i in range(0, self.__NumberOfFaces):
            dis = self.__FacePlanes[i] * pt
            # If the point is in the same half space with normal vector for any face of the cube,
            # then it is outside of the 3D polygon
            if(dis > 0):
                return False
            
        # If the point is in the opposite half space with normal vector for all 6 faces,
        # then it is inside of the 3D polygon            
        return True