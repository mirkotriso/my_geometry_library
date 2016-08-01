class GeoPolygon(object):

    def __init__(self, ptList): # ptList: vertice GeoPoint Array

        #alloc instance array memory
        self.__v = []   # vertice point array
        self.__idx = [] # vertice index array

        self.__n = len(ptList) # number of vertices
                        
        for pt in ptList:
            self.__v.append(pt)
            self.__idx.append(ptList.index(pt))

    @property
    def v(self):
        return self.__v

    @property
    def idx(self):
        return self.__idx

    @property
    def n(self):
        return self.__n