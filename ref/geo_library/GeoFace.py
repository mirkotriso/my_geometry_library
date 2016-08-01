class GeoFace(object):

    # ptList: vertice GeoPoint Array
    # idxList: vertice index Integer Array
    def __init__(self, ptList, idxList):

        #alloc instance array memory
        self.__v = []   # vertice point array
        self.__idx = [] # vertice index array

        self.__n = len(ptList) # number of vertices
                        
        for i in range(0, self.__n):
            self.__v.append(ptList[i])
            self.__idx.append(idxList[i])

    @property
    def v(self):
        return self.__v

    @property
    def idx(self):
        return self.__idx

    @property
    def n(self):
        return self.__n