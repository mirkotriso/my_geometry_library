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