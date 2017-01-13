from dev import point, vector_v2

print 'Some tests'
print '--------------'
p0 = point.Point(0,1,0)
print 'Create point'
print '-------------'
print 'Point p0: ', p0
print 'p0 coordinates: ', p0.coords
print ''
print 'Add another point'
p1 = point.Point(1.5,2,4)
print 'Point p1', p1
print 'Sum p0 and p1'
print 'p0 + p1 =', p0 + p1
print 'Distance between p0 and p1', p0.distance_to(p1)
print 'Iter through point coordinates'
print 'p0 coordinates'
for c in p0:
    print c
print 'p1 coordinates'
for c in p1:
    print c
print ''
print 'Vectors'
print '---------'
v1 = vector_v2.Vector(p1, p0)
print 'Vector v1: ', v1
print 'Vector v1 normalized: ', v1.normalized()