
import copy
import numpy as np

SIZE = (150, 150)

nodes = []
nodes.append( (127.27469778628347, 77.322339830751986) )
nodes.append( (39.966248586344314, 71.076250180488714) )
nodes.append( (80.086537224875883, 109.88506230474758) )
nodes.append( (96.514251724643756, 53.510220314713109) )
nodes.append( (0.0, 0.0) )
nodes.append( (71.625112769259644, 23.001562415813979) )
nodes.append( (0.0, 150.0) )
nodes.append( (150.0, 150.0) )
nodes.append( (150.0, 0.0) )

NUM_DUMMY_NODES = 0
VECTOR_LENGTH = len(nodes)-NUM_DUMMY_NODES

triangles = []
triangles.append( (6, 2, 7) )
triangles.append( (0, 8, 7) )
triangles.append( (0, 3, 8) )
triangles.append( (2, 0, 7) )
triangles.append( (0, 2, 3) )
triangles.append( (2, 1, 3) )
triangles.append( (1, 6, 4) )
triangles.append( (1, 2, 6) )
triangles.append( (5, 1, 4) )
triangles.append( (1, 5, 3) )
triangles.append( (8, 5, 4) )
triangles.append( (3, 5, 8) )


def point_to_vector(x, y):
    x = float(x)*SIZE[0]
    y = float(y)*SIZE[1]

    for i, t in enumerate(triangles):
        p1 = nodes[t[0]]
        p2 = nodes[t[1]]
        p3 = nodes[t[2]]

        x1 = float(p1[0])
        x2 = float(p2[0])
        x3 = float(p3[0])

        y1 = float(p1[1])
        y2 = float(p2[1])
        y3 = float(p3[1])

        a = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3))
        b = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3))
        c = 1 - a - b

        if a >= 0 and b >= 0 and c >= 0:
            #print "triangle:", i, t, (a, b, c)

            v = [0]*VECTOR_LENGTH
            if t[0] < VECTOR_LENGTH: v[t[0]] = a
            if t[1] < VECTOR_LENGTH: v[t[1]] = b
            if t[2] < VECTOR_LENGTH: v[t[2]] = c

            return v



def vector_to_point(v):

    x = 0.0
    y = 0.0
    sv = 0.0

    for i in range(0, VECTOR_LENGTH):
        x += v[i] * nodes[i][0]
        y += v[i] * nodes[i][1]
        sv += v[i]

    return (x/sv/SIZE[0], y/sv/SIZE[1])



