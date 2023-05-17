import numpy as np


points = np.loadtxt('/Users/jakehirst/Desktop/R.txt')

point1 = points[0]
point2 = points[1]
point3 = points[2]

v1 = point2 - point1
v2 = point3 - point1
normv1 = v1 / np.linalg.norm(v1)
normv2 = v2 / np.linalg.norm(v2)


reference_vector = np.array([1.0, 0.0])

dot1 = np.dot(reference_vector, normv1)
dot2 = np.dot(reference_vector, normv2)
ang1 = np.degrees(np.arccos(dot1))
ang2 = np.degrees(np.arccos(dot2))

if(ang1 > 90):
    ang1 = 180 - ang1
if(ang2 > 90):
    ang2 = 180 - ang2

    
avg_ang = (ang1 + ang2) / 2
print(f"avg angle =   {avg_ang}")

