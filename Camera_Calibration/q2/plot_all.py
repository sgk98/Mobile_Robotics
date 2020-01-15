import cv2
#import matplotlib.pyplot as plt
import numpy as np

K = np.array([[7.2153e+02,0,6.0955e+02],[0,7.2153e+02,1.7285e+02],[0,0,1]])


img=cv2.imread('image.png')
lineThickness = 2
#cv2.line(image, (x1, y1), (x2, y2), (0,255,0), lineThickness)

p1=np.array([2.6428839,1.65,9.90865169])

p2=np.array([4.63675196,1.65 ,9.82686339])

p3=np.array([4.63675196,1.65,4.10+9.82686339])

p4=np.array([2.6428839,1.65,9.90865169+4.10])


p5=np.array([2.6428839,1.65-1.38,9.90865169])

p6=np.array([4.63675196,1.65-1.38 ,9.82686339])

p7=np.array([4.63675196,1.65-1.38,4.10+9.82686339])

p8=np.array([2.6428839,1.65-1.38,9.90865169+4.10])

o1=np.dot(K,p1)
o2=np.dot(K,p2)
o3=np.dot(K,p3)
o4=np.dot(K,p4)
o5=np.dot(K,p5)
o6=np.dot(K,p6)
o7=np.dot(K,p7)
o8=np.dot(K,p8)

o1[0]/=o1[2]
o1[1]/=o1[2]

o2[0]/=o2[2]
o2[1]/=o2[2]


o3[0]/=o3[2]
o3[1]/=o3[2]


o4[0]/=o4[2]
o4[1]/=o4[2]

o5[0]/=o5[2]
o5[1]/=o5[2]

o6[0]/=o6[2]
o6[1]/=o6[2]


o7[0]/=o7[2]
o7[1]/=o7[2]


o8[0]/=o8[2]
o8[1]/=o8[2]

p1=o1
p2=o2
p3=o3
p4=o4
p5=o5
p6=o6
p7=o7
p8=o8

#cv2.rectangle(img, (int(o4[0]), int(o4[1])), (int(o2[0]), int(o2[1])), (255,0,0), 2)
cv2.line(img, (int(p1[0]),int(p1[1])), (int(p2[0]), int(p2[1])), (0,255,0), lineThickness)

cv2.line(img, (int(p2[0]),int(p2[1])), (int(p3[0]), int(p3[1])), (0,255,0), lineThickness)

cv2.line(img, (int(p3[0]),int(p3[1])), (int(p4[0]), int(p4[1])), (0,255,0), lineThickness)

cv2.line(img, (int(p4[0]),int(p4[1])), (int(p1[0]), int(p1[1])), (0,255,0), lineThickness)




cv2.line(img, (int(p5[0]),int(p5[1])), (int(p6[0]), int(p6[1])), (0,255,0), lineThickness)
cv2.line(img, (int(p6[0]),int(p6[1])), (int(p7[0]), int(p7[1])), (0,255,0), lineThickness)
cv2.line(img, (int(p7[0]),int(p7[1])), (int(p8[0]), int(p8[1])), (0,255,0), lineThickness)
cv2.line(img, (int(p8[0]),int(p8[1])), (int(p5[0]), int(p5[1])), (0,255,0), lineThickness)


cv2.line(img, (int(p1[0]),int(p1[1])), (int(p5[0]), int(p5[1])), (0,255,0), lineThickness)
cv2.line(img, (int(p2[0]),int(p2[1])), (int(p6[0]), int(p6[1])), (0,255,0), lineThickness)
cv2.line(img, (int(p3[0]),int(p3[1])), (int(p7[0]), int(p7[1])), (0,255,0), lineThickness)
cv2.line(img, (int(p4[0]),int(p4[1])), (int(p8[0]), int(p8[1])), (0,255,0), lineThickness)




cv2.imshow('img',img)
cv2.waitKey(10000)