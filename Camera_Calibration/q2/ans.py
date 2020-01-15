import cv2
#import matplotlib.pyplot as plt
import numpy as np

K = np.array([[7.2153e+02,0,6.0955e+02],[0,7.2153e+02,1.7285e+02],[0,0,1]])


img=cv2.imread('image.png')
#cv2.imshow('img',img)
#cv2.waitKey(10000)

h=1.65
c1=[802,293,1]
c2=[934,294,1]

kin=np.linalg.inv(K)

n=np.array([0,-1,0])

out1=h*np.dot(kin,c1)
out2=np.dot(np.dot(n,kin),c1)
print(out1)
print(out2)
final1=-out1/out2
print(final1)
print(final1.shape)


out1=h*np.dot(kin,c2)
out2=np.dot(np.dot(n,kin),c2)
print(out1)
print(out2)
final2=-out1/out2
print(final2)
print(final2.shape)


print(np.dot(K,final1))
print(np.dot(K,final2))