import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def fdlt(XW,XI):
	#print(XI)
	#print(XI.shape)
	x=XI[:,0]
	y=XI[:,1]
	z=XI[:,2]
	print(x)
	print(y)
	print(z)
	n=len(XW[:,0])
	print(n)
	xw=np.array(XW)
	#print(xw.shape)
	A=[]
	for i in range(n):
		t1=[xw[i][0],xw[i][1],xw[i][2],xw[i][3],0,0,0,0,-1*x[i]*xw[i][0],-1*x[i]*xw[i][1],-1*x[i]*xw[i][2],-1*x[i]]

		t2=[0,0,0,0,xw[i][0],xw[i][1],xw[i][2],xw[i][3],-1*y[i]*xw[i][0],-1*y[i]*xw[i][1],-1*y[i]*xw[i][2],-1*y[i]]
		A.append(t1)
		A.append(t2)
	A=np.array(A)
	u,s,v=np.linalg.svd(A)
	print("shape of v is",v.shape)
	out=[]
	for i in range(len(v)):
		out.append(v[i][-1])
	print(len(out))
	v=np.transpose(v)
	return v
	
#img=cv2.imread('image.png')
#cv2.imshow('img',img)
#cv2.waitKey(0)

K = np.array([[406.952636, 0.000000, 366.184147],[ 0.000000, 405.671292, 244.705127],[0.000000, 0.000000, 1.000000]])

x=[284.56243896,373.93179321,387.53588867,281.29962158,428.86453247,524.76373291,568.3659668,453.60995483]
y=[149.2925415,128.26719666,220.2270813,241.72782898,114.50731659,92.09218597,180.55757141,205.22370911]



 
 

X=[0,0.1315,0.1315,0,0.1315+0.0790,0.1315+0.0790+0.1315,0.1315+0.0790+0.1315,0.1315+0.0790]
Y=[0,0,0.1315,0.1315,0,0,0.1315,0.1315]
Z=[5,5,5,5,5,5,5,5]
Z=np.array(Z,dtype='float64')
#Z*=0.2


XI=[]
XW=[]
for i in range(len(x)):
	t1=[x[i],y[i],1.0]
	t2=[X[i],Y[i],Z[i],1.0]
	XI.append(t1)
	XW.append(t2)

XI=np.array(XI,dtype='float64')
XW=np.array(XW,dtype='float64')
#XI=XI[:6,:]
#XW=XW[:6,:]



v=fdlt(XW[:6,:],XI[:6,:])
out=[]
for i in range(len(v)):
	out.append(v[i][-1])

out=np.array(out)
out=np.reshape(out,(3,4))
print(out)

projected=np.dot(out,XW.T)
print("shape",XW.shape)
print(projected.shape)

x1,y1=[],[]
for i in range(8):
	x1.append(projected[0][i]/projected[2][i])
	y1.append(projected[1][i]/projected[2][i])


print("x1,y1")
print(x1)
print(y1)

print(x,y)

P=[]
for i in range(len(out)):
	tmp=[out[i][0],out[i][1],out[i][3]]
	P.append(tmp)

P=np.array(P)

kin=np.linalg.inv(K)
print('finally')
print(np.dot(kin,P))


H=np.dot(kin,P)
h1=H[:,0]
h2=H[:,1]
h12=np.cross(h1,h2)
print("homography")
print(H)

new_matrix=np.column_stack((h1,h2,h12))
u,s,v=np.linalg.svd(new_matrix)


new_det=np.linalg.det(np.dot(u,v))
print("det")
print(new_det)
new_d=np.array([[1,0,0],[0,1,0],[0,0,new_det]])
R=np.dot(u,np.dot(new_d,v))
print("rotation")
print(R)

print('translation')
h3=np.array(H[:,2],dtype='float64')
print(h3)
print(h1)
print(np.linalg.norm(h1))
#print(translation)
