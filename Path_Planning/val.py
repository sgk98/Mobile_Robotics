import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

A=np.array([[0,0,0,0,1],[0,0,0,1,0],[625,125,25,5,1],[500,75,10,1,0],[1,1,1,1,1]])
b=np.array([3,0,9,0,1])

px = np.linalg.pinv(A)
x=np.dot(px,b)
print(x)

by=np.array([0,0,5,0,2.5])
y=np.dot(px,by)
print(y)

def evaluate(x,tm):
    ans=0.0
    for i in range(len(x)):
        ans+=pow(tm,len(x)-1-i)*x[i]
    return ans

t=[i*0.1 for i in range(51)]
xx=[]
yy=[]
for i in range(len(t)):
    xx.append(evaluate(x,t[i]))
    yy.append(evaluate(y,t[i]))

plt.scatter(xx,yy)
plt.scatter([3],[0],c='w',marker='*')
plt.scatter([9],[5],c='w',marker='*')
plt.scatter([1],[2.5],c='r',marker='*')
plt.show()

##### Getting velocities

def evaluate_velocity(x,tm):
    ans = 0.0
    for i in range(len(x)-1):
        ans += (len(x)-1-i) * x[i] * pow(tm, len(x)-2-i)
    return ans

vx = []
vy = []
v = []
for i in range(len(t)):
    vx.append(evaluate_velocity(x,t[i]))
    vy.append(evaluate_velocity(y,t[i]))
    v.append(sqrt((vx[-1]**2) + (vy[-1]**2)))

plt.plot(np.array(vx), 'green', label="Horizontal Velocity")
plt.plot(np.array(vy), 'blue', label="Vertical Velocity")
plt.legend()
plt.show()

##### Getting velocities

def evaluate_acceleration(x,tm):
    ans = 0.0
    for i in range(len(x)-2):
        ans += (len(x)-1-i) * (len(x)-2-i) * x[i] * pow(tm, len(x)-3-i)
    return ans

ax = []
ay = []
a = []
for i in range(len(t)):
    ax.append(evaluate_acceleration(x,t[i]))
    ay.append(evaluate_acceleration(y,t[i]))
    a.append(sqrt((ax[-1]**2) + (ay[-1]**2)))

# plt.scatter(vx, vy)
plt.plot(np.array(ax), 'green', label='Horizontal Acceleration')
plt.plot(np.array(ay), 'blue', label='Vertical Acceleration')
plt.legend()
plt.show()
