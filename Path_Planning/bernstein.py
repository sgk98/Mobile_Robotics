import numpy as np
import matplotlib.pyplot as plt

def factorial(n):
    ans=1
    for i in range(2,n+1):
        ans*=i
    return ans

def nck(n,r):
    return factorial(n)/(factorial(r)*factorial(n-r))

def bernstein(arr,t,n=5,ti=0,tf=5):
    ans=0.0
    tau=(t-ti)/(tf-ti)
    n1=len(arr)
    for i in range(n1):
        ans+=nck(n,i)*pow(tau,i)*pow(1.0-tau,n-i)*arr[i]
    return ans

A=np.array([[1,0,0,0,0,0],[0,0,0,0,0,1],[-1,1,0,0,0,0],[0,0,0,0,-1,1],[pow(0.8,5),pow(0.8,4),0.4*pow(0.8,3),2*pow(0.8,2)/25.0,0.8/125,1/3125.0]])
b=np.array([3,9,0,0,1])

px = np.linalg.pinv(A)
x=np.dot(px,b)

b1=np.array([0,5,0,0,2.5])
y=np.dot(px,b1)

print(x)
print(y)

t=[i*0.1 for i in range(51)]
xx=[]
yy=[]
for i in range(len(t)):
    xx.append(bernstein(x,t[i]))
    yy.append(bernstein(y,t[i]))

plt.scatter(xx,yy)
plt.show()

#### velocities

def ber_vel(arr, t, n=5, ti=0, tf=5):
    ans = 0.0
    tau = (t-ti)/(tf-ti)
    for i in range(n+1):
        temp = 0.0
        if i==0:
            temp += -(n*pow(1-tau,n-1)) / (tf-ti)
        elif i==n:
            temp += (n*pow(tau,n-1)) / (tf-ti)
        else:
            temp += nck(n,i) * ((i*pow(tau,i-1)*pow(1-tau,n-i))   +   -((n-i)*pow(1-tau,n-i-1)*pow(tau,i))) / (tf-ti)
        ans += temp * arr[i]
    return ans

vx=[]
vy=[]
for i in range(len(t)):
    vx.append(ber_vel(x,t[i]))
    vy.append(ber_vel(y,t[i]))

plt.plot(vx,c='green')
plt.plot(vy,c='red')
plt.show()

### acceleration

def ber_acc(arr, t, n=5, ti=0, tf=5):
    ans = 0.0
    tau = (t-ti)/(tf-ti)
    for i in range(n+1):
        temp = 0.0
        if i > 1:
            temp += i*(i-1)*pow(tau,i-2)*pow(1-tau,n-i)
        if i > 0 and i < n:
            temp += -2 * (n-i) * i * pow(tau,i-1) * pow(1-tau,n-i-1)
        if i < n-1:
            temp += (n-i)*(n-i-1)*pow(tau,i)*pow(1-tau,n-i-2)
        ans += temp * nck(n,i) * arr[i] / ((tf-ti)**2)
    return ans

ax=[]
ay=[]
for i in range(len(t)):
    ax.append(ber_acc(x,t[i]))
    ay.append(ber_acc(y,t[i]))

plt.plot(ax,c='green')
plt.plot(ay,c='red')
plt.show()


