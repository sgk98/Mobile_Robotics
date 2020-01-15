import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

dt = 0.1
t_vec = [i*dt for i in range(51)]

def getPoly(A, bx, by):
    pA = np.linalg.pinv(A)
    return np.dot(pA, bx), np.dot(pA, by)

def getPerp(px, py, t):
    x1 = evaluateAtT(px, t)
    y1 = evaluateAtT(py, t)
    x2 = evaluateAtT(px, t+dt)
    y2 = evaluateAtT(py, t+dt)
    vecx, vecy = x2-x1, y2-y1
    perpx, perpy = vecy, -vecx
    fac = np.sqrt(perpx*perpx + perpy*perpy)
    return perpx/fac, perpy/fac

def evaluateAtT(p, t):
    ans = 0.0
    for j in range(len(p)):
        ans += pow(t, len(p)-1-j) * p[j]
    return ans

def evaluate(p):
    ret = []
    for i in range(len(t_vec)):
        tm = t_vec[i]
        ans = 0.0
        for j in range(len(p)):
            ans += pow(tm, len(p)-1-j) * p[j]
        ret.append(ans)
    return ret

def checkNoIntersection(px, py, cx, cy, r):
    x = evaluate(px)
    y = evaluate(py)
    for i in range(len(x)):
        if (x[i]-cx)**2 + (y[i]-cy)**2 <= r*r:
            print(x[i], y[i])
            return False
    return True

if __name__ == "__main__":
    cur_x, cur_y = 5, 4
    t1 = 1
    t2 = 4

    # passes through obstacle
    A = np.array([
        [0,0,0,0,0,1],        # pos at 0
        [0,0,0,0,1,0],        # vel at 0
        [3125,625,125,25,5,1],   # pos at end
        [3125,500,75,10,1,0],    # vel at end
        [t1**5,t1**4,t1**3,t1**2,t1**1,t1**0],        # intersection point 
        [t2**5,t2**4,t2**3,t2**2,t2**1,t2**0]       # at 3 seconds [offset pos]
    ])
    bx = np.array([3,0,9,0,1,5])
    by = np.array([0,0,5,0,2.5,4])

    px, py = getPoly(A, bx, by)
    perpX, perpY = getPerp(px, py, t2)

    xx = evaluate(px)
    yy = evaluate(py)
    circle1=plt.Circle((5,4),2,color='cyan',alpha=0.1)
    plt.gcf().gca().add_artist(circle1)
    circle1=plt.Circle((5,4),3,color='magenta',alpha=0.1)
    plt.gcf().gca().add_artist(circle1)
    plt.scatter(xx,yy)
    plt.scatter([3],[0],c='white',marker='D',edgecolors='black')
    plt.scatter([9],[5],c='white',marker='D',edgecolors='black')
    plt.scatter([1],[2.5],c='green',marker='D',edgecolors='black')
    plt.scatter([5],[4],c='red',marker='D',edgecolors='black')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    total_r = 3
    offset = 3
    itr = False

    final_px, final_py = None, None

    while True:
        if not itr:
            cur_x = 5 + offset * perpX
            cur_y = 4 + offset * perpY
        else:
            cur_x = 5 + -offset * perpX
            cur_y = 4 + -offset * perpY
            offset += 0.5
        bx[5] = cur_x
        by[5] = cur_y

        new_px, new_py = getPoly(A, bx, by)
        status = checkNoIntersection(new_px, new_py, 5, 4, total_r)

        xx = evaluate(new_px)
        yy = evaluate(new_py)
        circle1=plt.Circle((5,4),2,color='cyan',alpha=0.1)
        plt.gcf().gca().add_artist(circle1)
        circle1=plt.Circle((5,4),3,color='magenta',alpha=0.1)
        plt.gcf().gca().add_artist(circle1)
        plt.scatter(xx,yy)
        plt.scatter([3],[0],c='white',marker='D',edgecolors='black')
        plt.scatter([9],[5],c='white',marker='D',edgecolors='black')
        plt.scatter([1],[2.5],c='green',marker='D',edgecolors='black')
        plt.scatter([5],[4],c='red',marker='D',edgecolors='black')
        plt.scatter([cur_x], [cur_y],c='y',marker='P',edgecolors='black')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

        if status:
            final_px, final_py = new_px, new_py
            break
        itr ^= True

    print(new_px, new_py)

    xx = evaluate(new_px)
    yy = evaluate(new_py)
    plt.scatter(xx,yy)
    plt.scatter([3],[0],c='black',marker='*')
    plt.scatter([9],[5],c='black',marker='*')
    plt.scatter([1],[2.5],c='r',marker='*')
    plt.scatter([5],[4],c='green',marker='*')
    plt.show()
