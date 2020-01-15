import cv2
import numpy as np
import glob
import math
import random
sift = cv2.xfeatures2d.SIFT_create()

K = np.array([[7.070912000000e+02, 0.000000000000e+00, 6.018873000000e+02], [0.000000000000e+00, 7.070912000000e+02, 1.831104000000e+02], [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]],dtype='float64')


imgs=glob.glob('/home/shyamgopal/courses/MR/A2/mr19-assignment2-data/images/*.png')
imgs.sort()
C1=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
T_full=C1

gt=np.loadtxt('./mr19-assignment2-data/ground-truth.txt')

def get_original(R,t):
    mat = np.array([[R[0][0],R[0][1],R[0][2],t[0]],[R[1][0],R[1][1],R[1][2],t[1]],[R[2][0],R[2][1],R[2][2],t[2]],[0,0,0,1]],dtype='float64')
    print(mat.shape)
    return np.linalg.inv(mat)

def get_scale(gt,i):
    return math.sqrt(gt[i][3]**2 + gt[i][7]**2 + gt[i][11]**2)

def n_scale(gt,i):
    g1=gt[i-1]
    g2=gt[i]
    g1=np.append(g1,[0,0,0,1])
    g2=np.append(g2,[0,0,0,1])
    g1=np.reshape(g1,(4,4))
    g2=np.reshape(g2,(4,4))
    g3=np.dot(np.linalg.inv(g1),g2)
    return math.sqrt(g3[0][3]**2 + g3[1][3]**2 + g3[2][3]**2)


def write(all_T):
    write_arr=[]
    for i in range(len(all_T)):
        v=all_T[i]
        cur_ar=[v[0][0],v[0][1],v[0][2],v[0][3],v[1][0],v[1][1],v[1][2],v[1][3],v[2][0],v[2][1],v[2][2],v[2][3]]#,v[3][0],v[3][1],v[3][2],v[3][3]]

        write_arr.append(cur_ar)
    write_arr=np.array(write_arr)
    np.savetxt('pred1.txt',write_arr)



def compute_fundamental(x1, x2):
  x1=np.array(x1)
  x2=np.array(x2)
  n = x1.shape[1]

  A = np.zeros((n, 9))
  for i in range(n):
    A[i] = [x1[0, i] * x2[0, i],  x1[0, i] * x2[1, i],  x1[0, i] * x2[2, i],
            x1[1, i] * x2[0, i],  x1[1, i] * x2[1, i],  x1[1, i] * x2[2, i],
            x1[2, i] * x2[0, i],  x1[2, i] * x2[1, i],  x1[2, i] * x2[2, i],
           ]

  U, S, V = np.linalg.svd(A)
  F = V[-1].reshape(3, 3)

  U, S, V = np.linalg.svd(F)
  S[2] = 0
  F = np.dot(U, np.dot(np.diag(S), V))
  return F / F[2, 2]

def compute_fundamental_normalized(x1, x2):

  x1=np.array(x1)
  x2=np.array(x2)
  x1=x1.T
  x2=x2.T
  n = x1.shape[1]

  x1 = x1 / x1[2]
  mean_1 = np.mean(x1[:2], axis=1)
  S1 = np.sqrt(2) / np.std(x1[:2])
  T1 = np.array([[S1, 0, -S1 * mean_1[0]],
                    [0, S1, -S1 * mean_1[1]],
                    [0, 0, 1]])
  x1 = np.dot(T1, x1)

  x2 = x2 / x2[2]
  mean_2 = np.mean(x2[:2], axis=1)
  S2 = np.sqrt(2) / np.std(x2[:2])
  T2 = np.array([[S2, 0, -S2 * mean_2[0]],
                    [0, S2, -S2 * mean_2[1]],
                    [0, 0, 1]])
  x2 = np.dot(T2, x2)

  F = compute_fundamental(x1, x2)

  F = np.dot(T1.T, np.dot(F, T2))
  return F / F[2, 2]

def solve_ransac(src_pts,dst_pts):
  print("orig shape",src_pts.shape,dst_pts.shape)
  iterations=1000
  err_thresh=0.001
  top_inlier_count=0
  final_H=[]
  for _ in range(iterations):
    inds=random.sample([i for i in range(len(src_pts))],8)
    x1=[]
    x2=[]
    for ind in inds:
      s1=np.array([src_pts[ind][0],src_pts[ind][1],1.0],dtype='float64')
      d1=np.array([dst_pts[ind][0],dst_pts[ind][1],1.0],dtype='float64')
      x1.append(s1)
      x2.append(d1)

    F=compute_fundamental_normalized(x1,x2)
    #H=H/H[2][2]
    
    #projected=np.dot(H,n_src.T)
    #print(projected.shape)
    inlier_count=0
    for i in range(len(src_pts)):
      s1=np.array([src_pts[i][0],src_pts[i][1],1.0],dtype='float64')
      d1=np.array([dst_pts[i][0],dst_pts[i][1],1.0],dtype='float64')
      e1=np.dot(d1,np.dot(F,s1))
      #print(e1)
      if abs(e1)<err_thresh:
        inlier_count+=1
        #print("HHAHA")
    #print("inlier count",inlier_count)
    if inlier_count>top_inlier_count:
      top_inlier_count=inlier_count
      final_F=F
  print(top_inlier_count,"top count","total:",len(src_pts))
  #print(len(src_pts),top_inlier_count)
  return final_F

all_T=[]
for i in range(1,100):
    print(i)
    all_T.append(T_full)
    img1= cv2.imread(imgs[i-1])
    img2= cv2.imread(imgs[i])
    print(imgs[i-1],imgs[i])
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50) 
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2) 
    good = []
    pts1 = []
    pts2 = []

    for ii,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)   
    F= solve_ransac(pts1,pts2)
    print(F,F1)
    E = np.dot(K.T,np.dot(F,K))

    points, R, t, mask = cv2.recoverPose(E, pts1, pts2,K)
    print(gt.shape,i)
    scale=n_scale(gt,i)
    print("scale",scale)
    t[0] *= scale
    t[1] *= scale
    t[2] *= scale
    Ti = get_original(R,t)
    T_full=np.dot(T_full,Ti)


#output=open('pred.txt','w')
all_T.append(T_full)
write(all_T)


