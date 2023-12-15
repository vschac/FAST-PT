from __future__ import division
import numpy as np
from .J_table import J_table 
import sys
from time import time
from numpy import log, sqrt, exp, pi
from scipy.signal import fftconvolve as convolve

def IA_tij_feG2():
    # Ordering is \alpha, \beta, l_1, l_2, l, A coeficient
    l_mat_tij_feG2=np.array([[0,0,0,2,0,13/21],\
            [0,0,0,2,2,8/21],\
            [1,-1,0,2,1,1/2],\
            [-1,1,0,2,1,1/2]], dtype=float)
    table=np.zeros(10,dtype=float)
    for i in range(l_mat_tij_feG2.shape[0]):
        x=J_table(l_mat_tij_feG2[i])
        table=np.row_stack((table,x))
    return table[1:,:]

def IA_tij_heG2():
    # Ordering is \alpha, \beta, l_1, l_2, l, A coeficient
    l_mat_tij_heG2=np.array([[0,0,0,0,0,-9/70],\
            [0,0,2,0,0,-26/63],\
            [0,0,0,0,2,-15/49],\
            [0,0,2,0,2,-16/63],\
            [0,0,1,1,1,81/70],\
            [0,0,1,1,3,12/35],\
            [0,0,0,0,4,-16/245],\
            [1,-1,0,0,1,-3/10],\
            [1,-1,2,0,1,-1/3],\
            [1,-1,1,1,0,1/2],\
            [1,-1,1,1,2,1],\
            [1,-1,0,2,1,-1/3],\
            [1,-1,0,0,3,-1/5]], dtype=float)
    table=np.zeros(10,dtype=float)
    for i in range(l_mat_tij_heG2.shape[0]):
        x=J_table(l_mat_tij_heG2[i])
        table=np.row_stack((table,x))
    return table[1:,:]

def IA_tij_F2F2():
    # Ordering is \alpha, \beta, l_1, l_2, l, A coeficient
    l_mat_tij_F2F2=np.array([[0,0,0,0,0,1219/1470],\
            [0,0,0,0,2,671/1029],\
            [0,0,0,0,4,32/1715],\
            [2,-2,0,0,0,1/6],\
            [2,-2,0,0,2,1/3],\
            [1,-1,0,0,1,62/35],\
            [1,-1,0,0,4,8/35]], dtype=float)
    table=np.zeros(10,dtype=float)
    for i in range(l_mat_tij_F2F2.shape[0]):
        x=J_table(l_mat_tij_F2F2[i])
        table=np.row_stack((table,x))
    return table[1:,:]

def IA_tij_G2G2():
    # Ordering is \alpha, \beta, l_1, l_2, l, A coeficient
    l_mat_tij_G2G2=np.array([[0,0,0,0,0,851/1470],\
            [0,0,0,0,2,871/1029],\
            [0,0,0,0,4,128/1715],\
            [2,-2,0,0,0,1/6],\
            [2,-2,0,0,2,1/3],\
            [1,-1,0,0,1,54/35],\
            [1,-1,0,0,4,16/35]], dtype=float)
    table=np.zeros(10,dtype=float)
    for i in range(l_mat_tij_G2G2.shape[0]):
        x=J_table(l_mat_tij_G2G2[i])
        table=np.row_stack((table,x))
    return table[1:,:]

def IA_tij_F2G2():
    # Ordering is \alpha, \beta, l_1, l_2, l, A coeficient
    l_mat_tij_F2G2=np.array([[0,0,0,0,0,1003/1470],\
            [0,0,0,0,2,803/1029],\
            [0,0,0,0,4,64/1715],\
            [2,-2,0,0,0,1/6],\
            [2,-2,0,0,2,1/3],\
            [1,-1,0,0,1,58/35],\
            [1,-1,0,0,4,12/35]], dtype=float)
    table=np.zeros(10,dtype=float)
    for i in range(l_mat_tij_F2G2.shape[0]):
        x=J_table(l_mat_tij_F2G2[i])
        table=np.row_stack((table,x))
    return table[1:,:]

def P_IA_13G(k,P):
    N=k.size
    n = np.arange(-N+1,N )
    dL=log(k[1])-log(k[0])
    s=n*dL
    cut=7
    high_s=s[s > cut]
    low_s=s[s < -cut]
    mid_high_s=s[ (s <= cut) &  (s > 0)]
    mid_low_s=s[ (s >= -cut) &  (s < 0)]

    # For Zbar
    Z1=lambda r : -82. + 12/r**2 + 4*r**2 -6*r**4 +log((r+1.)/np.absolute(r-1.))*(-6/r**3+15/r-9*r-3*r**3+3*r**5)
    Z1_high=lambda r : -108 + (72/5)*r**2+ (12/7)*r**4-(352/105)*r**6-(164/105)*r**8 + (58/35)*r**10-(4/21)*r**12-(2/3)*r**14
    Z1_low=lambda r: -316/5+4/(35*r**8)+26/(21*r**6)+608/(105*r**4)-408/(35*r**2)+4/(3*r**12)-34/(21*r**10)

    f_mid_low=Z1(exp(-mid_low_s))
    f_mid_high=Z1(exp(-mid_high_s))
    f_high = Z1_high(exp(-high_s))
    f_low = Z1_low(exp(-low_s))

    f=np.hstack((f_low,f_mid_low,-72.,f_mid_high,f_high))
    # print(f)

    g= convolve(P, f) * dL
    g_k=g[N-1:2*N-1]
    deltaE2= k**3/(336.*pi**2) * P*g_k
    return deltaE2

def P_IA_13F(k,P):
	N=k.size
	n= np.arange(-N+1,N )
	dL=log(k[1])-log(k[0])
	s=n*dL

	cut=7
	high_s=s[s > cut]
	low_s=s[s < -cut]
	mid_high_s=s[ (s <= cut) &  (s > 0)]
	mid_low_s=s[ (s >= -cut) &  (s < 0)]

	Z=lambda r : (12./r**2 -158. + 100.*r**2-42.*r**4 \
	+ 3./r**3*(r**2-1.)**3*(7*r**2+2.)*log((r+1.)/np.absolute(r-1.)) ) *r
	Z_low=lambda r : (352./5.+96./.5/r**2 -160./21./r**4 - 526./105./r**6 +236./35./r**8) *r
	Z_high=lambda r: (928./5.*r**2 - 4512./35.*r**4 +416./21.*r**6 +356./105.*r**8) *r

	f_mid_low=Z(exp(-mid_low_s))
	f_mid_high=Z(exp(-mid_high_s))
	f_high = Z_high(exp(-high_s))
	f_low = Z_low(exp(-low_s))

	f=np.hstack((f_low,f_mid_low,80,f_mid_high,f_high))

	g= convolve(P, f) * dL
	g_k=g[N-1:2*N-1]
	P_bar= 1./252.* k**3/(2*pi)**2*P*g_k

	return P_bar
