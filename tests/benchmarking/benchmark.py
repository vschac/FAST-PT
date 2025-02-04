import numpy as np
from fastpt import FASTPT


d=np.loadtxt('Pk_test.dat')
k=d[:,0]; P=d[:,1]

C_window=.75

n_pad=int(0.5*len(k))
to_do=['all']
fpt=FASTPT(k,to_do=to_do,low_extrap=-5,high_extrap=3,n_pad=n_pad)


# calculate 1loop SPT (and time the operation)
P_spt=fpt.one_loop_dd(P,C_window=C_window)


#calculate tidal torque EE and BB P(k)
P_IA_tt=fpt.IA_tt(P,C_window=C_window)
P_IA_ta=fpt.IA_ta(P,C_window=C_window)
P_IA_mix=fpt.IA_mix(P,C_window=C_window)
P_RSD=fpt.RSD_components(P,1.0,C_window=C_window)
P_kPol=fpt.kPol(P,C_window=C_window)
P_OV=fpt.OV(P,C_window=C_window)
sig4=fpt.sig4

print(f"P_RSD: {P_RSD}")




'''
np.savetxt("../pt_bm_z0.txt",
           np.transpose([ks, pgg[0], pgm[0], pgi[0],
                         pii[0], pii_bb[0], pim[0]]),
           header='[0]-k  [1]-GG [2]-GM [3]-GI [4]-II [5]-II_BB [6]-IM')
np.savetxt("../pt_bm_z1.txt",
           np.transpose([ks, pgg[1], pgm[1], pgi[1],
                         pii[1], pii_bb[1], pim[1]]),
           header='[0]-k  [1]-GG [2]-GM [3]-GI [4]-II [5]-II_BB [6]-IM')
'''