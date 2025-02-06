import numpy as np
from fastpt import FASTPT


d=np.loadtxt('Pk_test.dat')
k=d[:,0]; P=d[:,1]
C_window=.75
n_pad=int(0.5*len(k))
to_do=['all']
fpt=FASTPT(k,to_do=to_do,low_extrap=-5,high_extrap=3,n_pad=n_pad)

# Base one-loop terms
P_spt = fpt.one_loop_dd(P, C_window=C_window)[0]  # Need [0] for P_22 + P_13
P_kPol = fpt.kPol(P, C_window=C_window)
P_OV = fpt.OV(P, C_window=C_window)

# Bias terms
P_bias = fpt.one_loop_dd_bias(P, C_window=C_window)
P_bias_b3nl = fpt.one_loop_dd_bias_b3nl(P, C_window=C_window)
P_bias_lpt_NL = fpt.one_loop_dd_bias_lpt_NL(P, C_window=C_window)

# CLEFT terms - requires cleft_z1 and z2
#P_cleft_Q_R = fpt.cleft_Q_R(P, C_window=C_window)
# ^^ Throws error because cleft_z1 and cleft_z2 are not defined

# Intrinsic Alignment terms
P_IA_tt = fpt.IA_tt(P, C_window=C_window)
P_IA_ta = fpt.IA_ta(P, C_window=C_window)
P_IA_mix = fpt.IA_mix(P, C_window=C_window)
P_IA_ct = fpt.IA_ct(P, C_window=C_window)
P_IA_ctbias = fpt.IA_ctbias(P, C_window=C_window)
P_IA_gb2 = fpt.IA_gb2(P, C_window=C_window)
P_IA_d2 = fpt.IA_d2(P, C_window=C_window)
P_IA_s2 = fpt.IA_s2(P, C_window=C_window)

# RSD terms - note P_RSD requires f=1.0 parameter
P_RSD = fpt.RSD_components(P, 1.0, C_window=C_window)
P_RSD_ABsum_components = fpt.RSD_ABsum_components(P, 1.0, C_window=C_window)
P_RSD_ABsum_mu = fpt.RSD_ABsum_mu(P, 1.0, 1.0, C_window=C_window)

# IR resummation
P_IRres = fpt.IRres(P, C_window=C_window)


# Debug: Print shapes
print(f"k shape: {k.shape}") # (3000,)
print(f"P_spt shape: {np.array(P_spt).shape}") #(3000,)
print(f"P_kPol shape: {np.array(P_kPol).shape}") #(3,3000)
print(f"P_OV shape: {np.array(P_OV).shape}") #(3000,)
#print(f"P_bias shape: {np.array(P_bias).shape}") inhomogeneuous array sizes
#print(f"P_bias_b3nl shape: {np.array(P_bias_b3nl).shape}") inhomogeneuous array sizes
#print(f"P_bias_lpt_NL shape: {np.array(P_bias_lpt_NL).shape}") inhomogeneuous array sizes
print(f"P_IA_tt shape: {np.array(P_IA_tt).shape}") #(2,3000)
print(f"P_IA_ta shape: {np.array(P_IA_ta).shape}") #(4,3000)
print(f"P_IA_mix shape: {np.array(P_IA_mix).shape}") #(4,3000)
print(f"P_IA_ct shape: {np.array(P_IA_ct).shape}") #(4,3000)
print(f"P_IA_ctbias shape: {np.array(P_IA_ctbias).shape}") #(2,3000)
print(f"P_IA_gb2 shape: {np.array(P_IA_gb2).shape}") #(3,3000)
print(f"P_IA_d2 shape: {np.array(P_IA_d2).shape}") #(3,3000)
print(f"P_IA_s2 shape: {np.array(P_IA_s2).shape}") #(3,3000)
print(f"P_RSD_ABsum_components shape: {np.array(P_RSD_ABsum_components).shape}") #(4,3000)
print(f"P_RSD_ABsum_mu shape: {np.array(P_RSD_ABsum_mu).shape}") #(3000,)
print(f"P_IRres shape: {np.array(P_IRres).shape}") #(3000,)


names = {
    'k': k,
    'P_spt': P_spt,
    'P_OV': P_OV,
    'P_RSD_ABsum_mu': P_RSD_ABsum_mu,
    'P_IRres': P_IRres,
    'P_kPol': P_kPol,
    'PIA_tt': P_IA_tt,
    'P_IA_ta': P_IA_ta,
    'P_IA_mix': P_IA_mix,
    'P_IA_ct': P_IA_ct,
    'P_IA_ctbias': P_IA_ctbias,
    'P_IA_gb2': P_IA_gb2,
    'P_IA_d2': P_IA_d2,
    'P_IA_s2': P_IA_s2,
    'P_RSD': P_RSD,
    'P_RSD_ABsum_components': P_RSD_ABsum_components
}
'''
for name, arr in names.items():
    try: 
        np.savetxt(f'{name}_benchmark.txt', arr, header=f'{name}')
    except AttributeError:
        print(f"Error saving {name} array")
        print(AttributeError.with_traceback())
'''

inhomogeneous_array_names = {
    'P_bias': P_bias,
    'P_bias_b3nl': P_bias_b3nl,
    'P_bias_lpt_NL': P_bias_lpt_NL,
}

for name, arr in inhomogeneous_array_names.items():
    print(f"\n{name} type: {type(arr)}")
    print(f"{name} length: {len(arr)}")
    for i, component in enumerate(arr):
        print(f"Component {i} shape: {np.array(component).shape}")


'''
# Modify data storage to handle multi-component returns
data = np.transpose([
    k,
    P_spt, P_kPol, P_OV,
    P_bias, P_bias_b3nl, P_bias_lpt_NL,
    P_IA_tt, P_IA_ta, P_IA_mix, P_IA_ct, P_IA_ctbias, P_IA_gb2, P_IA_d2, P_IA_s2,
    P_RSD_ABsum_mu,  # Only use the final combined result
    P_IRres
])

header = 'k P_spt P_kPol P_OV P_bias P_bias_b3nl P_bias_lpt_NL P_IA_tt P_IA_ta P_IA_mix P_IA_ct P_IA_ctbias P_IA_gb2 P_IA_d2 P_IA_s2 P_RSD P_IRres'
np.savetxt('fastpt_benchmark_results.txt', data, header=header)
'''

