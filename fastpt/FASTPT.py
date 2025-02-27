'''
	FASTPT is a numerical algorithm to calculate
	1-loop contributions to the matter power spectrum
	and other integrals of a similar type.
	The method is presented in papers arXiv:1603.04826 and arXiv:1609.05978
	Please cite these papers if you are using FASTPT in your research.

	Joseph E. McEwen (c) 2016
	mcewen.24@osu.edu

	Xiao Fang
	fang.307@osu.edu

	Jonathan A. Blazek
	blazek.35@osu.edu


	FFFFFFFF    A           SSSSSSSSS   TTTTTTTTTTTTTT             PPPPPPPPP    TTTTTTTTTTTT
	FF     	   A A         SS                 TT                   PP      PP        TT
	FF        A   A        SS                 TT                   PP      PP        TT
	FFFFF    AAAAAAA        SSSSSSSS          TT       ==========  PPPPPPPPP         TT
	FF      AA     AA              SS         TT                   PP                TT
	FF     AA       AA             SS         TT                   PP                TT
	FF    AA         AA    SSSSSSSSS          TT                   PP                TT


	The FASTPT class is the workhorse of the FASTPT algorithm.
	This class calculates integrals of the form:

	\int \frac{d^3q}{(2 \pi)^3} K(q,k-q) P(q) P(|k-q|)

	\int \frac{d^3q_1}{(2 \pi)^3} K(\hat{q_1} \dot \hat{q_2},\hat{q_1} \dot \hat{k}, \hat{q_2} \dot \hat{k}, q_1, q_2) P(q_1) P(|k-q_1|)

'''
from __future__ import division
from __future__ import print_function

from .info import __version__

import numpy as np
from numpy import exp, log, cos, sin, pi
from .fastpt_extr import p_window, c_window
from .matter_power_spt import P_13_reg, Y1_reg_NL, Y2_reg_NL
from .initialize_params import scalar_stuff, tensor_stuff
from .IA_tt import IA_tt
from .IA_ABD import IA_A, IA_DEE, IA_DBB, P_IA_B
from .IA_ta import IA_deltaE1, P_IA_deltaE2, IA_0E0E, IA_0B0B
from .IA_gb2 import IA_gb2_F2, IA_gb2_fe, IA_gb2_he, P_IA_13S2F2
from .IA_gb2 import IA_gb2_S2F2, IA_gb2_S2fe, IA_gb2_S2he
from .IA_ct import IA_tij_feG2, IA_tij_heG2, IA_tij_F2F2, IA_tij_G2G2, IA_tij_F2G2, P_IA_13G, P_IA_13F, IA_tij_F2G2reg
from .IA_ctbias import IA_gb2_F2, IA_gb2_G2, IA_gb2_S2F2, IA_gb2_S2G2
from .OV import OV
from .kPol import kPol
from .RSD import RSDA, RSDB
from . import RSD_ItypeII
from .P_extend import k_extend
from . import FASTPT_simple as fastpt_simple

log2 = log(2.)


def cached_property(method):
    """Decorator to cache property values"""
    cache_name = f'_{method.__name__}'
    
    def getter(instance):
        if not hasattr(instance, cache_name):
            setattr(instance, cache_name, method(instance))
        return getattr(instance, cache_name)
    
    return property(getter)

class FASTPT:

    def __init__(self, k, nu=None, to_do=None, param_mat=None, low_extrap=None, high_extrap=None, n_pad=None,
                 verbose=False):

        ''' inputs:
				* k grid
				* the to_do list: e.g. one_loop density density , bias terms, ...
				* low_extrap is the call to extrapolate the power spectrum to lower k-values,
					this helps with edge effects
				* n_pad is the number of zeros to add to both ends of the array. This helps with
					edge effects.
				* verbose is to turn on verbose settings.
		'''
        
        if (k is None or len(k) == 0):
            raise ValueError('You must provide an input k array.')
        
        if nu: print("Warning: nu is no longer needed for FAST-PT initialization.")
        

        # if no to_do list is given, default to fastpt_simple SPT case
        if (to_do is None):
            if (verbose):
                print(
                    'Note: You are using an earlier call structure for FASTPT. Your code will still run correctly, calling FASTPT_simple. See user manual.')
            if (nu is None):  # give a warning if nu=None that a default value is being used.
                print('WARNING: No value for nu is given. FASTPT_simple is being called with a default of nu=-2')
                nu = -2  # this is the default value for P22+P13 and bias calculation
            print("WARNING: No to_do list is given therefore calling FASTPT_simple. FASTPT_simple will soon be DEPRECATED.")
            self.pt_simple = fastpt_simple.FASTPT(k, nu, param_mat=param_mat, low_extrap=low_extrap,
                                                  high_extrap=high_extrap, n_pad=n_pad, verbose=verbose)
            return None
        # Exit initialization here, since fastpt_simple performs the various checks on the k grid and does extrapolation.
        


        self.cache = {} #Used for storing JK tensor and scalar values
        self.c_cache = {} #Used for storing c_m, c_n, and c_l values
        self.term_cache = {} #Used for storing individual terms from all FAST-PT functions
        self.__k_original = k
        self.extrap = False
        if (low_extrap is not None or high_extrap is not None):
            if (high_extrap < low_extrap):
                raise ValueError('high_extrap must be greater than low_extrap')
            self.EK = k_extend(k, low_extrap, high_extrap)
            k = self.EK.extrap_k()
            self.extrap = True

        self.low_extrap = low_extrap
        self.high_extrap = high_extrap
        self.__k_extrap = k #K extrapolation not padded

        
        # check for log spacing
        # print('Initializing k-grid quantities...')
        dk = np.diff(np.log(k))
        # dk_test=np.ones_like(dk)*dk[0]
        delta_L = (log(k[-1]) - log(k[0])) / (k.size - 1)
        dk_test = np.ones_like(dk) * delta_L

        log_sample_test = 'ERROR! FASTPT will not work if your in put (k,Pk) values are not sampled evenly in log space!'
        np.testing.assert_array_almost_equal(dk, dk_test, decimal=4, err_msg=log_sample_test, verbose=False)

        if (verbose):
            print(f'the minumum and maximum inputed log10(k) are: {np.min(np.log10(k))} and {np.max(np.log10(k))}')
            print(f'the grid spacing Delta log (k) is, {(log(np.max(k)) - log(np.min(k))) / (k.size - 1)}')
            print(f'number of input k points are, {k.size}')
            print(f'the power spectrum is extraplated to log10(k_min)={low_extrap}')
            print(f'the power spectrum is extraplated to log10(k_max)={high_extrap}')
            print(f'the power spectrum has {n_pad} zeros added to both ends of the power spectrum')


        # print(self.k_extrap.size, 'k size')
        # size of input array must be an even number
        if (k.size % 2 != 0):
            raise ValueError('Input array must contain an even number of elements.')
        # can we just force the extrapolation to add an element if we need one more? how do we prevent the extrapolation from giving us an odd number of elements? is that hard coded into extrap? or just trim the lowest k value if there is an odd numebr and no extrapolation is requested.

        if (n_pad != None):
            # Make sure n_pad is an integer
            if not isinstance(n_pad, int):
                n_pad = int(n_pad)
            self.n_pad = n_pad
            self.id_pad = np.arange(k.size) + n_pad
            d_logk = delta_L
            k_pad = np.log(k[0]) - np.arange(1, n_pad + 1) * d_logk
            k_pad = np.exp(k_pad)
            k_left = k_pad[::-1]

            k_pad = np.log(k[-1]) + np.arange(1, n_pad + 1) * d_logk
            k_right = np.exp(k_pad)
            k = np.hstack((k_left, k, k_right))
            n_pad_check = int(np.log(2) / delta_L) + 1
            if (n_pad < n_pad_check):
                print('*** Warning ***')
                print(f'You should consider increasing your zero padding to at least {n_pad_check}')
                print('to ensure that the minimum k_output is > 2k_min in the FASTPT universe.')
                print(f'k_min in the FASTPT universe is {k[0]} while k_min_input is {self.k_extrap[0]}')
        else:
            print("WARNING: N_pad is recommended but none has been provided, defaulting to 0.")
            self.n_pad = 0

        self.__k_final = k #log spaced k, with padding and extrap
        self.k_size = k.size
        # self.scalar_nu=-2
        self.N = k.size

        # define eta_m and eta_n=eta_m
        omega = 2 * pi / (float(self.N) * delta_L)
        self.m = np.arange(-self.N // 2, self.N // 2 + 1)
        self.eta_m = omega * self.m

        self.verbose = verbose

        # define l and tau_l
        self.n_l = self.m.size + self.m.size - 1
        self.l = np.arange(-self.n_l // 2 + 1, self.n_l // 2 + 1)
        self.tau_l = omega * self.l

        if to_do: print("Warning: to_do list is no longer needed for FAST-PT initialization. Terms will now be calculated as needed.")
        self.todo_dict = {
            'one_loop_dd': False, 'one_loop_cleft_dd': False, 
            'dd_bias': False, 'IA_all': False,
            'IA_tt': False, 'IA_ta': False, 
            'IA_mix': False, 'OV': False, 'kPol': False,
            'RSD': False, 'IRres': False, 
            'tij': False, 'gb2': False, 
            'all': False, 'everything': False
        }

        for entry in to_do:
            if entry in {'all', 'everything'}:
                for key in self.todo_dict:
                    self.todo_dict[key] = True
            elif entry in {'IA_all', 'IA'}:
                for key in ['IA_tt', 'IA_ta', 'IA_mix', 'gb2', 'tij']:
                    self.todo_dict[key] = True
            elif entry == 'dd_bias':
                self.todo_dict['one_loop_dd'] = True
                self.todo_dict['dd_bias'] = True
            elif entry == 'tij':
                for key in ['gb2', 'one_loop_dd', 'tij', 'IA_tt', 'IA_ta', 'IA_mix']:
                    self.todo_dict[key] = True
            elif entry in self.todo_dict:
                self.todo_dict[entry] = True
            elif entry == 'skip':
                #If todo list is skipped no terms will be calculated at Fast-PT initialization,
                #instead they will be calculated as they are needed then cached for later use.
                break
            else:
                raise ValueError(f'FAST-PT does not recognize {entry} in the to_do list.\n{self.todo_dict.keys()} are the valid entries.')

        
        ### INITIALIZATION of k-grid quantities ###
        if self.todo_dict['one_loop_dd'] or self.todo_dict['dd_bias'] or self.todo_dict['IRres']:
            self.X_spt
            self.X_lpt
            self.X_sptG

        if self.todo_dict['one_loop_cleft_dd']:
            self.X_cleft
        if self.todo_dict['IA_tt']: 
            self.X_IA_E 
            self.X_IA_B

        if self.todo_dict['IA_mix']:
            self.X_IA_A
            self.X_IA_DEE
            self.X_IA_DBB

        if self.todo_dict['IA_ta']:
            self.X_IA_deltaE1
            self.X_IA_0E0E
            self.X_IA_0B0B

        if self.todo_dict['gb2']:
            self.X_IA_gb2_fe
            self.X_IA_gb2_he

        if self.todo_dict['tij']:
            self.X_IA_tij_feG2
            self.X_IA_tij_heG2
            self.X_IA_tij_F2F2
            self.X_IA_tij_G2G2
            self.X_IA_tij_F2G2
            self.X_IA_tij_F2G2reg
            self.X_IA_gb2_F2
            self.X_IA_gb2_G2
            self.X_IA_gb2_S2F2
            self.X_IA_gb2_S2fe
            self.X_IA_gb2_S2he
            self.X_IA_gb2_S2G2

        if self.todo_dict['OV']: 
            self.X_OV

        if self.todo_dict['kPol']:
            self.X_kP1
            self.X_kP2
            self.X_kP3

        if self.todo_dict['RSD']:
            self.X_RSDA
            self.X_RSDB
        
    @property
    def k_original(self):
        return self.__k_original
    
    @property
    def k_extrap(self):
        return self.__k_extrap
    
    @property
    def k_final(self):
        return self.__k_final

    @cached_property
    def X_spt(self):
        nu = -2
        p_mat = np.array([[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [2, -2, 2, 0],
                    [1, -1, 1, 0], [1, -1, 3, 0], [2, -2, 0, 1]])
        return scalar_stuff(p_mat, nu, self.N, self.m, self.eta_m, self.l, self.tau_l)

    @cached_property
    def X_lpt(self):
        nu = -2
        p_mat = np.array([[0, 0, 0, 0], [0, 0, 2, 0], [2, -2, 2, 0],
                    [1, -1, 1, 0], [1, -1, 3, 0], [0, 0, 4, 0], [2, -2, 0, 1]])
        return scalar_stuff(p_mat, nu, self.N, self.m, self.eta_m, self.l, self.tau_l)

    @cached_property
    def X_sptG(self):
        nu = -2
        p_mat = np.array([[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [2, -2, 2, 0],
                        [1, -1, 1, 0], [1, -1, 3, 0], [2, -2, 0, 1]])
        return scalar_stuff(p_mat, nu, self.N, self.m, self.eta_m, self.l, self.tau_l)
    
    @cached_property
    def X_cleft(self):
        nu = -2
        p_mat = np.array([[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 4, 0], [1, -1, 1, 0], [1, -1, 3, 0], [-1, 1, 1, 0],
                        [-1, 1, 3, 0]])
        return scalar_stuff(p_mat, nu, self.N, self.m, self.eta_m, self.l, self.tau_l)
    
    @cached_property
    def X_IA_E(self):
        hE_tab, _ = IA_tt()
        p_mat_E = hE_tab[:, [0, 1, 5, 6, 7, 8, 9]]
        return tensor_stuff(p_mat_E, self.N, self.m, self.eta_m, self.l, self.tau_l)
    
    @cached_property
    def X_IA_B(self):
        _, hB_tab = IA_tt()
        p_mat_B = hB_tab[:, [0, 1, 5, 6, 7, 8, 9]]
        return tensor_stuff(p_mat_B, self.N, self.m, self.eta_m, self.l, self.tau_l)
    
    @cached_property
    def X_IA_A(self):
        IA_A_tab = IA_A()
        p_mat_A = IA_A_tab[:, [0, 1, 5, 6, 7, 8, 9]]
        return tensor_stuff(p_mat_A, self.N, self.m, self.eta_m, self.l, self.tau_l)

    @cached_property
    def X_IA_DEE(self):
        IA_DEE_tab = IA_DEE()
        p_mat_DEE = IA_DEE_tab[:, [0, 1, 5, 6, 7, 8, 9]]
        return tensor_stuff(p_mat_DEE, self.N, self.m, self.eta_m, self.l, self.tau_l)
    
    @cached_property
    def X_IA_DBB(self):
        IA_DBB_tab = IA_DBB()
        p_mat_DBB = IA_DBB_tab[:, [0, 1, 5, 6, 7, 8, 9]]
        return tensor_stuff(p_mat_DBB, self.N, self.m, self.eta_m, self.l, self.tau_l)
    
    @cached_property
    def X_IA_deltaE1(self):
        IA_deltaE1_tab = IA_deltaE1()
        return tensor_stuff(IA_deltaE1_tab[:, [0, 1, 5, 6, 7, 8, 9]], self.N, self.m, self.eta_m, self.l, self.tau_l)

    @cached_property
    def X_IA_0E0E(self):
        IA_0E0E_tab = IA_0E0E()
        return tensor_stuff(IA_0E0E_tab[:, [0, 1, 5, 6, 7, 8, 9]], self.N, self.m, self.eta_m, self.l, self.tau_l)

    @cached_property
    def X_IA_0B0B(self):
        IA_0B0B_tab = IA_0B0B()
        return tensor_stuff(IA_0B0B_tab[:, [0, 1, 5, 6, 7, 8, 9]], self.N, self.m, self.eta_m, self.l, self.tau_l)

    @cached_property
    def X_IA_gb2_fe(self):
        IA_gb2_fe_tab = IA_gb2_fe()
        p_mat_gb2_fe = IA_gb2_fe_tab[:, [0, 1, 5, 6, 7, 8, 9]]
        return tensor_stuff(p_mat_gb2_fe, self.N, self.m, self.eta_m, self.l, self.tau_l)

    @cached_property
    def X_IA_gb2_he(self):
        IA_gb2_he_tab = IA_gb2_he()
        p_mat_gb2_he = IA_gb2_he_tab[:, [0, 1, 5, 6, 7, 8, 9]]
        return tensor_stuff(p_mat_gb2_he, self.N, self.m, self.eta_m, self.l, self.tau_l)
    
    @cached_property
    def X_IA_tij_feG2(self):
        IA_tij_feG2_tab = IA_tij_feG2()
        p_mat_tij_feG2 = IA_tij_feG2_tab[:, [0, 1, 5, 6, 7, 8, 9]]
        return tensor_stuff(p_mat_tij_feG2, self.N, self.m, self.eta_m, self.l, self.tau_l)

    @cached_property
    def X_IA_tij_heG2(self):
        IA_tij_heG2_tab = IA_tij_heG2()
        p_mat_tij_heG2 = IA_tij_heG2_tab[:, [0, 1, 5, 6, 7, 8, 9]]
        return tensor_stuff(p_mat_tij_heG2, self.N, self.m, self.eta_m, self.l, self.tau_l) 
    
    @cached_property
    def X_IA_tij_F2F2(self):
        IA_tij_F2F2_tab = IA_tij_F2F2()
        p_mat_tij_F2F2 = IA_tij_F2F2_tab[:, [0, 1, 5, 6, 7, 8, 9]]
        return tensor_stuff(p_mat_tij_F2F2, self.N, self.m, self.eta_m, self.l, self.tau_l)

    @cached_property
    def X_IA_tij_G2G2(self):
        IA_tij_G2G2_tab = IA_tij_G2G2()
        p_mat_tij_G2G2 = IA_tij_G2G2_tab[:, [0, 1, 5, 6, 7, 8, 9]]
        return tensor_stuff(p_mat_tij_G2G2, self.N, self.m, self.eta_m, self.l, self.tau_l)
    
    @cached_property
    def X_IA_tij_F2G2(self):
        IA_tij_F2G2_tab = IA_tij_F2G2()
        p_mat_tij_F2G2 = IA_tij_F2G2_tab[:, [0, 1, 5, 6, 7, 8, 9]]
        return tensor_stuff(p_mat_tij_F2G2, self.N, self.m, self.eta_m, self.l, self.tau_l)
    
    @cached_property
    def X_IA_tij_F2G2reg(self):
        IA_tij_F2G2reg_tab =IA_tij_F2G2reg()
        p_mat_tij_F2G2reg_tab = IA_tij_F2G2reg_tab[:, [0, 1, 5, 6, 7, 8, 9]]
        return tensor_stuff(p_mat_tij_F2G2reg_tab, self.N, self.m, self.eta_m, self.l, self.tau_l)

    @cached_property
    def X_IA_gb2_F2(self):
        IA_gb2_F2_tab = IA_gb2_F2()
        p_mat_gb2_F2 = IA_gb2_F2_tab[:, [0, 1, 5, 6, 7, 8, 9]]
        return tensor_stuff(p_mat_gb2_F2, self.N, self.m, self.eta_m, self.l, self.tau_l)

    @cached_property
    def X_IA_gb2_G2(self):
        IA_gb2_G2_tab = IA_gb2_G2()
        p_mat_gb2_G2 = IA_gb2_G2_tab[:, [0, 1, 5, 6, 7, 8, 9]]
        return tensor_stuff(p_mat_gb2_G2, self.N, self.m, self.eta_m, self.l, self.tau_l)

    @cached_property
    def X_IA_gb2_S2F2(self):
        IA_gb2_S2F2_tab = IA_gb2_S2F2()
        p_mat_gb2_S2F2 = IA_gb2_S2F2_tab[:, [0, 1, 5, 6, 7, 8, 9]]
        return tensor_stuff(p_mat_gb2_S2F2, self.N, self.m, self.eta_m, self.l, self.tau_l)

    @cached_property
    def X_IA_gb2_S2fe(self):
        IA_gb2_S2fe_tab = IA_gb2_S2fe()
        p_mat_gb2_S2fe = IA_gb2_S2fe_tab[:, [0, 1, 5, 6, 7, 8, 9]]
        return tensor_stuff(p_mat_gb2_S2fe, self.N, self.m, self.eta_m, self.l, self.tau_l)

    @cached_property
    def X_IA_gb2_S2he(self):
        IA_gb2_S2he_tab = IA_gb2_S2he()
        p_mat_gb2_S2he = IA_gb2_S2he_tab[:, [0, 1, 5, 6, 7, 8, 9]]
        return tensor_stuff(p_mat_gb2_S2he, self.N, self.m, self.eta_m, self.l, self.tau_l)
    
    @cached_property
    def X_IA_gb2_S2G2(self):
        IA_gb2_S2G2_tab = IA_gb2_S2G2()
        p_mat_gb2_S2G2 = IA_gb2_S2G2_tab[:, [0, 1, 5, 6, 7, 8, 9]]
        return tensor_stuff(p_mat_gb2_S2G2, self.N, self.m, self.eta_m, self.l, self.tau_l)

    @cached_property
    def X_OV(self):
        OV_tab = OV()
        p_mat = OV_tab[:, [0, 1, 5, 6, 7, 8, 9]]
        return tensor_stuff(p_mat, self.N, self.m, self.eta_m, self.l, self.tau_l)

    @cached_property
    def X_kP1(self):
        tab1, _, _ = kPol()
        p_mat = tab1[:, [0, 1, 5, 6, 7, 8, 9]]
        return tensor_stuff(p_mat, self.N, self.m, self.eta_m, self.l, self.tau_l)

    @cached_property
    def X_kP2(self):
        _, tab2, _ = kPol()
        p_mat = tab2[:, [0, 1, 5, 6, 7, 8, 9]]
        return tensor_stuff(p_mat, self.N, self.m, self.eta_m, self.l, self.tau_l)

    @cached_property
    def X_kP3(self):
        _, _, tab3 = kPol()
        p_mat = tab3[:, [0, 1, 5, 6, 7, 8, 9]]
        return tensor_stuff(p_mat, self.N, self.m, self.eta_m, self.l, self.tau_l)
    
    @cached_property
    def X_RSDA(self):
        tabA, self.A_coeff = RSDA()
        p_mat = tabA[:, [0, 1, 5, 6, 7, 8, 9]]
        return tensor_stuff(p_mat, self.N, self.m, self.eta_m, self.l, self.tau_l)
    
    @cached_property
    def X_RSDB(self):
        tabB, self.B_coeff = RSDB()
        p_mat = tabB[:, [0, 1, 5, 6, 7, 8, 9]]
        return tensor_stuff(p_mat, self.N, self.m, self.eta_m, self.l, self.tau_l)





    def validate_params(self, P, **kwargs):
        if (P is None):
            raise ValueError('You must provide an input power spectrum array.')
        if (len(P) == 0):
            raise ValueError('You must provide an input power spectrum array.')
        if (len(P) != len(self.k_original)):
            raise ValueError(f'Input k and P arrays must have the same size. P:{len(P)}, K:{len(self.k_final)}')
            
        if (np.all(P == 0.0)):
            raise ValueError('Your input power spectrum array is all zeros.')

        P_window = kwargs.get('P_window', np.array([]))
        C_window = kwargs.get('C_window', None)

        if P_window is not None and P_window.size > 0:
            maxP = (log(self.k_final[-1]) - log(self.k_final[0])) / 2
            if len(P_window) != 2:
                raise ValueError(f'P_window must be a tuple of two values.')
            if P_window[0] > maxP or P_window[1] > maxP:
                raise ValueError(f'P_window value is too large. Decrease to less than {(log(self.k_final[-1]) - log(self.k_final[0])) / 2} to avoid over tapering.')

        if C_window is not None:
            if C_window < 0 or C_window > 1:
                raise ValueError('C_window must be between 0 and 1.')

        return None
    
    ############## ABSTRACTED BEHAVIOR METHODS ##############
    def _apply_extrapolation(self, *args):
        """ Applies extrapolation to multiple variables at once """
        if not self.extrap:
            return args if len(args) > 1 else args[0]  # Avoid returning a tuple for a single value
        return [self.EK.PK_original(var)[1] for var in args] if len(args) > 1 else self.EK.PK_original(args[0])[1]
    
    def _hash_arrays(self, arrays):
        """Helper function to create a hash from multiple numpy arrays or scalars"""
        if isinstance(arrays, tuple):
            return tuple(hash(arr.tobytes()) if isinstance(arr, np.ndarray) else hash(arr) 
                        for arr in arrays)
        return hash(arrays.tobytes()) if isinstance(arrays, np.ndarray) else hash(arrays)


    def _compute_one_loop_terms(self, P, X, P_window=None, C_window=None):
        """ Computes the one-loop power spectrum terms """
        nu = -2
        one_loop_coef = np.array([2 * 1219 / 1470., 2 * 671 / 1029., 2 * 32 / 1715., 
                                2 * 1 / 3., 2 * 62 / 35., 2 * 8 / 35., 1 / 3.])
    
        Ps, mat = self._compute_J_k_scalar(P, X, nu, P_window=P_window, C_window=C_window)
    
        P22_mat = np.multiply(one_loop_coef, np.transpose(mat))
        P22 = np.sum(P22_mat, 1)
        P13 = P_13_reg(self.k_extrap, Ps)
        P_1loop = P22 + P13

        return P_1loop, Ps, mat

    def _compute_J_k_scalar(self, P, X, nu, P_window=None, C_window=None):
        """Wrapper function to compute J_k_scalar with caching"""
        p_hash = self._hash_arrays(P)
        x_hash = self._hash_arrays(X)
    
        if P_window is not None:
            p_window_hash = self._hash_arrays(P_window)
        else:
            p_window_hash = None
        
        cache_key = ("J_k_scalar", p_hash, x_hash, nu, p_window_hash, C_window)
    
        if cache_key in self.cache:
            return self.cache[cache_key]

        result = self.J_k_scalar(P, X, nu, P_window, C_window)
        self.cache[cache_key] = result
        return result

    def _compute_J_k_tensor(self, P, X, P_window=None, C_window=None):
        """Wrapper function to compute J_k_tensor with caching"""
        # Create hashable versions of arrays and tuples
        p_hash = self._hash_arrays(P)
        x_hash = self._hash_arrays(X)
    
        if P_window is not None:
            p_window_hash = self._hash_arrays(P_window)
        else:
            p_window_hash = None
        
        cache_key = ("J_k_tensor", p_hash, x_hash, p_window_hash, C_window)
    
        if cache_key in self.cache:
            return self.cache[cache_key]

        result = self.J_k_tensor(P, X, P_window, C_window)
        self.cache[cache_key] = result
        return result



    ### Top-level functions to output final quantities ###
    
    def one_loop_dd(self, P, P_window=None, C_window=None):
        self.validate_params(P, P_window=P_window, C_window=C_window)

        # routine for one-loop spt calculations

        # get the roundtrip Fourier power spectrum, i.e. P=IFFT[FFT[P]]
        # get the matrix for each J_k component
        P_1loop, Ps, mat = self._compute_one_loop_terms(P, self.X_spt, P_window=P_window, C_window=C_window)

        

        if (self.todo_dict['dd_bias']):
            # if dd_bias is in to_do, this function acts like one_loop_dd_bias

            # Quadraric bias Legendre components
            # See eg section B of Baldauf+ 2012 (arxiv: 1201.4827)
            # Note pre-factor convention is not standardized
            # Returns relevant correlations (including contraction factors),
            # but WITHOUT bias values and other pre-factors.
            # Uses standard "full initialization" of J terms
            sig4 = np.trapz(self.k_extrap ** 3 * Ps ** 2, x=np.log(self.k_extrap)) / (2. * pi ** 2)
            self.sig4 = sig4
            # sig4 much more accurate when calculated in logk, especially for low-res input.
            Pd1d2 = 2. * (17. / 21 * mat[0, :] + mat[4, :] + 4. / 21 * mat[1, :])
            Pd2d2 = 2. * (mat[0, :])
            Pd1s2 = 2. * (8. / 315 * mat[0, :] + 4. / 15 * mat[4, :] + 254. / 441 * mat[1, :] + 2. / 5 * mat[5,
                                                                                                         :] + 16. / 245 * mat[
                                                                                                                          2,
                                                                                                                          :])
            Pd2s2 = 2. * (2. / 3 * mat[1, :])
            Ps2s2 = 2. * (4. / 45 * mat[0, :] + 8. / 63 * mat[1, :] + 8. / 35 * mat[2, :])
            Ps, P_1loop, Pd1d2, Pd2d2, Pd1s2, Pd2s2, Ps2s2 = self._apply_extrapolation(Ps, P_1loop, Pd1d2, Pd2d2, Pd1s2, Pd2s2, Ps2s2)

            return P_1loop, Ps, Pd1d2, Pd2d2, Pd1s2, Pd2s2, Ps2s2, sig4

        P_1loop, Ps = self._apply_extrapolation(P_1loop, Ps)

        return P_1loop, Ps

    # def get_P1loop(self, P, P_window=None, C_window=None):
    #     P22 = np.sum(P22_mat, 1)
    #     P13 = P_13_reg(self.k_extrap, Ps)
    #     P_1loop = P22 + P13


    
    def one_loop_dd_bias(self, P, P_window=None, C_window=None):
        self.validate_params(P, P_window=P_window, C_window=C_window)

        # routine for one-loop spt calculations

        # get the roundtrip Fourier power spectrum, i.e. P=IFFT[FFT[P]]
        # get the matrix for each J_k component
        P_1loop, Ps, mat = self._compute_one_loop_terms(P, self.X_spt, P_window=P_window, C_window=C_window)

        # Quadraric bias Legendre components
        # See eg section B of Baldauf+ 2012 (arxiv: 1201.4827)
        # Note pre-factor convention is not standardized
        # Returns relevant correlations (including contraction factors),
        # but WITHOUT bias values and other pre-factors.
        # Uses standard "full initialization" of J terms
        sig4 = np.trapz(self.k_extrap ** 3 * Ps ** 2, x=np.log(self.k_extrap)) / (2. * pi ** 2)
        Pd1d2 = 2. * (17. / 21 * mat[0, :] + mat[4, :] + 4. / 21 * mat[1, :])
        Pd2d2 = 2. * (mat[0, :])
        Pd1s2 = 2. * (8. / 315 * mat[0, :] + 4. / 15 * mat[4, :] + 254. / 441 * mat[1, :] + 2. / 5 * mat[5,
                                                                                                     :] + 16. / 245 * mat[
                                                                                                                      2,
                                                                                                                      :])
        Pd2s2 = 2. * (2. / 3 * mat[1, :])
        Ps2s2 = 2. * (4. / 45 * mat[0, :] + 8. / 63 * mat[1, :] + 8. / 35 * mat[2, :])

        Ps, P_1loop, Pd1d2, Pd2d2, Pd1s2, Pd2s2, Ps2s2 = self._apply_extrapolation(Ps, P_1loop, Pd1d2, Pd2d2, Pd1s2, Pd2s2, Ps2s2)

        #			return P_1loop, Pd1d2, Pd2d2, Pd1s2, Pd2s2, Ps2s2, sig4, Ps #original
        return P_1loop, Ps, Pd1d2, Pd2d2, Pd1s2, Pd2s2, Ps2s2, sig4  # new,for consistency

    
    def one_loop_dd_bias_b3nl(self, P, P_window=None, C_window=None):
        self.validate_params(P, P_window=P_window, C_window=C_window)

        # routine for one-loop spt calculations

        # get the roundtrip Fourier power spectrum, i.e. P=IFFT[FFT[P]]
        # get the matrix for each J_k component
        P_1loop, Ps, mat = self._compute_one_loop_terms(P, self.X_spt, P_window=P_window, C_window=C_window)

        sig4 = np.trapz(self.k_extrap ** 3 * Ps ** 2, x=np.log(self.k_extrap)) / (2. * pi ** 2)
        Pd1d2 = 2. * (17. / 21 * mat[0, :] + mat[4, :] + 4. / 21 * mat[1, :])
        Pd2d2 = 2. * (mat[0, :])
        Pd1s2 = 2. * (8. / 315 * mat[0, :] + 4. / 15 * mat[4, :] + 254. / 441 * mat[1, :] + 2. / 5 * mat[5,
                                                                                                     :] + 16. / 245 * mat[
                                                                                                                      2,
                                                                                                                      :])
        Pd2s2 = 2. * (2. / 3 * mat[1, :])
        Ps2s2 = 2. * (4. / 45 * mat[0, :] + 8. / 63 * mat[1, :] + 8. / 35 * mat[2, :])
        sig3nl = Y1_reg_NL(self.k_extrap, Ps)

        Ps, P_1loop, Pd1d2, Pd2d2, Pd1s2, Pd2s2, Ps2s2, sig3nl = self._apply_extrapolation(Ps, P_1loop, Pd1d2, Pd2d2, Pd1s2, Pd2s2, Ps2s2, sig3nl)

        #			return P_1loop, Pd1d2, Pd2d2, Pd1s2, Pd2s2, Ps2s2, sig4, Ps #original
        return P_1loop, Ps, Pd1d2, Pd2d2, Pd1s2, Pd2s2, Ps2s2, sig4, sig3nl  # new,for consistency

    
    def one_loop_dd_bias_lpt_NL(self, P, P_window=None, C_window=None):
        self.validate_params(P, P_window=P_window, C_window=C_window)

        # get the roundtrip Fourier power spectrum, i.e. P=IFFT[FFT[P]]
        # get the matrix for each J_k component
        _, Ps, mat = self._compute_one_loop_terms(P, self.X_lpt, P_window=P_window, C_window=C_window)

        [j000, j002, j2n22, j1n11, j1n13, j004, j2n20] = [mat[0, :], mat[1, :], mat[2, :], mat[3, :], mat[4, :],
                                                          mat[5, :], mat[6, :]]

        P22 = 2. * ((1219. / 1470.) * j000 + (671. / 1029.) * j002 + (32. / 1715.) * j004 + (1. / 3.) * j2n22 + (
                62. / 35.) * j1n11 + (8. / 35.) * j1n13 + (1. / 6.) * j2n20)

        sig4 = np.trapz(self.k_extrap ** 3 * Ps ** 2, x=np.log(self.k_extrap)) / (2. * pi ** 2)

        X1 = ((144. / 245.) * j000 - (176. / 343.) * j002 - (128. / 1715.) * j004 + (16. / 35.) * j1n11 - (
                16. / 35.) * j1n13)
        X2 = ((16. / 21.) * j000 - (16. / 21.) * j002 + (16. / 35.) * j1n11 - (16. / 35.) * j1n13)
        X3 = (50. / 21.) * j000 + 2. * j1n11 - (8. / 21.) * j002
        X4 = (34. / 21.) * j000 + 2. * j1n11 + (8. / 21.) * j002
        X5 = j000

        Y1 = Y1_reg_NL(self.k_extrap, Ps)
        Y2 = Y2_reg_NL(self.k_extrap, Ps)

        Pb1L = X1 + Y1
        Pb1L_2 = X2 + Y2
        Pb1L_b2L = X3
        Pb2L = X4
        Pb2L_2 = X5

        Ps, Pb1L, Pb1L_2, Pb1L_b2L, Pb2L, Pb2L_2 = self._apply_extrapolation(Ps, Pb1L, Pb1L_2, Pb1L_b2L, Pb2L, Pb2L_2)
        X1, X2, X3, X4, X5, Y1, Y2 = self._apply_extrapolation(X1, X2, X3, X4, X5, Y1, Y2)

        return Ps, Pb1L, Pb1L_2, Pb1L_b2L, Pb2L, Pb2L_2, sig4

    
    def cleft_Q_R(self, P, P_window=None, C_window=None):
        self.validate_params(P, P_window=P_window, C_window=C_window)


        nu_arr = -2
        # get the roundtrip Fourier power spectrum, i.e. P=IFFT[FFT[P]]
        # get the matrix for each J_k component
        Ps, mat = self.J_k_scalar(P, self.X_cleft, nu_arr, P_window=P_window, C_window=C_window)

        [j000, j002, j004, j1n11, j1n13, jn111, jn113] = [mat[0, :], mat[1, :], mat[2, :], mat[3, :], mat[4, :],
                                                          mat[5, :], mat[6, :]]

        FQ1 = (8./15.)*j000 - (16./21.)*j002 + (8./35.)*j004
        FQ2 = (4./5.)*j000 - (4./7.)*j002 - (8./35.)*j004 + (2./5.)*j1n11 - (2./5.)*j1n13 + (2./5.)*jn111 - (2./5.)*jn113
        FQ5 = (2./3.)*j000 - (2./3.)*j002 + (2./5.)*jn111 - (2./5.)*jn113
        FQ8 = (2./3.)*j000 - (2./3.)*j002
        FQs2 = (-4./15.)*j000 + (20./21.)*j002 - (24./35.)*j004


        FR1 = cleft_Z1(self.k_extrap, Ps)
        FR2 = cleft_Z2(self.k_extrap, Ps)

        # ipdb.set_trace()

        Ps_ep, FQ1_ep, FQ2_ep, FQ5_ep, FQ8_ep, FQs2_ep, FR1_ep, FR2_ep = self._apply_extrapolation(Ps, FQ1, FQ2, FQ5, FQ8, FQs2, FR1, FR2)

        return FQ1_ep,FQ2_ep,FQ5_ep,FQ8_ep,FQs2_ep,FR1_ep,FR2_ep,self.k_extrap,FR1,FR2

    
    def IA_tt(self, P, P_window=None, C_window=None):
        self.validate_params(P, P_window=P_window, C_window=C_window)
        P_E, A = self._compute_J_k_tensor(P, self.X_IA_E, P_window=P_window, C_window=C_window)
        P_B, A = self._compute_J_k_tensor(P, self.X_IA_B, P_window=P_window, C_window=C_window)
        P_B, P_E = self._apply_extrapolation(P_B, P_E)
        return 2. * P_E, 2. * P_B

    ## eq 21 EE; eq 21 BB

    
    def IA_mix(self, P, P_window=None, C_window=None):
        self.validate_params(P, P_window=P_window, C_window=C_window)
        P_A, A = self._compute_J_k_tensor(P, self.X_IA_A, P_window=P_window, C_window=C_window)
        P_A = self._apply_extrapolation(P_A)

        P_Btype2 = P_IA_B(self.k_original, P)

        P_DEE, A = self._compute_J_k_tensor(P, self.X_IA_DEE, P_window=P_window, C_window=C_window)
        P_DBB, A = self._compute_J_k_tensor(P, self.X_IA_DBB, P_window=P_window, C_window=C_window)
        P_DBB, P_DEE = self._apply_extrapolation(P_DBB, P_DEE)

        return 2 * P_A, 4 * P_Btype2, 2 * P_DEE, 2 * P_DBB

    ## eq 18; eq 19; eq 27 EE; eq 27 BB

    
    def IA_ta(self, P, P_window=None, C_window=None):
        P_deltaE1 = self.get_P_deltaE1(P, P_window=P_window, C_window=C_window)
        P_deltaE2 = self.get_P_deltaE2(P)
        P_0E0E = self.get_P_0E0E(P, P_window=P_window, C_window=C_window)
        P_0B0B = self.get_P_0B0B(P, P_window=P_window, C_window=C_window)
        return P_deltaE1, P_deltaE2, P_0E0E, P_0B0B
    
    def get_P_deltaE1(self, P, P_window=None, C_window=None):
        if "P_deltaE1" in self.term_cache: return self.term_cache["P_deltaE1"]
        P_deltaE1, A = self._compute_J_k_tensor(P, self.X_IA_deltaE1, P_window=P_window, C_window=C_window)
        P_deltaE1 = self._apply_extrapolation(P_deltaE1)
        self.term_cache["P_deltaE1"] = 2 * P_deltaE1
        return 2 * P_deltaE1
    def get_P_deltaE2(self, P):
        if "P_deltaE2" in self.term_cache: return self.term_cache["P_deltaE2"]
        P_deltaE2 = P_IA_deltaE2(self.k_original, P)
        #Add extrap?
        self.term_cache["P_deltaE2"] = 2 * P_deltaE2
        return 2 * P_deltaE2
    def get_P_0E0E(self, P, P_window=None, C_window=None):
        if "P_0E0E" in self.term_cache: return self.term_cache["P_0E0E"]
        P_0E0E, A = self._compute_J_k_tensor(P, self.X_IA_0E0E, P_window=P_window, C_window=C_window)
        P_0E0E = self._apply_extrapolation(P_0E0E)
        self.term_cache["P_0E0E"] = P_0E0E
        return P_0E0E
    def get_P_0B0B(self, P, P_window=None, C_window=None):
        if "P_0B0B" in self.term_cache: return self.term_cache["P_0B0B"]
        P_0B0B, A = self._compute_J_k_tensor(P, self.X_IA_0B0B, P_window=P_window, C_window=C_window)
        P_0B0B = self._apply_extrapolation(P_0B0B)
        self.term_cache["P_0B0B"] = P_0B0B
        return P_0B0B

    ## eq 12 (line 2); eq 12 (line 3); eq 15 EE; eq 15 BB

    
    def IA_der(self, P, P_window=None, C_window=None):
        self.validate_params(P, P_window=P_window, C_window=C_window)
        P_der = (self.k_original**2)*P
        return P_der
    
    
    def IA_ct(self,P,P_window=None, C_window=None):
        self.validate_params(P, P_window=P_window, C_window=C_window)
        P_feG2, A = self._compute_J_k_tensor(P,self.X_IA_tij_feG2, P_window=P_window, C_window=C_window)
        P_heG2, A = self._compute_J_k_tensor(P,self.X_IA_tij_heG2, P_window=P_window, C_window=C_window)
        P_F2F2, A = self._compute_J_k_tensor(P,self.X_IA_tij_F2F2, P_window=P_window, C_window=C_window)
        P_G2G2, A = self._compute_J_k_tensor(P,self.X_IA_tij_G2G2, P_window=P_window, C_window=C_window)
        P_F2G2, A = self._compute_J_k_tensor(P,self.X_IA_tij_F2G2, P_window=P_window, C_window=C_window)
        P_feG2, P_heG2, P_F2F2, P_G2G2, P_F2G2 = self._apply_extrapolation(P_feG2, P_heG2, P_F2F2, P_G2G2, P_F2G2)
        P_A00E,A,B,C = self.IA_ta(P, P_window=P_window, C_window=C_window)
        P_A0E2,D,E,F = self.IA_mix(P,P_window=P_window, C_window=C_window)
        P_13F = P_IA_13F(self.k_original, P)
        P_13G = P_IA_13G(self.k_original,P,)
        nu=-2
        Ps, mat = self._compute_J_k_scalar(P, self.X_spt, nu, P_window=P_window, C_window=C_window)
        one_loop_coef = np.array(
            [2 * 1219 / 1470., 2 * 671 / 1029., 2 * 32 / 1715., 2 * 1 / 3., 2 * 62 / 35., 2 * 8 / 35., 1 / 3.])
        P22_mat = np.multiply(one_loop_coef, np.transpose(mat))
        P_22F = np.sum(P22_mat, 1)

        one_loop_coefG= np.array(
            [2*1003/1470, 2*803/1029, 2*64/1715, 2*1/3, 2*58/35, 2*12/35, 1/3])
        PsG, matG = self._compute_J_k_scalar(P, self.X_sptG, nu, P_window=P_window, C_window=C_window)
        P22G_mat = np.multiply(one_loop_coefG, np.transpose(matG))
        P_22G = np.sum(P22G_mat, 1)
        P_22F, P_22G = self._apply_extrapolation(P_22F, P_22G)
        P_tEtE = P_F2F2+P_G2G2-2*P_F2G2
        P_0tE = P_22G-P_22F+P_13G-P_13F
        P_0EtE = np.subtract(P_feG2,(1/2)*P_A00E)
        P_E2tE = np.subtract(P_heG2,(1/2)*P_A0E2)
            
        return 2*P_0tE,2*P_0EtE,2*P_E2tE,2*P_tEtE
    
    
    def IA_ctbias(self,P,P_window=None, C_window=None):
        self.validate_params(P, P_window=P_window, C_window=C_window)
        P_F2, A = self._compute_J_k_tensor(P,self.X_IA_gb2_F2, P_window=P_window, C_window=C_window)
        P_G2, A = self._compute_J_k_tensor(P,self.X_IA_gb2_G2, P_window=P_window, C_window=C_window)
        P_F2, P_G2 = self._apply_extrapolation(P_F2, P_G2)
        P_d2tE = P_G2-P_F2
        P_S2F2, A = self._compute_J_k_tensor(P, self.X_IA_gb2_S2F2, P_window=P_window, C_window=C_window)

        #P_13S2F2 = P_IA_13S2F2(self.k_original, P)

        P_S2G2, A = self._compute_J_k_tensor(P, self.X_IA_gb2_S2G2, P_window=P_window, C_window=C_window)
        P_S2F2, P_S2G2 = self._apply_extrapolation(P_S2F2, P_S2G2)
        P_s2tE=P_S2G2-P_S2F2

        return 2*P_d2tE,2*P_s2tE
    

    
    def IA_gb2(self,P,P_window=None, C_window=None):
        self.validate_params(P, P_window=P_window, C_window=C_window)
        P_fe, A = self._compute_J_k_tensor(P,self.X_IA_gb2_fe, P_window=P_window, C_window=C_window)
        P_he, A = self._compute_J_k_tensor(P,self.X_IA_gb2_he, P_window=P_window, C_window=C_window)
        P_F2, A = self._compute_J_k_tensor(P,self.X_IA_gb2_F2, P_window=P_window, C_window=C_window)
        P_fe, P_he, P_F2 = self._apply_extrapolation(P_fe, P_he, P_F2)
        sig4 = np.trapz(self.k_original ** 3 * P ** 2, x=np.log(self.k_original)) / (2. * pi ** 2)
        P_gb2sij = P_F2
        P_gb2sij2 = P_he
        P_gb2dsij = P_fe
        return 2*P_gb2sij, 2*P_gb2dsij, 2*P_gb2sij2
    
    
    def IA_d2(self,P,P_window=None, C_window=None):
        self.validate_params(P, P_window=P_window, C_window=C_window)
        P_fe, A = self._compute_J_k_tensor(P,self.X_IA_gb2_fe, P_window=P_window, C_window=C_window)
        P_he, A = self._compute_J_k_tensor(P,self.X_IA_gb2_he, P_window=P_window, C_window=C_window)
        P_F2, A = self._compute_J_k_tensor(P,self.X_IA_gb2_F2, P_window=P_window, C_window=C_window)
        P_fe, P_he, P_F2 = self._apply_extrapolation(P_fe, P_he, P_F2)
        sig4 = np.trapz(self.k_original ** 3 * P ** 2, x=np.log(self.k_original)) / (2. * pi ** 2)
        P_d2E = P_F2
        P_d20E = P_he
        P_d2E2 = P_fe
        return 2*P_d2E, 2*P_d20E, 2*P_d2E2
    
    
    def IA_s2(self, P, P_window=None, C_window=None):
        self.validate_params(P, P_window=P_window, C_window=C_window)
        P_S2F2, A = self._compute_J_k_tensor(P, self.X_IA_gb2_S2F2, P_window=P_window, C_window=C_window)
        #P_13S2F2 = P_IA_13S2F2(self.k_original, P)

        P_S2fe, A = self._compute_J_k_tensor(P, self.X_IA_gb2_S2fe, P_window=P_window, C_window=C_window)
        P_S2he, A = self._compute_J_k_tensor(P, self.X_IA_gb2_S2he, P_window=P_window, C_window=C_window)
        P_S2F2, P_S2fe, P_S2he = self._apply_extrapolation(P_S2F2, P_S2fe, P_S2he)
        #THIS LINE RIGHT HERE
        P_s2E=P_S2F2#+2*P_13S2F2
        P_s20E=P_S2fe
        P_s2E2=P_S2he
        return 2*P_s2E, 2*P_s20E, 2*P_s2E2

    
    def OV(self, P, P_window=None, C_window=None):
        self.validate_params(P, P_window=P_window, C_window=C_window)
        P, A = self._compute_J_k_tensor(P, self.X_OV, P_window=P_window, C_window=C_window)
        P = self._apply_extrapolation(P)
        return P * (2 * pi) ** 2

    
    def kPol(self, P, P_window=None, C_window=None):
        self.validate_params(P, P_window=P_window, C_window=C_window)
        P1, A = self._compute_J_k_tensor(P, self.X_kP1, P_window=P_window, C_window=C_window)
        P2, A = self._compute_J_k_tensor(P, self.X_kP2, P_window=P_window, C_window=C_window)
        P3, A = self._compute_J_k_tensor(P, self.X_kP3, P_window=P_window, C_window=C_window)
        P1, P2, P3 = self._apply_extrapolation(P1, P2, P3)
        return P1 / (80 * pi ** 2), P2 / (160 * pi ** 2), P3 / (80 * pi ** 2)


    def RSD_components(self, P, f, P_window=None, C_window=None):
        self.validate_params(P, P_window=P_window, C_window=C_window)
        _, A = self._compute_J_k_tensor(P, self.X_RSDA, P_window=P_window, C_window=C_window)

        A1 = np.dot(self.A_coeff[:, 0], A) + f * np.dot(self.A_coeff[:, 1], A) + f ** 2 * np.dot(self.A_coeff[:, 2], A)
        A3 = np.dot(self.A_coeff[:, 3], A) + f * np.dot(self.A_coeff[:, 4], A) + f ** 2 * np.dot(self.A_coeff[:, 5], A)
        A5 = np.dot(self.A_coeff[:, 6], A) + f * np.dot(self.A_coeff[:, 7], A) + f ** 2 * np.dot(self.A_coeff[:, 8], A)

        _, B = self._compute_J_k_tensor(P, self.X_RSDB, P_window=P_window, C_window=C_window)

        B0 = np.dot(self.B_coeff[:, 0], B) + f * np.dot(self.B_coeff[:, 1], B) + f ** 2 * np.dot(self.B_coeff[:, 2], B)
        B2 = np.dot(self.B_coeff[:, 3], B) + f * np.dot(self.B_coeff[:, 4], B) + f ** 2 * np.dot(self.B_coeff[:, 5], B)
        B4 = np.dot(self.B_coeff[:, 6], B) + f * np.dot(self.B_coeff[:, 7], B) + f ** 2 * np.dot(self.B_coeff[:, 8], B)
        B6 = np.dot(self.B_coeff[:, 9], B) + f * np.dot(self.B_coeff[:, 10], B) + f ** 2 * np.dot(self.B_coeff[:, 11], B)

        A1, A3, A5, B0, B2, B4, B6 = self._apply_extrapolation(A1, A3, A5, B0, B2, B4, B6)

        P_Ap1 = RSD_ItypeII.P_Ap1(self.k_original, P, f)
        P_Ap3 = RSD_ItypeII.P_Ap3(self.k_original, P, f)
        P_Ap5 = RSD_ItypeII.P_Ap5(self.k_original, P, f)

        return A1, A3, A5, B0, B2, B4, B6, P_Ap1, P_Ap3, P_Ap5

    
    def RSD_ABsum_components(self, P, f, P_window=None, C_window=None):
        self.validate_params(P, P_window=P_window, C_window=C_window)
        A1, A3, A5, B0, B2, B4, B6, P_Ap1, P_Ap3, P_Ap5 = self.RSD_components(P, f, P_window, C_window)
        ABsum_mu2 = self.k_original * f * (A1 + P_Ap1) + (f * self.k_original) ** 2 * B0
        ABsum_mu4 = self.k_original * f * (A3 + P_Ap3) + (f * self.k_original) ** 2 * B2
        ABsum_mu6 = self.k_original * f * (A5 + P_Ap5) + (f * self.k_original) ** 2 * B4
        ABsum_mu8 = (f * self.k_original) ** 2 * B6

        return ABsum_mu2, ABsum_mu4, ABsum_mu6, ABsum_mu8

    
    def RSD_ABsum_mu(self, P, f, mu_n, P_window=None, C_window=None):
        self.validate_params(P, P_window=P_window, C_window=C_window)
        ABsum_mu2, ABsum_mu4, ABsum_mu6, ABsum_mu8 = self.RSD_ABsum_components(P, f, P_window, C_window)
        ABsum = ABsum_mu2 * mu_n ** 2 + ABsum_mu4 * mu_n ** 4 + ABsum_mu6 * mu_n ** 6 + ABsum_mu8 * mu_n ** 8
        return ABsum

    
    def IRres(self, P, L=0.2, h=0.67, rsdrag=135, P_window=None, C_window=None):
        self.validate_params(P, P_window=P_window, C_window=C_window)
        # based on script by M. Ivanov. See arxiv:1605.02149, eq 7.4

        # put this function in the typical fast-pt format, with minimal additional function calls.
        # We can also include a script to do additional things: e.g read in r_BAO from class/camb output
        # or calculate r_BAO from cosmological params.
        from scipy import interpolate
        k = self.k_original
        rbao = h * rsdrag * 1.027  # linear BAO scale
        # set up splining to create P_nowiggle
        kmin = k[0]
        kmax = k[-1]
        knode1 = pi / rbao
        knode2 = 2 * pi / rbao
        klogleft = np.arange(log(kmin), log(3.e-3), 0.2)
        klogright = np.arange(log(0.6), log(kmax), 0.085)
        klogright = np.hstack((log(knode1), log(knode2), klogright))
        kloglist = np.concatenate((klogleft, klogright))
        klist = np.exp(kloglist)

        # how to deal with extended k and P? which values should be used here? probably the extended versions?
        plin = interpolate.InterpolatedUnivariateSpline(k, P)
        logPs = np.log(plin(klist))
        logpsmooth = interpolate.InterpolatedUnivariateSpline(kloglist, logPs)

        def psmooth(x):
            return exp(logpsmooth(log(x)))

        def pw(x):
            return plin(x) - psmooth(x)

        # compute Sigma^2 and the tree-level IR-resummed PS
        import scipy.integrate as integrate

        Sigma = integrate.quad(lambda x: (4 * pi) * psmooth(x) * (
                1 - 3 * (2 * rbao * x * cos(x * rbao) + (-2 + rbao ** 2 * x ** 2) * sin(rbao * x)) / (
                x * rbao) ** 3) / (3 * (2 * pi) ** 3), kmin, L)[0]

        # speed up by using trap rule integration?
        # change to integration over log-k(?):
        # 		Sigma = integrate.quad(lambda x: x*(4*pi)*psmooth(x)*(1-3*(2*rbao*x*cos(x*rbao)+(-2+rbao**2*x**2)*sin(rbao*x))/(x*rbao)**3)/(3*(2*pi)**3), np.log(kmin), np.log(L))[0]
        def presum(x):
            return psmooth(x) + pw(x) * exp(-x ** 2 * Sigma)

        P = presum(k)
        out_1loop = self.one_loop_dd(P, P_window=P_window, C_window=C_window)[0]
        # p1loop = interpolate.InterpolatedUnivariateSpline(k,out_1loop) # is this necessary? out_1loop should already be at the correct k spacing
        return psmooth(k) + out_1loop + pw(k) * exp(-k ** 2 * Sigma) * (1 + Sigma * k ** 2)

    ######################################################################################
    ### functions that use the older version structures. ###
    
    def one_loop(self, P, P_window=None, C_window=None):

        return self.pt_simple.one_loop(P, P_window=P_window, C_window=C_window)

    
    def P_bias(self, P, P_window=None, C_window=None):

        return self.pt_simple.P_bias(P, P_window=P_window, C_window=C_window)

    ######################################################################################
    ### Core functions used by top-level functions ###
    def _cache_fourier_coefficients(self, P_b, C_window=None):
        """Cache and return Fourier coefficients for a given biased power spectrum"""
        from numpy.fft import rfft
    
        # Create cache key
        cache_key = ("fourier_coeffs", hash(P_b.tobytes()), C_window)
    
        if cache_key in self.c_cache:
            return self.c_cache[cache_key]
    
        # Calculate coefficients
        c_m_positive = rfft(P_b)
        c_m_positive[-1] = c_m_positive[-1] / 2.
        c_m_negative = np.conjugate(c_m_positive[1:])
        c_m = np.hstack((c_m_negative[::-1], c_m_positive)) / float(self.N)
    
        # Apply window if specified
        if C_window is not None:
            if self.verbose:
                print('windowing the Fourier coefficients')
            c_m = c_m * c_window(self.m, int(C_window * self.N // 2.))
    
        # Cache and return
        self.c_cache[cache_key] = c_m
        return c_m

    def _cache_convolution(self, c1, c2, g_m, g_n, h_l, two_part_l=None):
        """Cache and return convolution results"""
        from scipy.signal import fftconvolve
    
        # Create cache key
        cache_key = ("convolution", 
                    hash(c1.tobytes()), 
                    hash(c2.tobytes()),
                    hash(g_m.tobytes()), 
                    hash(g_n.tobytes()),
                    hash(h_l.tobytes()),
                    hash(two_part_l.tobytes()) if two_part_l is not None else None)
    
        if cache_key in self.c_cache:
            return self.c_cache[cache_key]
    
        # Calculate convolution
        C_l = fftconvolve(c1 * g_m, c2 * g_n)
        #Old comments about C_l
        # C_l=convolve(c_m*self.g_m[i,:],c_m*self.g_n[i,:])
        # multiply all l terms together
        # C_l=C_l*self.h_l[i,:]*self.two_part_l[i]
    
        # Apply additional terms
        if two_part_l is not None:
            C_l = C_l * h_l * two_part_l
        else:
            C_l = C_l * h_l
        
        # Cache and return
        self.c_cache[cache_key] = C_l
        return C_l


    def J_k_scalar(self, P, X, nu, P_window=None, C_window=None):
        from numpy.fft import ifft, irfft

        pf, p, g_m, g_n, two_part_l, h_l = X

        if (self.low_extrap is not None):
            P = self.EK.extrap_P_low(P)

        if (self.high_extrap is not None):
            P = self.EK.extrap_P_high(P)

        P_b = P * self.k_extrap ** (-nu)

        if (self.n_pad != 0):
            P_b = np.pad(P_b, pad_width=(self.n_pad, self.n_pad), mode='constant', constant_values=0)

        c_m = self._cache_fourier_coefficients(P_b, C_window)

        A_out = np.zeros((pf.shape[0], self.k_size))
        for i in range(pf.shape[0]):
            # convolve f_c and g_c
            # C_l=np.convolve(c_m*self.g_m[i,:],c_m*self.g_n[i,:])
            C_l = self._cache_convolution(c_m, c_m, g_m[i,:], g_n[i,:], h_l[i,:], two_part_l[i])

            # set up to feed ifft an array ordered with l=0,1,...,-1,...,N/2-1
            c_plus = C_l[self.l >= 0]
            c_minus = C_l[self.l < 0]

            C_l = np.hstack((c_plus[:-1], c_minus))
            A_k = ifft(C_l) * C_l.size  # multiply by size to get rid of the normalization in ifft

            A_out[i, :] = np.real(A_k[::2]) * pf[i] * self.k_final ** (-p[i] - 2)
        # note that you have to take every other element
        # in A_k, due to the extended array created from the
        # discrete convolution

        P_out = irfft(c_m[self.m >= 0]) * self.k_final ** nu * float(self.N)
        if (self.n_pad != 0):
            # get rid of the elements created from padding
            P_out = P_out[self.id_pad]
            A_out = A_out[:, self.id_pad]

        return P_out, A_out

    
    def J_k_tensor(self, P, X, P_window=None, C_window=None):
        from scipy.signal import fftconvolve
        from numpy.fft import ifft, rfft

        pf, p, nu1, nu2, g_m, g_n, h_l = X

        if (self.low_extrap is not None):
            P = self.EK.extrap_P_low(P)

        if (self.high_extrap is not None):
            P = self.EK.extrap_P_high(P)

        A_out = np.zeros((pf.size, self.k_size))

        P_fin = np.zeros(self.k_size)

        for i in range(pf.size):

            P_b1 = P * self.k_extrap ** (-nu1[i])
            P_b2 = P * self.k_extrap ** (-nu2[i])

            if (P_window is not None):
                # window the input power spectrum, so that at high and low k
                # the signal smoothly tappers to zero. This make the input
                # more "like" a periodic signal

                if (self.verbose):
                    print('windowing biased power spectrum')
                W = p_window(self.k_extrap, P_window[0], P_window[1])
                P_b1 = P_b1 * W
                P_b2 = P_b2 * W

            if (self.n_pad != 0):
                P_b1 = np.pad(P_b1, pad_width=(self.n_pad, self.n_pad), mode='constant', constant_values=0)
                P_b2 = np.pad(P_b2, pad_width=(self.n_pad, self.n_pad), mode='constant', constant_values=0)
            c_m = self._cache_fourier_coefficients(P_b1, C_window)
            c_n = self._cache_fourier_coefficients(P_b2, C_window)

            # convolve f_c and g_c
            C_l = self._cache_convolution(c_m, c_n, g_m[i,:], g_n[i,:], h_l[i,:])

            # set up to feed ifft an array ordered with l=0,1,...,-1,...,N/2-1
            c_plus = C_l[self.l >= 0]
            c_minus = C_l[self.l < 0]

            C_l = np.hstack((c_plus[:-1], c_minus))
            A_k = ifft(C_l) * C_l.size  # multiply by size to get rid of the normalization in ifft

            A_out[i, :] = np.real(A_k[::2]) * pf[i] * self.k_final ** (p[i])
            # note that you have to take every other element
            # in A_k, due to the extended array created from the
            # discrete convolution
            P_fin += A_out[i, :]
        # P_out=irfft(c_m[self.m>=0])*self.k**self.nu*float(self.N)
        if (self.n_pad != 0):
            # get rid of the elements created from padding
            # P_out=P_out[self.id_pad]
            A_out = A_out[:, self.id_pad]
            P_fin = P_fin[self.id_pad]

        return P_fin, A_out






### Example script ###
if __name__ == "__main__":
    # An example script to run FASTPT for (P_22 + P_13) and IA.
    # Makes a plot for P_22 + P_13.
    from time import time

    # Version check
    print(f'This is FAST-PT version {__version__}')

    # load the data file

    d = np.loadtxt('Pk_test.dat')
    # declare k and the power spectrum
    k = d[:, 0]
    P = d[:, 1]

    # set the parameters for the power spectrum window and
    # Fourier coefficient window
    # P_window=np.array([.2,.2])
    C_window = .75
    # document this better in the user manual

    # padding length
    n_pad = int(0.5 * len(k))
    #	to_do=['one_loop_dd','IA_tt']
    to_do = ['one_loop_dd']
    #	to_do=['dd_bias','IA_all']
    # to_do=['all']

    # initialize the FASTPT class
    # including extrapolation to higher and lower k
    t1 = time()
    fpt = FASTPT(k, to_do=to_do, low_extrap=-5, high_extrap=3, n_pad=n_pad)

    t2 = time()
    # calculate 1loop SPT (and time the operation)
    # P_spt=fastpt.one_loop_dd(P,C_window=C_window)
    P_lpt = fpt.one_loop_dd_bias_lpt(P, C_window=C_window)

    # for M = 10**14 M_sun/h
    b1L = 1.02817
    b2L = -0.0426292
    b3L = -2.55751
    b1E = 1 + b1L

    # for M = 10**14 M_sun/h
    # b1_lag = 1.1631
    # b2_lag = 0.1162

    # [Ps, Pnb, Pb1L, Pb1L_2, Pb1L_b2L, Pb2L, Pb2L_2, Pb3L, Pb1L_b3L] = [P_lpt[0],P_lpt[1],P_lpt[2],P_lpt[3],P_lpt[4],P_lpt[5],P_lpt[6],P_lpt[7],P_lpt[8]]
    [Ps, Pnb, Pb1L, Pb1L_2, Pb1L_b2L, Pb2L, Pb2L_2] = [P_lpt[0], P_lpt[1], P_lpt[2], P_lpt[3], P_lpt[4], P_lpt[5],
                                                       P_lpt[6]]

    # Pgg_lpt = (b1E**2)*P + Pnb + (b1L)*(Pb1L) + (b1L**2)*Pb1L_2 + (b1L*b2L)*Pb1L_b2L + (b2L)*(Pb2L) + (b2L**2)*Pb2L_2 + (b3L)*(Pb3L) + (b1L*b3L)*Pb1L_b3L
    Pgg_lpt = (b1E ** 2) * P + Pnb + (b1L) * (Pb1L) + (b1L ** 2) * Pb1L_2 + (b1L * b2L) * Pb1L_b2L + (b2L) * (Pb2L) + (
            b2L ** 2) * Pb2L_2

    # print([pnb,pb1L,pb1L_2,pb2L,Pb1L_b2L])

    t3 = time()
    print('initialization time for', to_do, "%10.3f" % (t2 - t1), 's')
    print('one_loop_dd recurring time', "%10.3f" % (t3 - t2), 's')

    # calculate tidal torque EE and BB P(k)
    # P_IA_tt=fastpt.IA_tt(P,C_window=C_window)
    # P_IA_ta=fastpt.IA_ta(P,C_window=C_window)
    # P_IA_mix=fastpt.IA_mix(P,C_window=C_window)
    # P_RSD=fastpt.RSD_components(P,1.0,C_window=C_window)
    # P_kPol=fastpt.kPol(P,C_window=C_window)
    # P_OV=fastpt.OV(P,C_window=C_window)
    # P_IRres=fastpt.IRres(P,C_window=C_window)
    # make a plot of 1loop SPT results
    import matplotlib.pyplot as plt

    ax = plt.subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r'$P(k)$', size=30)
    ax.set_xlabel(r'$k$', size=30)

    ax.plot(k, P, label='linear')
    # ax.plot(k,P_spt[0], label=r'$P_{22}(k) + P_{13}(k)$' )
    ax.plot(k, Pgg_lpt, label='P_lpt')

    plt.legend(loc=3)
    plt.grid()
    plt.show()
