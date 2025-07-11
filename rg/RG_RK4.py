'''
	Code to calcualte Renormalization group resutls using
	RK4 integrator.
	This simple integration will probably not work for k_max greater than 1.
	Please see the paper for more details.

	J. E. McEwen (c) 2016
	mcewen.24@osu.edu
'''

import numpy as np
import matplotlib.pyplot as plt
from fastpt.utils.fastpt_extr import p_window
from fastpt.core.FASTPT import FASTPT
import time, sys


def RG_RK4(name,k,P,d_lambda,max,n_pad,P_window,C_window):

	x=max/d_lambda-int(max/d_lambda)
	if ( x !=0.0 ):
		raise ValueError('You need to send a d_lambda step so that max/d_lambda has no remainder to reach Lambda=max')

	#save name
	name='RG_RK4_'+name
	# The spacing in log(k)
	Delta=np.log(k[1])-np.log(k[0])

	# This window function tapers the edge of the power spectrum.
	# It is applied to each Runge-Kutta step.
	# You could change it here.
	W=p_window(k,P_window[0],P_window[1])
	#W=1
	t1=time.time()

	# windowed initial power spectrum
	P_0=P*W

	nu=-2
	fastpt=FASTPT.FASTPT(k,nu=nu,n_pad=n_pad)
	P_spt=fastpt.one_loop(P_0,C_window=C_window)
	P_spt=P_0+P_spt
	# initial lambda
	Lambda=0

	d_out=np.zeros((3,k.size+1))
	d_out[0,:]=np.append(Lambda,k)
	d_out[2,:]=np.append(Lambda,P_0)
	d_out[1,:]=np.append(Lambda,P_spt)


	i=0
	while (Lambda <= max):
	#for i in range(N_steps):

		k1=fastpt.one_loop(P,C_window=C_window)
		k1=k1*W

		k2=fastpt.one_loop(k1*d_lambda/2.+ P,C_window=C_window)
		k2=k2*W

		k3=fastpt.one_loop(k2*d_lambda/2. +P,C_window=C_window)
		k3=k3*W

		k4=fastpt.one_loop(k3*d_lambda +P,C_window=C_window)
		k4=k4*W

		# full step
		P=P+1/6.*(k1+2*k2+2*k3+k4)*d_lambda

		# check for failure.
		if (np.any(np.isnan(P))):
			print('RG flow has failed. It could be that you have not chosen a step size well.')
			print('You may want to consider a smaller step size.')
			sys.exit()

		if (np.any(np.isinf(P))):
			print('RG flow has failed. It could be that you have not chosen a step size well.')
			print('You may want to consider a smaller step size.')
			sys.exit()

		#update lambda and the iteration
		i=i+1
		Lambda+=d_lambda
		#print 'lambda', Lambda

		# update data for saving
		d_update=np.append(Lambda,P)
		d_out=np.row_stack((d_out,d_update))
		# set to True to check at each step
		if (False):

			ax=plt.subplot(111)
			ax.set_xscale('log')
			ax.set_yscale('log')
			ax.set_xlabel('k')

			ax.plot(k,P)
			ax.plot(k,P_0, color='red')

			plt.grid()
			plt.show()

	# save the data
	t2=time.time()
	print('time to run seconds', t2-t1)
	print('time to run minutes', (t2-t1)/60.)
	print('number of iterations and output shape', i, d_out.shape)
	np.save(name,d_out)
	# return last step
	return P

if __name__ == "__main__":

	V=sys.version_info[0]
	if (V < 3):
		from ConfigParser import SafeConfigParser
	if (V >=3 ):
		from configparser import SafeConfigParser
	parser = SafeConfigParser()

	name='kmax1_example.ini'

	parser.read(name)

	k_max=parser.getfloat('floats', 'k_max')
	k_min=parser.getfloat('floats', 'k_min')
	step=parser.getfloat('floats', 'step')
	max=parser.getfloat('floats', 'max')
	P_right=parser.getfloat('floats', 'P_w_right')
	P_left=parser.getfloat('floats', 'P_w_left')
	C_window=parser.getfloat('floats', 'C_window')
	n_pad=parser.getint('integers', 'n_pad')
	down_sample=parser.getint('integers', 'down_sample')
	read_name=parser.get('files', 'in_file')
	name=parser.get('files', 'out_file')


	d=np.loadtxt(read_name)	# load data
	k=d[:,0]
	P=d[:,1]

	id=np.where( (k >= k_min) & (k <= k_max) )[0]
	k=k[id]
	P=P[id]

	k=k[::down_sample]
	P=P[::down_sample]

	# if your array is not even in size, FAST-PT will not work-
	# trim if so.
	if (k.size % 2 != 0):
		k=k[:-1]
		P=P[:-1]

	print('Details of run.')
	print('save name :', name)
	print('k min and max:', k_min, k_max)
	print('d log k', np.log(k[1])-np.log(k[0]) )
	print('step size : ', step)
	print('grid size : ', k.size)
	print('down sample factor:', down_sample)
	print(' d lambda', step)

	P_window=np.array([P_left,P_right])


	P_rg=RG_RK4(name,k,P,step,max,n_pad,P_window,C_window)

	ax=plt.subplot(111)
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_xlabel('k')

	ax.set_ylabel(r'$P(k)$', size=30)
	ax.set_xlabel(r'$k$', size=30)

	ax.plot(k,P, label='linear power')
	ax.plot(k,P_rg, label='RG' )

	plt.legend(loc=3)

	plt.grid()
	plt.show()
