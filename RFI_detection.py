
import numpy as np
import os,sys

import time

import matplotlib.pyplot as plt

from RFI_support import *

from numba import jit,prange






#---------------------------------------------------------
# 1 . Functions for performing SK
#---------------------------------------------------------

#Compute SK on a 2D array of power values

@jit(parallel=True)
def SK_EST(a,m,n=1,d=1):
	"""
	Compute SK on a 2D array of power values.

	Parameters
	-----------
	a : ndarray
		2-dimensional array of power values. Shape (Num Channels , Num Raw Spectra)
	n : int
		integer value of N in the SK function. Inside accumulations of spectra.
	m : int
		integer value of M in the SK function. Outside accumulations of spectra.
	d : float
		shape parameter d in the SK function. Usually 1 but can be empirically determined.
	
	Returns
	-----------
	out : ndarray
		Spectrum of SK values.
	"""

	#make s1 and s2 as defined by whiteboard (by 2010b Nita paper)
	a = a[:,:m]*n
	sum1=np.sum(a,axis=1)
	sum2=np.sum(a**2,axis=1)
	sk_est = ((m*n*d+1)/(m-1))*(((m*sum2)/(sum1**2))-1)                     
	return sk_est


@jit(parallel=True)
def SK_EST_alt(s1,s2,m,n=1,d=1):
	"""
	Compute SK on a 2D array of power values, using s1 and s2 given instead of data

	Parameters
	-----------

	n : int
		integer value of N in the SK function. Inside accumulations of spectra.
	m : int
		integer value of M in the SK function. Outside accumulations of spectra.
	d : float
		shape parameter d in the SK function. Usually 1 but can be empirically determined.
	
	Returns
	-----------
	out : ndarray
		Spectrum of SK values.
	"""
	sk_est = ((m*n*d+1)/(m-1))*(((m*s2)/(s1**2))-1)                     
	return sk_est


#multiscale variant
#only takes n=1 for now
#takes sum1 and sum2 as arguments rather than computing inside
@jit(parallel=True)
def ms_SK_EST(s1,s2,m,n=1,d=1):
	"""
	Multi-scale Variant of SK_EST.

	Parameters
	-----------
	s1 : ndarray
		2-dimensional array of power values. Shape (Num Channels , Num Raw Spectra)

	s2 : ndarray
		2-dimensional array of squared power values. Shape (Num Channels , Num Raw Spectra)

	m : int
		integer value of M in the SK function. Outside accumulations of spectra.

	ms0 : int
		axis 0 multiscale
	
	ms1 : int
		axis 1 multiscale
	
	Returns
	-----------
	out : ndarray
		Spectrum of SK values.
	"""
	
                 #((m*n*d+1)/(m-1))*(((m*sum2)/(sum1**2))-1)
	sk_est = ((m*n*d+1)/(m-1))*(((m*s2)/(s1**2))-1)
	#print(sk_est)
	return sk_est


def master_SK_EST(data,SK_M,m,n,n=1,d=1):
	

	num_skbins = data.shape[1]//SK_M


	data = np.abs(np.reshape(data,(data.shape[0],-1,SK_M,data.shape[2]))**2

	s1 = np.sum(data,axis=2)
	s2 = np.sum(data**2,axis=2)

	spect_block = np.mean(data,axis=1)


	































