# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 16:28:50 2016

@author: ilker bayram
"""
#import pdb
import numpy
import HMM

A = numpy.array([[0.95,0.05],[0.1,0.9]]) # the state transition probability matrix

B = numpy.array([[3/float(6), 2/float(6), 1/float(6)],[0.1,0.1,0.8]]) # the output probability distribution for each state

p = numpy.array([1,0]) # start from the first state

T = 5000 # length of the chain

O, S = HMM.GenerateChain(A,B,p,T) # observed chain O and the states S producing the observations 

# parameters for HMM model estimation
N = B.shape[0] # number of states
M = B.shape[1] # size of the alphabet
MAX_ITER = 100 # number of iterations to use for model estimation

# estimate the model using the Baum-Welch algorithm
A2, B2, p2 = HMM.EstimateModel( O, N, M, MAX_ITER )

# compute estimation error 
# absolute sum of the difference, normalized by N for A, sqrt(NM) for B
erA = numpy.sum( abs( A.flatten() - A2.flatten() ) ) / float(N)
erB = numpy.sum( abs( B.flatten() - B2.flatten() ) ) / numpy.sqrt( float(N*M) )
erP = numpy.sum( abs( p.flatten() - p2.flatten() ) )
print '\n Normalized Error for A : {0} \n Normalized Error for B : {1}'.format(erA, erB)

# estimate the states based on the model estimate
q = HMM.EstimateStates(O, A2, B2, p2)

cor = ( 100 * numpy.sum( numpy.double(q == S) ) ) / T
print '\n {0}% correct state estimation'.format(cor)