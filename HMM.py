# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:32:04 2016

@author: ilker bayram
"""

import numpy

def GenerateChain( A, B, p, T ):
    # generates an HMM chain given the model parameters
    #
    # input variables :
    # A : State transition probability matrix (N x N)
    # B : Observation probability distribution (N x M)
    # p : initial probability distribution (N x 1)
    # T : length of the chain
    #
    # output variable : 
    # O : sequence of length T -- it is assumed that the alphabet is {0,...,M-1}
    # S : hidden state sequence
    
    N = B.shape[0] # number of states
    M = B.shape[1] # number of distinct elements in the alphabet
    
    dist = p # set the initial distribution
    O = numpy.zeros( (T,) , dtype = int ) # will hold the observation sequence
    S = numpy.zeros((T,), dtype = int ) # will hold the state sequence
    for t in range(0,T):
        S[t] = numpy.random.choice(numpy.arange(0, N), p = dist) # select the state        
        O[t] = numpy.random.choice(numpy.arange(0, M), p = B[S[t],:]) # select the state
        dist = A[S[t],:]
    
    return O, S
        
    
    
def ForwardMap( O, A, B, p ):
    # the alpha map used in the Forward-Backward Procedure related to the HMM
    # model desribed by A, B, p
    #
    # input variables : 
    # O : observation sequence
    # A : State transition probability matrix (N x N)
    # B : Observation probability distribution (N x M)
    # p : initial probability distribution (N x 1)
    #
    # output variables : 
    # alp : the forward variables
    # c : normalizing constants that multiply alp so that alp sums to unity at each instant
    
    N = B.shape[0] # number of states
    T = O.shape[0] # number of time instances
    
    alp = numpy.zeros((N,T))
    c = numpy.ones((T))
    
    # initialization
    alp[:,0] = B[:,O[0]] * p
    c[0] = 1 / numpy.sum(alp[:,0])
    alp[:,0] = alp[:,0] * c[0]

    # induction starts
    for t in range(1,T):        
        alp[:,t] = numpy.dot( A.transpose(), alp[:,t-1] ) * B[:,O[t]]
        c[t] = 1 / numpy.sum( alp[:,t] )
        alp[:,t] = c[t] * alp[:,t]
    
    return alp, c
    
def BackwardMap( O, A, B ):
    # the beta map used in the Forward-Backward Procedure related to the HMM
    # model desribed by A, B, p
    #
    # input variables : 
    # O : observation sequence
    # A : State transition probability matrix (N x N)
    # B : Observation probability distribution (N x M)    
    #
    # output variables : 
    # bet : the backward variables
    # d : normalizing constants that multiply alp so that alp sums to unity at each instant
    
    N = B.shape[0] # number of states
    T = O.shape[0] # number of time instances
    
    bet = numpy.ones((N,T))
    d = numpy.ones((T))
        
    # induction starts
    for t in range(T-2,-1,-1):        
        bet[:,t] = numpy.dot( A, B[:,O[t+1]] * bet[:,t+1] )
        d[t] = 1 / numpy.sum( bet[:,t] )
        bet[:,t] = d[t] * bet[:,t]
    
    return bet, d

def EstimateStates(O,A,B,p):
    # ML state estimation for the HMM model desribed by A, B, p
    #
    # input variables
    # O : observation sequence
    # A : State transition probability matrix (N x N)
    # B : Observation probability distribution (N x M)
    # p : initial probability distribution (N x 1)
    #
    # output variable :
    # q : estimated states
    
    N = B.shape[0] # number of states
    T = O.shape[0] # number of time instances
    
    delta = numpy.zeros((N,T)) # delta map
    psi = numpy.zeros((N,T),dtype = int) # psi map
    
    # initialization
    delta[:,0] = B[:, O[0]] * p

    # recursion
    for t in range(1,T):
        u = A * delta[:,t-1].reshape((N,1))
        for jy in range(0,N):
            delta[jy, t] = numpy.max( u[:,jy] * B[jy, O[t]] )
            psi[jy, t] = numpy.argmax( u[:,jy] )
        
        delta[:,t] /= numpy.sum( delta[:,t] )
    
    # termination
    q = numpy.zeros(T,dtype = int)
    q[T-1] = numpy.argmax(delta[:,T-1])
    for t in range(T-2,-1,-1):
        q[t] = psi[ q[t+1], t+1]

    return q

def EstimateModel( O, N, M, MAX_ITER ):
    # ML model estimation given a sequence of observations, using the Baum-Welch (EM) algorithm
    #
    # input variables : 
    # O : observation sequence
    # N : number of states
    # M : number of distinct elements in the alphabet
    # MAX_ITER : maximum number of iterations for the Baum-Welch algorithm
    #
    # output variables :
    # A : State transition probability matrix (N x N)
    # B : Observation probability distribution (N x M)
    # p : initial probability distribution (N x 1)
    
    T = O.shape[0] # number of time instances
    
    # random initialization
    A = 0.1 * numpy.ones((N,N)) / float(N) + 0.9 * numpy.identity(N)
    
    B = numpy.random.uniform(0,1,(N,M))
    b = numpy.sum(B, axis=1)
    b = numpy.reshape(b,(N,1))
    B = B / b
    
    # S0 : states start from 0
    # if S0 is true, use this initialization
    p = numpy.zeros(N)
    p[0] = 1.0
    # if S0 is false, initialize p randomly    
    #p = numpy.random.uniform(0,1,N)
    #p = p / numpy.sum(p)
    
    
    ksi = numpy.zeros((N,N,T))
    gam = numpy.zeros((N,T))
    
    for iter in range( 0, MAX_ITER ):
        #print iter
        print '\r HMM Model Estimation : [{0}{1}] {2}% Complete'.format('|'*(10 * (iter+1) / MAX_ITER), ' '*(10 - 10 * (iter+1) / MAX_ITER), 100 * (iter+1) / MAX_ITER),
        alp, c = ForwardMap( O, A, B, p )
        bet, d = BackwardMap( O, A, B )
        
        #compute ksi and gamma
        for t in range( 0, T-1 ):
            den = 0.0 # denominator...
            for i in range( 0, N ):
                for jy in range( 0, N ):
                    ksi[ i, jy, t ] = alp[i, t] * A[i, jy] * B[jy, O[t+1] ] * bet[jy, t+1]
                    den +=  ksi[ i, jy, t ]            
            # normalize
            ksi[:,:,t] /= den
            # compute gamma
            kt = ksi[:,:,t]
            gam[:,t] = numpy.sum(kt, axis=1)        
        # update the model parameters
        p = gam[:,0]
        A = numpy.sum(ksi, axis=2)
        den = numpy.sum(gam, axis=1)
        den = numpy.reshape(den,(N,1))
        A /= den
        den = numpy.reshape(den, N)  
        for m in range(0, M):
            mask = numpy.double( O == m )
            B[:,m] = numpy.sum( mask * gam , axis=1)
            B[:,m] /= den
        
    
    return A, B, p