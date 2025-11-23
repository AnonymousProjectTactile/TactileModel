

import sys
sys.path.append('Module3/')
 
import os 
import numpy as np 
import open3d as o3d
import numbers
from warnings import warn
import torch as th
import math

import matplotlib.pyplot as plt
from scipy.spatial import Delaunay 
from scipy.spatial import cKDTree

# from utils_pc import *


""" CPD Registration """


def gaussian_kernel(X, beta, Y=None):
    if Y is None:
        Y = X
    diff = th.sub(X[:, None, :], Y[None, :,  :]) # (M,1,D)-(1,M,D) -> (M,M,D)
    diff = th.square(diff)
    diff = th.sum(diff, dim=2)
    return th.exp(-diff / (2 * beta**2))


def low_rank_eigen(G, num_eig):
    S, Q = th.linalg.eigh(G)
    eig_indices = th.flip(th.argsort(th.abs(S)), dims=(0, ))[:num_eig].tolist()
    Q = Q[:, eig_indices]  # eigenvectors
    S = S[eig_indices]  # eigenvalues.
    return Q, S


class EM(object):
    
    def __init__(self, X, Y, 
                 alpha, beta, 
                 num_eig=100, low_rank = False, sigma2=None, 
                 max_iterations=None, tolerance=None, w=None ):
        
        
        self.device = th.device('cuda:0')
        self.X = th.tensor(X, dtype=th.float64).float().to(self.device) # 
        self.Y = th.tensor(Y, dtype=th.float64).float().to(self.device)
        self.TY = th.tensor(Y, dtype=th.float64).float().to(self.device)
        self.sigma2 = self.initialize_sigma2() if sigma2 is None else sigma2
        if type(self.sigma2) is not th.Tensor:
            self.sigma2 = th.tensor(self.sigma2, dtype=th.float64).float().to(self.device)
        (self.N, self.D) = self.X.shape
        (self.M, _) = self.Y.shape
        self.tolerance = th.tensor(0.001, dtype=th.float64).float().to(self.device) if tolerance is None else tolerance
        if type(self.tolerance) is not th.Tensor:
            self.tolerance = th.tensor(self.tolerance, dtype=th.float64).float().to(self.device)
        self.w = th.tensor(0.0, dtype=th.float64).float().to(self.device) if w is None else w
        if type(self.w) is not th.Tensor:
            self.w = th.tensor(self.w, dtype=th.float64).float().to(self.device)
        self.max_iterations = 100 if max_iterations is None else max_iterations
        self.iteration = 0
        self.diff = th.tensor(np.inf, dtype=th.float64).float().to(self.device)
        self.q = th.tensor(np.inf, dtype=th.float64).float().to(self.device)
        self.P = th.zeros((self.M, self.N), dtype=th.float64).to(self.device)
        self.Pt1 = th.zeros((self.N, 1), dtype=th.float64).to(self.device)
        self.P1 = th.zeros((self.M, 1), dtype=th.float64).to(self.device)
        self.PX = th.zeros((self.M, self.D), dtype=th.float64).to(self.device)
        self.Np = th.tensor(0., dtype=th.float64).float().to(self.device)
        self.alpha = alpha # 2 if alpha is None else alpha
        self.beta = beta # 4 if beta is None else beta
        self.alpha = th.tensor(self.alpha, dtype=th.float64).float().to(self.device)
        self.beta = th.tensor(self.beta, dtype=th.float64).float().to(self.device)
        self.W = th.zeros((self.M, self.D), dtype=th.float64).float().to(self.device) # (M,D)
        self.G = gaussian_kernel(self.Y, self.beta).to(self.device) # (M,M)
        self.num_eig = th.tensor(num_eig, dtype=th.int64).to(self.device)
        self.low_rank = low_rank
        if self.low_rank is True:
            self.Q, self.S = low_rank_eigen(self.G, self.num_eig)
            self.inv_S = th.diag(th.div(1., self.S))
            self.S = th.diag(self.S)
            self.E = th.tensor(0., dtype=th.float64).float().to(self.device)


    
    
    def initialize_sigma2(self):
        (N, D) = self.X.shape
        (M, _) = self.Y.shape
        diff = th.sub(self.X[None, :, :], self.Y[:, None, :]) # (1,N,D)-(M,1,D) -> (M,N,D) 
        err = th.pow(diff, 2) # (M,N,D) 
        return th.sum(err) / (D * M * N)
    
    
    
    def register(self, callback=lambda **kwargs: None):
        self.transform_point_cloud()
        while self.iteration < self.max_iterations and self.diff > self.tolerance:
            self.iterate()
            if callable(callback):
                kwargs = {'iteration': self.iteration, 'error': self.q.detach().cpu().numpy(), 'X': self.X, 'Y': self.TY}
                callback(**kwargs)
        return self.TY, self.get_registration_parameters()
    
    
    
    def iterate(self):
        """ one time iteration """
        self.expectation()
        self.maximization()
        self.iteration += 1



    def expectation(self):
        """ E-Step:  """
        P = th.sum(th.pow(self.X[None, :, :] - self.TY[:, None, :], 2), dim=2) # (M, N)
        P = th.exp(th.div(-P, (2.*self.sigma2))) # 
        c = th.pow(2.*th.tensor(math.pi, dtype=th.float64)*self.sigma2, (self.D/2.))*self.w/(1. - self.w)*self.M/self.N

        den = th.sum(P, dim = 0, keepdims = True) # (1, N)
        den = th.clamp(den, th.finfo(self.X.dtype).eps, None) + c

        self.P = th.div(P, den) # (M,N)
        self.Pt1 = th.sum(self.P, dim=0).reshape(-1, 1)
        self.P1 = th.sum(self.P, dim=1).reshape(-1, 1)
        self.Np = th.sum(self.P1)
        self.PX = th.mm(self.P, self.X)
        
        
        
        
        
    def maximization(self):
        """ M-Step """
        self.update_transform()     
        self.transform_point_cloud() 
        self.update_variance()          
        
    
    def update_transform(self):
        """ """
        if self.low_rank is False: #
            A = th.mm(th.diag(self.P1.reshape(-1, )), self.G) + self.alpha * self.sigma2 * th.eye(self.M, dtype=th.float64).float().to(self.device)
            B = self.PX - th.mm(th.diag(self.P1.reshape(-1, )), self.Y)
            self.W = th.linalg.solve(A, B)# 
        else:
            dP = th.diag(self.P1.reshape(-1, ))
            dPQ = th.mm(dP, self.Q)
            F = th.sub(self.PX, th.mm(dP, self.Y))

            self.W = 1. / (self.alpha * self.sigma2) * (F - th.mm(dPQ, (th.linalg.solve((self.alpha * self.sigma2 * self.inv_S + th.mm(self.Q.T, dPQ)),(th.mm(self.Q.T, F))))))
            QtW = th.mm(self.Q.T, self.W)
            self.E = self.E + self.alpha / 2. * th.trace(th.mm(QtW.T, th.mm(self.S, QtW)))


    def transform_point_cloud(self, Y=None):
        """ transformation """
        if Y is not None:
            G = gaussian_kernel(X=Y, beta=self.beta, Y=self.Y)
            return Y + th.mm(G, self.W)
        else:
            if self.low_rank is False:
                self.TY = self.Y + th.mm(self.G, self.W) # Y+GW 
            else:
                self.TY = self.Y + th.mm(self.Q, th.mm(self.S, th.mm(self.Q.T, self.W)))
            return


    def update_variance(self):
        qprev = self.sigma2
        self.q = th.tensor(np.inf, dtype=th.float64).float().to(self.device)
        xPx = th.mm(self.Pt1.permute(1, 0), th.sum(th.mul(self.X, self.X), dim=1).reshape(-1, 1))
        yPy = th.mm(self.P1.permute(1, 0), th.sum(th.mul(self.TY, self.TY), dim=1).reshape(-1, 1))
        trPXY = th.sum(th.mul(self.TY, self.PX))
        self.sigma2 = th.div((xPx - 2. * trPXY + yPy), (self.Np * self.D))
        if self.sigma2 <= 0.:
            self.sigma2 = (self.tolerance / 10.).clone()
        self.diff = th.abs(self.sigma2 - qprev)


    def get_registration_parameters(self):
        return self.G, self.W




