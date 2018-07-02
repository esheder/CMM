#!/usr/bin/env python3
"""Solves the two-region boundary source problem.

"""

import numpy as np
from scipy.optimize import newton_krylov
from scipy.linalg import sqrtm, expm

def get_tallied(flx, D):
    """Extracts tallied value rather than diffusion coefficient

    """
    return 6*np.multiply(flx,D)

def read_inp(f):
    """Reads data files

    """
    with open(f, 'r') as of:
        lines = of.readlines()
    n = len(lines)
    sigS = np.zeros((n,n))
    sigT = np.zeros((n,n))
    D = np.zeros((n,1))
    flx = np.zeros((n,1))
    for i, line in enumerate(lines):
        flx[i], sigT[i,i], D[i] = line.split()[:3]
        sigS[i,:] = line.split()[3:]
    return sigT, sigS, flx, D

def BCsolve(Up, Um, Dp, Dm, Jp, d):
    """Solves the boundary interface problems

    """
    exUp = expm(-d*Up)
    exUmp = expm(d*Um)
    exUmm = expm(-d*Um)
    n = exUp.shape[0]
    b = np.zeros((3*n,1))
    for i,j in enumerate(range(2*n,3*n)):
        b[j] = Jp[i]
    A = np.zeros((3*n,3*n))
    for i,j in enumerate(range(0,n)):
        for k,l in enumerate(range(0,n)):
            A[j,l] = exUmm[i,k]
        for k,l in enumerate(range(0,n)):
            A[j,l] = exUmp[i,k]
        for k,l in enumerate(range(0,n)):
            A[j,l] = -exUp[i,k]
            

def solve(sigT, sigS, phi0, phi1, d, Jp):
    """Solve for the diffusion coefficient using non-linear solvers

    """

    def residual(D):
        """Returns how close we are to convergence in each variable

        """
        n = D.shape[0]
        Dm = np.zeros_like(sigT)
        Dp = np.zeros_like(sigT)
        for i in range(n):
            Dm[i,i], Dp[i,i] = D[i,0], D[i,1]
        phi0m, phi0p = phi0
        phi1m, phi1p = phi1
        Up = sqrtm(inv(Dp)*(sigT-sigS))
        Um = sqrtm(inv(Dm)*(sigT-sigS))
        Ap, exU = BCsolve(Up,Um,Dp,Dm,Jp,d)
        rm = Dm.dot(phi0m) - Dm.dot(d*exU.dot(Ap)) - phi1m
        rp = Dp.dot(phi0p) + Dp.dot(d*exU.dot(Ap)) - phi1p
        return np.concatenate(rm,rp)
    
    guess = np.ones((Jp.shape[0],2))
    newton_krylov(residual, guess, method='lgmres')

if __name__ == '__main__':
    from argparse import ArgumentParser as AP

    parser = AP(description="Solves the very specific problem")
    parser.add_argument('-i', default='data.dat', help="CMM tally results")
    parser.add_argument('-o', help="Output file")

    args = parser.parse_args()

    inp = read_inp(args.i)
    sigT, sigS, flx, D = inp
    rJ = get_tallied(flx, D)
    d = 1.0
    Jp = np.zeros(flx.size)
    Jp[0] = 1.0
    D = solve(sigT, sigS, flx, rJ, d, Jp)
