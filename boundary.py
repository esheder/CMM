#!/usr/bin/env python3
"""Solves the two-region boundary source problem.

"""

import numpy as np
from scipy.optimize import newton_krylov
from scipy.linalg import sqrtm, expm, solve
from util import *

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
    sigT = np.zeros((n,))
    D = np.zeros((n,))
    flx = np.zeros((n,))
    for i, line in enumerate(lines):
        flx[i], sigT[i], crap, D[i] = line.split()[:4]
        sigS[i,:] = line.split()[4:]
    return sigT, sigS, flx, D

def BCsolve(Up, Um, Dp, Dm, Jp, d):
    """Solves the boundary interface problems

    Up = sqrt(sigA+/D), in 1/cm, where the + sign means it belongs to the right region
    Um = Same as Up but for the left region
    Dp/m = Diffusion coefficient in right/left region in cm
    Jp = Left boundary condition
    d = Left region width in cm

    """
    
    exUp = expm(-d*Up)
    exUmp = expm(d*Um)
    exUmm = expm(-d*Um)
    n = exUp.shape[0]
    b = np.zeros((3*n,))
    for i,j in enumerate(range(2*n,3*n)):
        b[j] = Jp[i]

    M = BCCLS(exUp, exUmp, exUmm, Up, Um, Dp, Dm, d)
    v = linsolve(M,b)
    Ap = v[2*n:]
    return Ap,exUp

def solve(reg, d, Jp):
    """Solve for the diffusion coefficient using non-linear solvers

    reg = Two region objects in a list
    d = Length between regions in cm
    Jp = Left boundary condition current

    """
    
    phi0m, phi0p = reg[0].flx, reg[1].flx
    phi1m, phi1p = reg[0].rJ, reg[1].rJ

    def residual(D):
        """Returns how close we are to convergence in each variable

        """
        n = D.shape[0]
        Dm, Dp = np.diag(D[:,0]), np.diag(D[:,1])
        Up = np.real(sqrtm(diag_inv(Dp).dot(reg[1].sigA)))
        Um = np.real(sqrtm(diag_inv(Dm).dot(reg[0].sigA)))
        Ap, exU = BCsolve(Up,Um,Dp,Dm,Jp,d)
        rm = Dm.dot(phi0m) - Dm.dot(d*exU.dot(Ap)) - phi1m
        rp = Dp.dot(phi0p) + Dp.dot(d*exU.dot(Ap)) - phi1p
        return np.vstack((rm,rp)).T
    
    guess = np.ones((phi0m.shape[0],2))
    return newton_krylov(residual, guess, method='lgmres')
    

if __name__ == '__main__':
    from argparse import ArgumentParser as AP

    parser = AP(description="Solves the very specific problem")
    parser.add_argument('-i', nargs = 2, default=['leftreg.dat.csv', 'rightreg.dat.csv'], help="CMM tally results")
    parser.add_argument('-o', help="Output file")

    args = parser.parse_args()

    regions = []
    for f in args.i:
        inp = read_inp(f)
        sigT, sigS, flx, D = inp
        rJ = get_tallied(flx, D)
        regions.append(Region(sigT, sigS, flx, rJ))
    d = 40.0
    Jp = np.zeros_like(flx)
    Jp[0] = 1.0
    D = solve(regions, d, Jp)
    print(D)
