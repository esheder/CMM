#!/usr/bin/env python3
"""Solves the two-region boundary source problem.

"""

import numpy as np
from scipy.optimize import newton_krylov
from scipy.linalg import sqrtm
from scipy.integrate import quad
from util import *
from numpy import exp, sinh, sqrt

class Region:
    """Data holding object for each region

    """

    def _check_fit(T,S):
        """Check that T,S match sizes

        """
        n = T.shape[0]
        if S.shape != (n,n) or len(T.shape) != 1:
            raise ValueError("Matrix shapes don't align")
        

    def __init__(self, T, S, f, J):
        Region._check_fit(T,S)
        self.sigA = -S
        n = S.shape[0]
        for i in range(n):
            self.sigA[i,i] += T[i]
        self.flx = f
        self.rJ = J
        self.arate = (T - np.sum(S, axis=1)).dot(f)
        #self.__test__()

    def __test__(self):
        print(self.sigA)
        
def get_tallied(flx, D):
    """Extracts tallied value rather than diffusion coefficient

    """
    return 3*np.multiply(flx,D)

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

def source_strength(regs, leakage):
    """Calculate the total source strength used in the Monte Carlo for normalization purposes.

    All neutrons absorbed must come from somewhere. If we know the absorption rate and the leakage
    percentage we can determine how many neutrons were originally sent out.

    regs = Region data from MC tally
    leakage = Leakage percentage. Because of how the OpenMC setup happened, half of all source
              neutrons immediately die because they are actually shot into the left void.

    """

    arate = sum(reg.arate for reg in regs)
    emission = arate / (1.0 - leakage)
    return emission / 2.0 #Because of the oddity that the actual emission includes 50% of DoAs

def source_boundary(enerb, st=1.0):
    """Calcuate the boundary source with the usual watt spectrum.

    enerb = Energy group boundaries
    st = source total strength

    """

    def watt(E,a=988000.0,b=2.249e-6):
        """Watt spectrum.

        P(E)dE = exp(-E/a)*sinh(sqrt(b*E))dE

        E = Energy in eV
        a = exponential scaling parameter in eV
        b = sinh scaling parameter in 1/eV

        """

        return exp(-E/a)*sinh(sqrt(b*E))

    nrm = quad(lambda x: watt(x), enerb[0], enerb[-1])[0]
    return np.array([(st/nrm)*quad(lambda x: watt(x), le, ue)[0]
                     for le, ue in zip(enerb[:-1], enerb[1:])])

def solve(reg, d, Jp):
    """Solve for the diffusion coefficient using non-linear solvers

    reg = Two region objects in a list
    d = Length between regions in cm
    Jp = Left boundary condition current

    """
    
    phi0m, phi0p = reg[0].flx, reg[1].flx
    phi1m, phi1p = reg[0].rJ, reg[1].rJ
    n = phi0m.shape[0]

    @PrintRes
    def residual(D):
        """Returns how close we are to convergence in each variable

        D = Two dimensional diffusion coefficient matrix. 
            First column is left region, second is right region.

        """
        n = D.shape[0]
        Dm, Dp = np.diag(D[:,0]), np.diag(D[:,1])
        Up = np.real(sqrtm(diag_inv(Dp).dot(reg[1].sigA)))
        Um = np.real(sqrtm(diag_inv(Dm).dot(reg[0].sigA)))
        Ap, exU = BCsolve(Up,Um,Dp,Dm,Jp,d)
        rm = Dm.dot(phi0m) - Dm.dot(d*exU.dot(Ap)) - phi1m
        rp = Dp.dot(phi0p) + Dp.dot(d*exU.dot(Ap)) - phi1p
        return np.vstack((rm,rp)).T

    #Initial guess is phi1 / (3.0 * phi0)
    guess = np.vstack((np.divide(phi1m, phi0m) / 3.0, np.divide(phi1p, phi0p) / 3.0)).T
    print('starting nonlinear solver')
    sol = newton_krylov(residual, guess, method='lgmres')
    #print(residual(sol))
    return sol
    

if __name__ == '__main__':
    from argparse import ArgumentParser as AP

    parser = AP(description="Solves the very specific problem")
    parser.add_argument('-i', nargs=2, default=['input/leftreg.dat.csv', 'input/rightreg.dat.csv'],
                        help="CMM tally results")
    parser.add_argument('-e', default='input/EnergyGroup.dat', help="Energy group structure")
    parser.add_argument('-l', default=0.738, type=float,
                        help="Percentage of neutrons that leaked in OpenMC")
    parser.add_argument('-o', help="Output file")

    args = parser.parse_args()

    regions = []
    for f in args.i:
        inp = read_inp(f)
        sigT, sigS, flx, D = inp
        rJ = get_tallied(flx, D)
        regions.append(Region(sigT, sigS, flx, rJ))
    with open(args.e, 'r') as f:
        lines = f.readlines()[1:] #Skip header
    eb = np.array([float(v.strip()) for v in lines[::-1]]) #Reverse order because E_0 is top energy
    st = source_strength(regions, args.l)
    #print(st)
    Jp = source_boundary(eb, st)
    #Jp = np.zeros_like(Jp)
    #Jp[0] = 1.0
    d = 20.0
    D = solve(regions, d, Jp)
    print(D)
