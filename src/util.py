"""Mathematical utility for solving the nonlinear problem

"""


import numpy as np
from scipy.sparse.linalg import aslinearoperator
from scipy.sparse.linalg import bicgstab as solve
from scipy.sparse import csr_matrix
from scipy.linalg import expm

def PrintRes(f):
    def f_wrap(*args, **kwargs):
        res = f(*args, **kwargs)
        print(res)
        return res
    return f_wrap

class BCCLS:
    """Boundary conditions linear operator like object

    """
    def __init__(self, exUp, exUmp, exUmm, Up, Um, Dp, Dm, d):
        self.exUp = exUp
        self.exUmp = exUmp
        self.exUmm = exUmm
        self.Up = Up
        self.Um = Um
        self.Dp = csr_matrix(Dp)
        self.Dm = csr_matrix(Dm)
        self.d = d
        self.n = Dm.shape[0]
        self.shape = (3*self.n,3*self.n)
        self.dtype = 'float64'
        #self._test_()

    def _test_(self):
        print('Up')
        print(self.Up.shape)
        print(self.Up)
        print('Um')
        print(self.Um.shape)
        print(self.Um)
        print('Dp')
        print(self.Dp.shape)
        print(self.Dp)
        print('Dm')
        print(self.Dm.shape)
        print(self.Dm)
        print('exUp')
        print(self.exUp.shape)
        print(self.exUp)
        

    def matvec(self, v):
        """Matrix vector multiplication

        """
        if v.shape[0] != 3*self.n:
            raise ValueError("Moo!")
        
        n = self.n
        sol = np.zeros_like(v)
        v1,v2,v3 = v[:n],v[n:2*n],v[2*n:]
        sol[:n] = self.exUmm.dot(v1) + self.exUmp.dot(v2) - self.exUp.dot(v3)
        sol[n:2*n] = (self.Dm.dot(self.Um.dot(self.exUmp.dot(v2) - self.exUmm.dot(v1))) +
                      self.Dp.dot(self.Up.dot(self.exUp.dot(v3))))
        sol[2*n:] = v1 + v2 - (3.0*self.Dm.dot(self.Um.dot(v2 - v1)))
        return sol

@PrintRes
def linsolve(M,b):
    A = aslinearoperator(M)
    v, info = solve(A,b,x0=b,tol=1e-6)
    if info == 0:
        return v
    elif info <0:
        raise Exception('Breakdown or bad input with code %d' % info)
    else:
        raise Exception('Non-convergence after %d iterations' % info)

def diag_inv(A):
    """Invert the diagonal of a matrix, assumes no zeros.

    """

    B = A.copy()
    n = A.shape[0]
    for i in range(n):
        B[i,i] = 1.0 / A[i,i]
    return B

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
