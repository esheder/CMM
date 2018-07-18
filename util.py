
import numpy as np
from scipy.sparse.linalg import aslinearoperator
from scipy.sparse.linalg import bicgstab as solve

class BCCLS:
    """Boundary conditions linear operator like object

    """
    def __init__(self, exUp, exUmp, exUmm, Up, Um, Dp, Dm, d):
        self.exUp = exUp
        self.exUmp = exUmp
        self.exUmm = exUmm
        self.Up = Up
        self.Um = Um
        self.Dp = Dp
        self.Dm = Dm
        self.d = d
        self.n = Dm.shape[0]
        self.shape = (3*self.n,3*self.n)
        self.dtype = 'float64'
        #self._test_()

    def _test_(self):
        """
        print('Up')
        print(self.Up)
        print('Um')
        print(self.Um)
        print('Dp')
        print(self.Dp)
        print('Dm')
        print(self.Dm)
        """
        print('exUp')
        print(self.exUp)
        

    def matvec(self, v):
        """Matrix vector multiplication.

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
        #self.__test__()

    def __test__(self):
        print(self.sigA)

    
def linsolve(M,b):
    A = aslinearoperator(M)
    v, info = solve(A,b,x0=b,tol=1e-6)
    if info == 0:
        return v
    else:
        print(info)


def diag_inv(A):
    """Invert the diagonal of a matrix, assumes no zeros.

    """

    B = A.copy()
    n = A.shape[0]
    for i in range(n):
        B[i,i] = 1.0 / A[i,i]
    return B
