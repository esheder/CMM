"""Mathematical utility for solving the nonlinear problem

"""


import numpy as np
from scipy.sparse.linalg import aslinearoperator
#from scipy.sparse.linalg import bicgstab as solve
from scipy.sparse.linalg import gmres as solve
from scipy.sparse import csr_matrix
from scipy.linalg import expm
from math import pi
import logging

logger = logging.getLogger('main' if __name__ == '__main__' else __name__)
logger.setLevel(logging.DEBUG)
hndlr = logging.FileHandler('test.log', mode='w')
frmtr = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s: %(message)s')
hndlr.setFormatter(frmtr)
logger.addHandler(hndlr)

def PrintRes(f):
    def f_wrap(*args, **kwargs):
        res = f(*args, **kwargs)
        print('The results of %s were:' % f.__name__)
        print(res)
        return res
    return f_wrap

class BCCLS:
    """Boundary conditions linear operator like object

    """
    def __init__(self, exUp, exUmp, exUmm, Up, Um, Dp, Dm, d,
                 logfile='test.log', loglevel=logging.WARNING):
        self.log = logging.getLogger( '%s.%s' %(__name__, str(type(self))) )
        hndlr = logging.FileHandler(logfile)
        self.log.setLevel(loglevel)
        frmtr = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s: \n %(message)s')
        hndlr.setFormatter(frmtr)
        self.log.addHandler(hndlr)
        
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
        self._test_()

    def _test_(self):
        self.log.debug('Testing BCCLS construction')
        self.log.debug('Up')
        self.log.debug(self.Up.shape)
        self.log.debug(self.Up)
        self.log.debug('Um')
        self.log.debug(self.Um.shape)
        self.log.debug(self.Um)
        self.log.debug('Dp')
        self.log.debug(self.Dp.shape)
        self.log.debug(self.Dp)
        self.log.debug('Dm')
        self.log.debug(self.Dm.shape)
        self.log.debug(self.Dm)
        self.log.debug('exUp')
        self.log.debug(self.exUp.shape)
        self.log.debug(self.exUp)
        

    def matvec(self, v):
        """Matrix vector multiplication

        """
        if v.shape[0] != 3*self.n:
            raise ValueError("Moo!")
        
        n = self.n
        sol = np.zeros_like(v)
        v1,v2,v3 = v[:n],v[n:2*n],v[2*n:]
        sol[:n] = partialRight(v1+v2, -self.Dm.dot(self.Um.dot(v2-v1)))
        sol[n:2*n] = self.exUmm.dot(v1) + self.exUmp.dot(v2) - self.exUp.dot(v3)
        sol[2*n:] = (-self.Dm.dot(self.Um.dot(self.exUmp.dot(v2) - self.exUmm.dot(v1))) -
                     (-self.Dp.dot(-self.Up.dot(self.exUp.dot(v3)))) )
        return sol

#@PrintRes
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

def partialRight(flx, cur):
    return 0.25*( flx - (2.0*cur) )

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
    b[:n] = Jp
    logger.debug('Testing BC input:\n %s', b)
    M = BCCLS(exUp, exUmp, exUmm, Up, Um, Dp, Dm, d)
    v = linsolve(M,b)
    logger.debug('Test linsolve:\n%s', M.matvec(v)-b)
    Ap = v[2*n:]
    logger.debug('Scalar flux on interface:\n%s', exUp.dot(Ap))
    logger.debug('Net current on interface:\n%s', -Dp.dot(-Up.dot(exUp.dot(Ap))))
    logger.debug('Partial right current on interface:\n%s',
                 partialRight(exUp.dot(Ap), -Dp.dot(-Up.dot(exUp.dot(Ap)))) )
    return Ap,exUp
