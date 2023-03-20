import numpy as np
import scipy
from scipy.special import expit
from scipy.sparse import csr_matrix


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')
    
    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """
    
    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        # print('x', x.shape)
        # print('A', self.A.shape)
        # print('b', self.b.shape)
        return 0.5 * np.dot(x, np.dot(self.A, x)) - np.dot(self.b, x)
        # your code here

    def grad(self, x):
        return np.dot(self.A, x) - self.b
        # your code here

        
class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.
    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()
    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        
        xTAb = self.matvec_Ax(x) * self.b
        
        return 1.0 / self.b.shape[0] * np.sum(np.log(1 + np.exp(-xTAb))) + self.regcoef * 0.5 * np.linalg.norm(x) ** 2
        # your code here

    def grad(self, x):
        xTAb = self.matvec_Ax(x) * self.b
        return -1.0 / self.b.shape[0] * self.matvec_ATx(self.b * scipy.special.expit(-xTAb)) + self.regcoef * x
        # your code here


def create_log_reg_oracle(A, b, regcoef):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """
    # if isinstance(A, csr_matrix):
    #     A = A.toarray() 
    # matvec_Ax = lambda x: np.dot(A, x)  # your code here
    # matvec_ATx = lambda x: np.dot(A.T, x)  # your code here
    matvec_Ax = lambda x: A * x if scipy.sparse.issparse(A) else np.dot(A, x)  ## your code here
    matvec_ATx = lambda x: A.T * x if scipy.sparse.issparse(A) else np.dot(A.T, x)  # your code here

    def matmat_ATsA(s):
        # your code here
        if (scipy.sparse.issparse(A)):
            return A.T * np.diag(s) * A
        # return np.dot(A.T, np.dot(np.diag(s), A))
        return A.T.dot(np.diag(s)).dot(A)
        # return None

    return LogRegL2Oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)
