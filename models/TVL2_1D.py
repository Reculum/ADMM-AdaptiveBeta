

from VariationalModel import *


class TVL2_1DClass(VariationalModelClass):

    #L(x, y, l)_{beta} = phi(x, y) + <l, (D, -Id) @ (x, y)> + beta/2 | (D, -Id) @ (x, y)|^{2}
    #phi(x, y) = (mu / 2) * |A @ x - b|_{2}^{2} + |y|_{1} 

    #n is the dimension of D in M: (n-1) x n

    A = None
    D = None
    AtA = None
    DtD = None
    Atb = None
    b = None

    def __init__(self, A, b, mu):

        _, self.n = np.shape(A)

        uni = np.ones(shape=(self.n,))
        meno_uni = -1 * np.ones(shape=(self.n - 1, ))

        self.A = A
        self.D = (np.diag(uni, 0) + np.diag(meno_uni, 1))[:(self.n-1), :]

        self.DtD = (self.D).T @ self.D
        self.AtA = (self.A).T @ self.A
        self.Atb = (self.A).T @ b

        fid = lambda x: (mu / 2) * np.linalg.norm( A @ x - b )**2
        reg = lambda x: np.linalg.norm(x[:len(x)-1] - x[1:], ord=1)
        proxstep = lambda x, y, l, beta: self.__proxStep__(x, y, l, beta)
        dualstep = lambda x, y, l, beta: self.__lambdaStep__(x, y, l, beta)

        self.setFidelity(fid)
        self.setRegularizer(reg)
        self.setMu(mu)

        self.setPrimalStep(proxstep)
        self.setDualStep(dualstep)

    
    def __proX__(self, y, l, beta):
            
        tempA = self.DtD + (self.mu / beta) * self.AtA
        tempB = (self.D).T @ (y - (1/beta) * l) + (self.mu / beta) * self.Atb

        upper, lower = la.cho_factor(tempA, lower=True)

        return la.cho_solve( (upper, lower), tempB )

    def __proY__(self, x, l, beta):

        q = (self.D @ x) + (1/beta) * l
        qSize = np.size(q)
        
        OneOverBeta = (1/beta) * np.ones(shape=(qSize,))

        return np.sign(q) * np.maximum(np.abs(q) - OneOverBeta, np.zeros(shape=(qSize,)))
    
    def __proxStep__(self, x_k, y_k, l, beta):

        x_k_1 = self.__proX__(y_k, l, beta)
        y_k_1 = self.__proY__(x_k_1, l, beta)

        return x_k_1, y_k_1
    

    def __lambdaStep__(self, x_k, y_k, l, beta):

        return l + beta * (self.D @ x_k - y_k)

        



