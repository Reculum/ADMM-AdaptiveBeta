from LanzaModels.VariationalModel import *


class TIK_1DClass(VariationalModelClass):

    #L(x, y, l)_{beta} = phi(x, y) + <l, (D, -Id) @ (x, y)> + beta/2 | (D, -Id) @ (x, y)|^{2}
    #phi(x, y) = (mu / 2) * |A @ x - b|_{2}^{2} + (1/2) * |y|_{2}^{2} 

    #n is the dimension of D in M: n x n
    #we applied anti-reflexive BCs

    D : np.ndarray
    A : np.ndarray
    b : np.ndarray
    AtA : np.ndarray
    DtD : np.ndarray
    Atb : np.ndarray 

    def __init__(self, A, b, mu):

        _, self.n = np.shape(A)
        self.A = A
        self.b = b

        uni = np.ones(shape=(self.n - 1,))
        meno_uni = -1 * np.ones(shape=(self.n, ))
        self.D = np.diag(meno_uni, 0) + np.diag(uni, 1)
        self.D[self.n - 1][self.n - 1] = 1
        self.D[self.n - 1][self.n - 2] = -1

        self.setXmatrixConstr(self.D)
        self.setYmatrixConstr(-np.eye(self.n))
        self.setConstrObj(np.zeros(shape=(self.n,)))

        self.DtD = (self.D).T @ self.D
        self.AtA = (self.A).T @ self.A
        self.Atb = (self.A).T @ self.b

        fid = lambda x: np.linalg.norm( A @ x - b )**2
        reg = lambda x: np.linalg.norm(self.D @ x, ord=2)
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
        return (1 / (beta + 1)) * (beta * (self.D @ x) + l)
    
    def __proxStep__(self, x_k, y_k, l, beta):

        x_k_1 = self.__proX__(y_k, l, beta)
        y_k_1 = self.__proY__(x_k_1, l, beta)

        return x_k_1, y_k_1
    

    def __lambdaStep__(self, x_k, y_k, l, beta):

        return l + beta * (self.D @ x_k - y_k)

    def __call__(self, x):
        return (self.mu/2) * self.fidelity(x) + self.regularizer(x)

        



