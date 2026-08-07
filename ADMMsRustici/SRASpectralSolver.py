from ADMMsRustici.Solver import *



class SRASolverClass(SolverClass):
    """
    Implementation of the Spectral Radius Approximation (SRA) method
    for ADMM penalty parameter selection (McCann & Wohlberg, 2024, Algorithm 1).
    """
    def __init__(self, varMod, x0, y0, l0, beta0, T=5, tau_incr=2, tau_decr=2):
        
        super().__init__(varMod, x0, y0, l0, beta0)
        self.IterationStep = self.__SRAStep__
        self.T = T
        self.tau_incr = tau_incr
        self.tau_decr = tau_decr
        self.k = 0

    def __SRAStep__(self, x, y, l, beta):
        self.k += 1
        E = self.VarModel.Q

        # 1. Standard ADMM primal and dual steps
        x_k_1, y_k_1 = self.VarModel.primalStep(x, y, l, beta)
        l_k_1 = self.VarModel.dualStep(x_k_1, y_k_1, l, beta)

        # 2. Update penalty parameter every T iterations
        if self.k % self.T == 1:
            # p = ||l^{(k+1)} - l^{(k)}||_2 (Norm of dual variable change)
            p = np.linalg.norm(l_k_1 - l)
            
            # q = ||E(y^{(k+1)} - y^{(k)})||_2 (Norm of projected primal change)
            q = np.linalg.norm(E @ (y_k_1 - y))

            # 3. Handle zero/boundary edge cases according to Algorithm 1
            if p == 0 and q > 0:
                beta_k_1 = beta / self.tau_decr
            elif p > 0 and q == 0:
                beta_k_1 = self.tau_incr * beta
            elif p == 0 and q == 0:
                beta_k_1 = beta
            else:
                beta_k_1 = p / q
        else:
            beta_k_1 = beta

        return (x_k_1, y_k_1, l_k_1, beta_k_1)