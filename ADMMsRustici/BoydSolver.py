from ADMMsRustici.Solver import *



class BoydSolverClass(SolverClass):
    
    def __init__(self, varMod, x0, y0, l0, beta0, tau_incr=2, tau_decr=2, mult=10):
        
        super().__init__(varMod, x0, y0, l0, beta0)
        self.IterationStep = self.__ResidualBalancingStep__
        self.tau_incr = tau_incr
        self.tau_decr = tau_decr
        self.mult = mult
        self.k = 0

    def __ResidualBalancingStep__(self, x, y, l, beta):
        E = self.VarModel.Q

        # 1. Standard ADMM primal and dual steps
        x_k_1, y_k_1 = self.VarModel.primalStep(x, y, l, beta)
        l_k_1 = self.VarModel.dualStep(x_k_1, y_k_1, l, beta)

        # 2. Compute primal and dual residual norms
        # ||r^{k+1}|| = (1 / beta) * ||l^{k+1} - l^k||
        r_norm = np.linalg.norm(l_k_1 - l) / beta
        
        # ||s^{k+1}|| = beta * ||E(y^{k+1} - y^k)||
        s_norm = beta * np.linalg.norm(E @ (y_k_1 - y))

        # 3. Update beta if one residual dominates the other
        if r_norm > self.mult * s_norm:
            beta_k_1 = beta * self.tau_incr
        elif s_norm > self.mult * r_norm:
            beta_k_1 = beta / self.tau_decr
        else:
            beta_k_1 = beta

        return (x_k_1, y_k_1, l_k_1, beta_k_1)