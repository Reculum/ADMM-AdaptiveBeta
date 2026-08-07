from ADMMsRustici.Solver import *



class FigueredoSolverClass(SolverClass):

    def __init__(self, varMod, x0, y0, l0, beta0):
        super().__init__(varMod, x0, y0, l0, beta0)
        self.IterationStep = self.__SpectralStep__
        
        # State tracking variables for adaptivity (checkpoint k0)
        self.iterCounter = 0
        self.xkHAT = None
        self.ykHAT = None
        self.lkHAT = None
        self.l_k0 = None

    def __proj__(self, Tmin, Tmax, x):
        if x < Tmin: 
            return Tmin
        elif x > Tmax: 
            return Tmax
        else: 
            return x

    def __SpectralStep__(self, x, y, l, beta):
        # 1. Compute standard primal and dual updates for iteration k+1
        x_k_1, y_k_1 = self.VarModel.primalStep(x, y, l, beta)
        l_k_1 = self.VarModel.dualStep(x_k_1, y_k_1, l, beta)

        # 2. Compute intermediate dual predictor: lkHAT_next = l + beta * (P*x_{k+1} + Q*y_k - c)
        lkHAT_next = l + beta * (self.VarModel.P @ x_k_1 + self.VarModel.Q @ y - self.VarModel.c)

        beta_k_1 = beta

        # 3. Initialize checkpoint k0 on the very first iteration
        if self.xkHAT is None:
            self.xkHAT = x_k_1.copy()
            self.ykHAT = y_k_1.copy()
            self.lkHAT = lkHAT_next.copy()
            self.l_k0 = l_k_1.copy()
        else:
            # 4. Adapt penalty parameter every Tf = 2 iterations
            if self.iterCounter > 0 and self.iterCounter % 2 == 0:
                # Difference vectors relative to reference checkpoint k0 (Eq. 17)
                d_h_hat = self.VarModel.P @ (x_k_1 - self.xkHAT)
                d_lam_hat = self.lkHAT - lkHAT_next
                d_g_hat = self.VarModel.Q @ (y_k_1 - self.ykHAT)
                d_lam = self.l_k0 - l_k_1

                def _compute_spectral_step(d_f, d_l, current_beta):
                    dot_prod = np.dot(d_f, d_l)
                    norm_f = np.linalg.norm(d_f)
                    norm_lam = np.linalg.norm(d_l)
                    cor = dot_prod / (norm_f * norm_lam + 1e-12) # Eq. 20

                    if np.abs(dot_prod) > 1e-10 and norm_f > 1e-12:
                        alpha_SD = np.dot(d_l, d_l) / dot_prod
                        alpha_MG = dot_prod / (norm_f ** 2)
                        step = alpha_MG if (2 * alpha_MG > alpha_SD) else (alpha_SD - 0.5 * alpha_MG) # Eq. 18
                    else:
                        step = current_beta
                    return step, cor

                # Spectral stepsizes for x-update (alpha) and y-update (beta)
                alpha_hat, alpha_cor = _compute_spectral_step(d_h_hat, d_lam_hat, beta)
                beta_hat, beta_cor = _compute_spectral_step(d_g_hat, d_lam, beta)

                # Safeguarding update logic (Eq. 21 with eps_cor = 0.2)
                eps_cor = 0.2
                if alpha_cor > eps_cor and beta_cor > eps_cor:
                    beta_k_1 = np.sqrt(max(alpha_hat * beta_hat, 1e-8))
                elif alpha_cor > eps_cor and beta_cor <= eps_cor:
                    beta_k_1 = max(alpha_hat, 1e-4)
                elif alpha_cor <= eps_cor and beta_cor > eps_cor:
                    beta_k_1 = max(beta_hat, 1e-4)
                else:
                    beta_k_1 = beta

                # print(f"[Iter {self.iterCounter}] ")
                # print(f"a_cor: {alpha_cor:.2f}, b_cor: {beta_cor:.2f} | ")
                # print(f"a_hat: {alpha_hat:.4f}, b_hat: {beta_hat:.4f} | ")
                # print(f"New Beta: {beta_k_1:.4f}")

                # Update reference checkpoint k0 for next adaptation window
                self.xkHAT = x_k_1.copy()
                self.ykHAT = y_k_1.copy()
                self.lkHAT = lkHAT_next.copy()
                self.l_k0 = l_k_1.copy()

        self.iterCounter += 1
        return (x_k_1, y_k_1, l_k_1, beta_k_1)