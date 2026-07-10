from ADMMs.Solver import *



class SpectralSolverClass(SolverClass):

	

	#approx. 2**(-1 / 100)
	ro = 0.993
	omega = ro


	def __init__(self, varMod, x0, y0, l0, beta0):

		super().__init__(varMod, x0, y0, l0, beta0)
		self.IterationStep = self.__SpectralStep__

	def __proj__(self, Tmin, Tmax, x):
		if(x < Tmin): return Tmin
		elif(x > Tmax): return Tmax
		else: return x



	def __SpectralStep__(self, x, y, l, beta):

		E = self.VarModel.Q

		x_k_1, y_k_1 = self.VarModel.primalStep(x, y, l, beta)
		l_k_1 = self.VarModel.dualStep(x_k_1, y_k_1, l, beta)
		spectralRatio = (np.linalg.norm(l_k_1) / np.linalg.norm(E @ y_k_1))

		wk = self.omega
		beta_k_1 = (1 - wk) * beta + wk * self.__proj__(1e-4, 1e4, spectralRatio)

		self.omega *= self.ro
		
		return (x_k_1, y_k_1, l_k_1, beta_k_1)