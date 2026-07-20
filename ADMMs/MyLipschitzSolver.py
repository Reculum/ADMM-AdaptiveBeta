from ADMMs.Solver import *
from models.VariationalModel import *

class My_Lipschitz_SolverClass(SolverClass):


	def __init__(self, varMod : VariationalModelClass, x0, y0, l0, beta0, L, H):

		super().__init__(varMod, x0, y0, l0, beta0)

		self.L = L
		self.H = H
		self.IterationStep = self.__MyStep__


	def __MyStep__(self, xk, yk, lk, betak):

		xk1, yk1 = self.__MnonTangere__(xk, yk, lk, betak)
		xk2, yk2, betak1 = self.__backtrackBeta__(xk1, yk1, betak)
		lk1 = lk + betak1 * (self.VarModel.D @ xk1 - yk1)

		return xk2, yk2, lk1, betak1

	def __AugmLag__(self, varX, varY):

			phiZ = self.VarModel(np.concat([varX, varY]))
			res = np.linalg.norm(self.VarModel.P @ varX - self.VarModel.Q @ varY
								- self.VarModel.c)**2
			prodScal = cvxpy.scalar_product(self.lk, res)

			return (self.mu / 2) * fid + reg + prodScal + (self.betak / 2) * cvxpy.sum_squares(res)

	def __TrustRegionDescent(self, x0 : np.ndarray, y0 : np.ndarray):
		pass

	def __MnonTangere__(self, x0 :np.ndarray, y0 : np.ndarray, l0 : np.ndarray, beta0):

		tol = 1e-6
		tolStep = 1e-3

		PointFound = False

		while (not PointFound):

			L0 = self.__AugmLag__(cvxpy.Constant(x0), cvxpy.Constant(y0))
			problem = cvxpy.Problem(cvxpy.Minimize(g), [self.__AugmLag__(x, y) <= L0])
			problem.solve(solver=cvxpy.CLARABEL, verbose=False)

			xk = x.value
			yk = y.value

			if (np.linalg.norm(D @ xk - yk) > tol): PointFound = True
			else:
				
				x1, y1 = self.VarModel.primalStep(x0, y0, l0, beta0)

				if (np.linalg.norm(np.concat([x1, y1]) - np.concat([x0, y0]))) <= tolStep:
					return x1, y1
				
				x0 = x1
				y0 = y1

		
		return xk, yk
	

	def __backtrackBeta__(self, x0, y0, beta0):

		tol = 1e-6

		nX = x0.size
		nY = y0.size

		D = self.VarModel.D

		x = cvxpy.Variable(nX)
		y = cvxpy.Variable(nY)

		epsilon = D @ x0 - y0

		betaLower = 0
		betaUpper = np.inf
		beta = beta0

		#initial iter

		f = self.__AugmLag__(x, y) + beta * cvxpy.scalar_product(epsilon, D @ x - y)
		problem = cvxpy.Problem(cvxpy.Minimize(f))
		problem.solve(solver=cvxpy.CLARABEL, verbose=False)

		xk = x.value
		yk = y.value
		rk = D @ xk - yk

		while (np.abs(np.dot(epsilon, rk)) >= tol):

			if (np.dot(epsilon, rk) > 0):
				betaLower = beta
			else: betaUpper = beta

			if (not (betaUpper == np.inf)):
				beta = (betaUpper + betaLower)/2
			else:
				beta *= 2

			f = self.__AugmLag__(x, y) + beta * cvxpy.scalar_product(epsilon, D @ x - y)
			problem = cvxpy.Problem(cvxpy.Minimize(f))
			problem.solve(solver=cvxpy.CLARABEL, verbose=False)

			xk = x.value
			yk = y.value
			rk = D @ xk - yk
		

		return xk, yk, beta