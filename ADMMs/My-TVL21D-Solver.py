from ADMMs.Solver import *
from models.TVL2_1D import *

class My_TVL21D_SolverClass(SolverClass):


	def __init__(self, varMod : TVL2_1DClass, x0, y0, l0, beta0):

		super().__init__(varMod, x0, y0, l0, beta0)
		self.IterationStep = self.__StdStep__


	def __MyStep__(self, xk, yk, lk, betak):

		xk1, yk1 = self.__MnonTangere__(xk, yk, lk, betak)
		betak1 = self.__backtrackBeta__(xk1, yk1, betak)
		lk1 = lk + betak1 * (self.VarModel.D @ xk1 - yk1)

		return xk1, yk1, lk1, betak1

	def __AugmLag__(self, varX, varY):

			fid = self.VarModel.fidelity(varX)
			reg = self.VarModel.regularizer(varY)
			res = self.VarModel.D @ varX - varY

			prodScal = np.dot(self.lk, res)

			return (self.VarModel.mu / 2) * fid + reg + prodScal + (self.betak / 2) * np.linalg.norm(res)**2	

	def __MnonTangere__(self, x0 :np.ndarray, y0 : np.ndarray, l0 : np.ndarray, beta0):

		tol = 1e-6

		nX = x0.size()
		nY = y0.size()

		D = self.VarModel.D

		x = cvxpy.Variable(nX)
		y = cvxpy.Variable(nY)	
		
		g = cvxpy.sum_squares(D @ x - y)

		PointFound = False

		while (not PointFound):

			L0 = self.__AugmLag__(cvxpy.Constant(x0), cvxpy.Constant(y0))
			problem = cvxpy.Problem(cvxpy.Minimize(g), [self.__AugmLag__(x, y) <= L0])
			problem.solve(solver=cvxpy.ECOS, verbose=False)

			xk = x.value
			yk = y.value

			if (np.linalg.norm(D @ xk - yk) > tol): PointFound = True
			else:
				x0, y0 = self.VarModel.primalStep(x0, y0, l0, beta0)

		
		return xk, yk
	

	def __backtrackBeta__(self, x0, y0, beta0):

		tol = 1e-6

		nX = x0.size()
		nY = y0.size()

		D = self.VarModel.D

		x = cvxpy.Variable(nX)
		y = cvxpy.Variable(nY)

		epsilon = D @ x0 - y0

		xk = x0
		yk = y0
		rk = D @ xk - yk

		betaLower = 0
		betaUpper = np.inf()
		beta = beta0

		while (np.dot(epsilon, rk) >= tol):

			f = self.__AugmLag__(x, y) + beta * cvxpy.scalar_product(epsilon, D @ x - y)
			problem = cvxpy.Problem(cvxpy.Minimize(f))
			problem.solve(solver=cvxpy.ECOS, verbose=False)

			xk = x.value
			yk = y.value
			rk = D @ xk - yk

			if (np.dot(epsilon, rk) > 0):
				betaLower = beta
			else: betaUpper = beta

			if (not (betaUpper == np.inf)):
				beta = (betaUpper + betaLower)/2
			else:
				beta *= 2
		

		return beta







		






	




	
	