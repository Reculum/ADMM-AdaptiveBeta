from ADMMs.Solver import *
from models.TVL2_1D import *

class MyBacktrackingSolverClass(SolverClass):


	def __init__(self, varMod, x0, y0, l0, beta0):

		super().__init__(varMod, x0, y0, l0, beta0)

		self.IterationStep = self.__MyStep__

		self.PQ = np.hstack([self.VarModel.P, self.VarModel.Q])


	def __MyStep__(self, xk, yk, lk, betak):

		xk1, yk1 = self.__MnonTangere__(xk, yk, lk, betak)
		xk2, yk2, betak1 = self.__backtrackBeta__(xk1, yk1, betak)
		lk1 = lk + betak1 * (self.VarModel.D @ xk1 - yk1)

		print("#############################")
		print(f"{betak1}")

		zk2 = np.concat([xk2, yk2])
		betak1 = 1 / ( np.linalg.norm(self.PQ @ zk2 - self.VarModel.c) )
				
		return xk2, yk2, lk1, betak1

	# def __AugmLag__(self, varX, varY):

	# 		fid = cvxpy.sum_squares(self.A @ varX - self.b)
	# 		reg = cvxpy.norm1(varY)
	# 		res = cvxpy.matmul(self.D, varX) - varY
	# 		prodScal = cvxpy.scalar_product(self.lk, res)

	# 		return (self.mu / 2) * fid + reg + prodScal + (self.betak / 2) * cvxpy.sum_squares(res)

	def __MnonTangere__(self, x0 :np.ndarray, y0 : np.ndarray, l0 : np.ndarray, beta0):

		#iterate the Primal Step until we are close to the global minimum

		tolStep = 1e-3

		PointFound = False

		xk = x0
		yk = y0

		while (not PointFound):
				
			xk1, yk1 = self.VarModel.primalStep(xk, yk, l0, beta0)

			if (np.linalg.norm(np.concat([xk1, yk1]) - np.concat([xk, yk]))) <= tolStep:
				PointFound = True
			
			xk = xk1
			yk = yk1

			print("MNonTangereIter")
		
		return xk, yk
	

	def __backtrackBeta__(self, x0, y0, beta0):

		tolStep = 1e-3
		tolResidue = 1e-3
		c = self.VarModel.c	

		xk = x0
		yk = y0

		z0 = np.concat([x0, y0])
		zk = np.concat([xk, yk])		
		

		l0 = self.lk
		epsilon = self.PQ @ z0 - c
		beta = beta0
		stepSize = np.inf	


		while (stepSize >= tolStep):

			betaLower = 0
			betaUpper = np.inf

			Lbeta = l0 + beta * epsilon

			xk1, yk1 = self.VarModel.primalStep(xk, yk, Lbeta, beta0)

			zk1 = np.concat([xk1, yk1])
			rk1 = self.PQ @ zk1 - c

			resk1 = np.dot(epsilon, rk1)

			while (np.abs(resk1) >= tolResidue):

				#print("FindingBetaIter")

				if (resk1 < 0):
					betaUpper = beta
				else:
					betaLower = beta

				if (not(betaUpper == np.inf)):
					beta = (betaUpper + betaLower)/2
				else:
					beta *= 2

				Lbeta = l0 + beta * epsilon

				xk1, yk1 = self.VarModel.primalStep(xk, yk, Lbeta, beta0)

				zk1 = np.concat([xk1, yk1])
				rk1 = self.PQ @ zk1 - c

				resk1 = np.dot(epsilon, rk1)


			stepSize = np.linalg.norm(zk1 - zk)
			print(f"StepSize: {stepSize}")

			xk = xk1
			yk = yk1
			zk = zk1


		return xk, yk, beta	
