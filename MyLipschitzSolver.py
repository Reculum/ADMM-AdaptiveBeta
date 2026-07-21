from ADMMs.Solver import *
from models.VariationalModel import *

class My_Lipschitz_SolverClass(SolverClass):


	def __init__(self, varMod : VariationalModelClass, x0, y0, l0, beta0, L, H, dPhi):

		super().__init__(varMod, x0, y0, l0, beta0)

		self.L = L
		self.H = H
		self.PQ = np.hstack([self.VarModel.P, self.VarModel.Q])
		self.dPhi = dPhi

		self.IterationStep = self.__MyStep__


	def __MyStep__(self, xk, yk, lk, betak):

		xk1, yk1 = self.__MnonTangere__(xk, yk, lk, betak)
		xk2, yk2, betak1 = self.__backtrackBeta__(xk1, yk1, betak)
		lk1 = lk + betak1 * (self.VarModel.D @ xk1 - yk1)

		return xk2, yk2, lk1, betak1

	def __AugmLag__(self, varX, varY):

			fid = self.VarModel.fidelity(varX)
			reg = self.VarModel.regularizer(varY)
			res = self.VarModel.P @ varX - self.VarModel.Q @ varY - self.VarModel.c
			prodScal = np.dot(self.lk, res)

			return (self.VarModel.mu / 2) * fid + reg + prodScal + (self.betak / 2) * np.linalg.norm(res)**2

	def __TrustRegionDescent__(self, x0 : np.ndarray, y0 : np.ndarray, eta, df, tol = 1e-6):

		c = self.VarModel.c

		z0 = np.concat([x0, y0])	

		xk = x0
		yk = y0
		zk = z0
		fk = self.__AugmLag__(xk, yk)
		
		resk = np.dot(eta, self.PQ @ z0 - c)
		dg = eta @ self.PQ
		dg = dg / np.linalg.norm(dg)

		Converged = False

		while (not(Converged)):

			dfk = df(zk)
			dfk = dfk / np.linalg.norm(dfk)
			
			rLow = 0
			rUpp = np.inf
			r = 1

			while ( rUpp - rLow > (tol / (2 * (self.L + self.H))) ):

				zk1 = zk - r * (dfk + dg)
				xk1, yk1 = zk1[:self.VarModel.n], zk1[self.VarModel.n:]

				fk1 = self.__AugmLag__(xk1, yk1)

				if (fk1 > fk):
					rUpp = r
					r = (rUpp + rLow)/2
				elif (not(rUpp == np.inf)):
					rLow = r
					r = (rUpp + rLow)/2
				else:
					rLow = r
					r *= 2

			if ((self.L + self.H) * np.linalg.norm(zk1 - zk) <=  tol):
				Converged = True

			xk = xk1
			yk = yk1
			zk = zk1
			fk = fk1
			resk = np.dot(eta, self.PQ @ zk - c)
		

		return (zk, resk > tol)

		


	def __MnonTangere__(self, x0 :np.ndarray, y0 : np.ndarray, l0 : np.ndarray, beta0):

		tol = 1e-6
		tolStep = 1e-3
		df = lambda z: self.dPhi(z) + (self.lk @ self.PQ) + self.betak * ( (self.PQ @ z - c) @ self.PQ )

		PointFound = False

		P = self.VarModel.P
		Q = self.VarModel.Q
		PQ = np.hstack([P, Q])
		c = self.VarModel.c

		while (not PointFound):

			z0 = np.concat([x0, y0])
			eta = PQ @ z0 - c
			eta = eta / np.linalg.norm(eta)	

			(zk, PositiveRes) = self.__TrustRegionDescent__(x0, y0, eta, df, tol)
			xk, yk = zk[:self.VarModel.n], zk[self.VarModel.n:]

			if (PositiveRes): PointFound = True
			else:
				
				x1, y1 = self.VarModel.primalStep(x0, y0, l0, beta0)

				if (np.linalg.norm(np.concat([x1, y1]) - np.concat([x0, y0]))) <= tolStep:
					return x1, y1
				
				x0 = x1
				y0 = y1

		
		return xk, yk
	

	def __backtrackBeta__(self, x0, y0, beta0, tol = 1e-6):

		tol = 1e-6
		c = self.VarModel.c

		z0 = np.concat([x0, y0])
		epsilon = self.PQ @ z0 - c

		betaNegative = beta0
		dfNegative = lambda z: self.dPhi(z) + (self.lk @ self.PQ) + self.betak * ( (self.PQ @ z - c) @ self.PQ ) \
							   + betaNegative * (epsilon, self.PQ @ z - c)
		
		(Mink, IsResPositive) = self.__TrustRegionDescent__(x0, y0, epsilon, dfNegative)
		
		xMin, yMin = Mink[:self.VarModel.n], Mink[self.VarModel.n:]

		#find negative lower bound

		while (IsResPositive):
			
			betaNegative *= 2
			dfNegative = lambda z: self.dPhi(z) + (self.lk @ self.PQ) + self.betak * ( (self.PQ @ z - c) @ self.PQ ) \
							   + betaNegative * (epsilon, self.PQ @ z - c)
						
			(Mink, IsResPositive) = self.__TrustRegionDescent__(xMin, yMin, epsilon, dfNegative)
			xMin, yMin = Mink[:self.VarModel.n], Mink[self.VarModel.n:]
		


		betaLower = 0
		betaUpper = betaNegative
		betaPositive = (betaLower + betaUpper)/2
		
		dfPositive = lambda z: self.dPhi(z) + (self.lk @ self.PQ) + self.betak * ( (self.PQ @ z - c) @ self.PQ ) \
							   + betaPositive * (epsilon, self.PQ @ z - c)
		
		(Maxk, IsResPositive) = self.__TrustRegionDescent__(xMin, yMin, -epsilon, dfPositive)
		resk = self.PQ @ Maxk - c
		
		#find proper minimum location

		while (np.abs(np.dot(epsilon, resk)) >= tol):

			if (IsResPositive > 0):
				betaLower = betaPositive
			else: betaUpper = betaPositive

			betaPositive = (betaUpper + betaLower)/2

			dfPositive = lambda z: self.dPhi(z) + (self.lk @ self.PQ) + self.betak * ( (self.PQ @ z - c) @ self.PQ ) \
							   + betaPositive * (epsilon, self.PQ @ z - c)
		
			(Maxk, IsResPositive) = self.__TrustRegionDescent__(xMin, yMin, -epsilon, dfPositive)
			resk = self.PQ @ Maxk - c
		
		xMax, yMax = Maxk[:self.VarModel.n], Maxk[self.VarModel.n:]
		

		#backtrack beta to match the minimum
		betaLower = 0
		betaUpper = betaPositive
		beta = (betaUpper + betaLower)/2

		isMax = False
		isMin = False

		while (not(isMax) or not(isMin)):

			
			df = lambda z: self.dPhi(z) + (self.lk @ self.PQ) + self.betak * ( (self.PQ @ z - c) @ self.PQ ) \
								+ beta * (epsilon, self.PQ @ z - c)
			
			(CandidateMinumum, _) = self.__TrustRegionDescent__(xMax, yMax, epsilon, df)
			(CandidateMaximum, _) = self.__TrustRegionDescent__(xMax, yMax, -epsilon, df)

			if (np.linalg.norm(CandidateMinumum - Maxk) <= tol):
				isMin = True
			
			if (np.linalg.norm(CandidateMaximum - Maxk) <= tol):
				isMax = True
			

			if (isMin):
				betaLower = beta
			if (isMax):
				betaUpper = beta
			
			beta = (betaLower + betaUpper)/2
		

		return xMax, yMax, beta





			



			

