from ADMMsRustici.Solver import *

class MyBacktrackingSolverClass(SolverClass):


	def __init__(self, varMod, x0, y0, l0, beta0, Lphi):

		super().__init__(varMod, x0, y0, l0, beta0)

		self.IterationStep = self.__MyStep__

		self.PQ = np.hstack([self.VarModel.P, self.VarModel.Q])

		self.Lphi = Lphi
		self.minSigma = np.min(np.linalg.svdvals(self.PQ))

		self.residueConst = (self.Lphi / self.minSigma)


	def __MyStep__(self, xk, yk, lk, betak):

		xk1, yk1 = self.__MnonTangere__(xk, yk, lk, betak)
		xk2, yk2, betak1 = self.__backtrackBeta__(xk1, yk1, betak)
		lk1 = lk + betak1 * (self.VarModel.D @ xk1 - yk1)

		print("#############################")
		print(f"{betak1}")

		zk2 = np.concat([xk2, yk2])
		betak1 = 1 / ( np.linalg.norm(self.PQ @ zk2 - self.VarModel.c) )
				
		return xk2, yk2, lk1, betak1

	def __AugmLag__(self, varX, varY, l, beta):

			fid = self.VarModel.fidelity(varX)
			reg = self.VarModel.regularizer(varY)
			res = self.VarModel.P @ varX + self.VarModel.Q @ varY - self.VarModel.c
			prodScal = np.dot(res, l)

			return (self.VarModel.mu / 2) * fid + reg + prodScal + (beta / 2) * np.linalg.norm(res)**2


	def __approxMinimum__(self, x0 :np.ndarray, y0 : np.ndarray, l0 : np.ndarray, beta0, err = 1e-6):

		L0 = self.__AugmLag__(x0, y0, l0, beta0)
		R = L0 + (1/(2 * beta0)) * np.linalg.norm(l0)**2
		Diam = (2/self.minSigma) * (np.sqrt(R) * np.sqrt(2 / beta0) + np.linalg.norm(self.Varmodel.c) + (1/beta0) * np.linalg.norm(l0))

		maxIter = 1000
		PointFound = False
		iter = 0

		xk = x0
		yk = y0

		while (not(PointFound) and (iter <= maxIter)):

			xk1, yk1 = self.VarModel.primalStep(xk, yk, l0, beta0)
			DualRes = beta0 * self.VarModel.P.T @ (self.VarModel.Q @ (yk1 - yk))
			DualResNorm = np.linalg.norm(DualRes)

			if (DualResNorm * Diam <= err):
				PointFound = True
			else:
				iter += 1


		return (xk1, yk1)


	


	def __MnonTangere__(self, x0 :np.ndarray, y0 : np.ndarray, l0 : np.ndarray, beta0):

		L0 = self.__AugmLag__(x0, y0, l0, beta0)
		R0 = L0 + (1/(2 * beta0)) * np.linalg.norm(l0)**2
		Diam0 = (2/self.minSigma) * (np.sqrt(R0) * np.sqrt(2 / beta0) + np.linalg.norm(self.Varmodel.c) + (1/beta0) * np.linalg.norm(l0))

		beta1 = beta0 + 1
		L1 = self.__AugmLag__(x0, y0, l0, beta1)
		R1 = L1 + (1/(2 * beta1)) * np.linalg.norm(l0)**2
		Diam1 = (2/self.minSigma) * (np.sqrt(R1) * np.sqrt(2 / beta1) + np.linalg.norm(self.Varmodel.c) + (1/beta1) * np.linalg.norm(l0))

		maxIter = 1000
		PointFound = False
		iter = 0

		xk = x0
		yk = y0

		uk = x0
		vk = y0

		while (not(PointFound) and (iter <= maxIter)):

			xk1, yk1 = self.VarModel.primalStep(xk, yk, l0, beta0)
			uk1, vk1 = self.VarModel.primalStep(uk, vk, l0, beta1)

			Lk0 = self.__AugmLag__(xk1, yk1, l0, beta0)
			Lk1 = self.__AugmLag__(uk1, vk1, l0, beta1)

			DualRes0 = beta0 * self.VarModel.P.T @ (self.VarModel.Q @ (yk1 - yk))
			DualRes1 = beta1 * self.VarModel.P.T @ (self.VarModel.Q @ (vk1 - vk))

			DualResNorm0 = np.linalg.norm(DualRes0)
			DualResNorm1 = np.linalg.norm(DualRes1)

			err0 = DualResNorm0 * Diam0
			err1 = DualResNorm1 * Diam1

			deltaL = np.abs(Lk1 - Lk0)
			ERR = 2 * np.max(err0, err1)

			if (err0 <= beta0 * (deltaL - ERR)):
				PointFound = True
			else:
				iter += 1


		return (xk1, yk1)
	

	def __backtrackBeta__(self, x0, y0, beta0):

		ImgTol = 1e-3
		tolResidue = 1e-3
		l0 = self.lk
		epsilon = self.VarModel.P @ x0 + self.VarModel.Q @ y0 - self.VarModel.c

		xk = x0
		yk = y0		
		
		beta = beta0
		ImgErr = np.inf

		while (ImgErr >= ImgTol):			

			lBeta = l0 + beta * epsilon

			xk1, yk1 = self.VarModel.primalStep(xk, yk, lBeta, beta0)

			rk1 = self.VarModel.P @ xk1 + self.VarModel.Q @ xk1 - self.VarModel.c
			resk1 = np.dot(epsilon, rk1)

			while (np.abs(resk1) >= tolResidue):

				betaLower = 0
				betaUpper = np.inf

				#print("FindingBetaIter")

				if (resk1 < 0):
					betaUpper = beta
				else:
					betaLower = beta

				if (not(betaUpper == np.inf)):
					beta = (betaUpper + betaLower)/2
				else:
					beta *= 2

				lBeta = l0 + beta * epsilon

				xk1, yk1 = self.VarModel.primalStep(xk, yk, lBeta, beta0)

				rk1 = self.VarModel.P @ xk1 + self.VarModel.Q @ xk1 - self.VarModel.c
				resk1 = np.dot(epsilon, rk1)


			Lk = self.__AugmLag__(xk, yk, lBeta, beta0)
			Rk = Lk + (1/(2 * beta0)) * np.linalg.norm(lBeta)**2
			Diamk = (2/self.minSigma) * (np.sqrt(Rk) * np.sqrt(2 / beta0) + np.linalg.norm(self.Varmodel.c) + (1/beta0) * np.linalg.norm(lBeta))

			DualResk = beta0 * self.VarModel.P.T @ (self.VarModel.Q @ (yk1 - yk))
			DualResNormk = np.linalg.norm(DualResk)

			ImgErr = Diamk * DualResNormk
			print(f"ImgErr: {ImgErr}")

			xk = xk1
			yk = yk1


		return xk, yk, beta		


		
