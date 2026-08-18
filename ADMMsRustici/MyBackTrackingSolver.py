from ADMMsRustici.Solver import *

class MyBacktrackingSolverClass(SolverClass):


	def __init__(self, varMod, x0, y0, l0, beta0, Lphi):

		super().__init__(varMod, x0, y0, l0, beta0)

		self.IterationStep = self.__MyStep__

		self.PQ = np.hstack([self.VarModel.P, self.VarModel.Q])

		self.Lphi = Lphi
		self.minSigma = np.min(np.linalg.svdvals(self.PQ))

		self.residueConst = (self.Lphi / self.minSigma)
		self.ro = 0.5


	def __MyStep__(self, xk, yk, lk, betak):

		ResidueMin0 = (1/betak) * (np.linalg.norm(lk) + self.residueConst)

		xk1, yk1 = self.__MnonTangere__(xk, yk, lk, betak)
		xk2, yk2, beta2, dualResk = self.__backtrackBeta__(xk1, yk1, betak)
		lk1 = lk + beta2 * (self.VarModel.P @ xk1 + self.VarModel.Q @ yk1 - self.VarModel.c)

		betak1 = (np.linalg.norm(lk1) + self.residueConst) / (self.ro * ResidueMin0)
				
		return xk2, yk2, lk1, betak1, dualResk

	def __AugmLag__(self, varX, varY, l, beta):

			# fid = self.VarModel.fidelity(varX)
			# reg = self.VarModel.regularizer(varY)
			res = self.VarModel.P @ varX + self.VarModel.Q @ varY - self.VarModel.c
			prodScal = np.dot(res, l)

			return self.VarModel(varX, varY) + prodScal + (beta / 2) * np.linalg.norm(res)**2


	


	def __MnonTangere__(self, x0 :np.ndarray, y0 : np.ndarray, l0 : np.ndarray, beta0):

		L0 = self.__AugmLag__(x0, y0, l0, beta0)
		R0 = L0 + (1/(2 * beta0)) * np.linalg.norm(l0)**2
		Diam0 = (2/self.minSigma) * (np.sqrt(R0) * np.sqrt(2 / beta0) + np.linalg.norm(self.VarModel.c) + (1/beta0) * np.linalg.norm(l0))

		beta1 = beta0 + 1
		L1 = self.__AugmLag__(x0, y0, l0, beta1)
		R1 = L1 + (1/(2 * beta1)) * np.linalg.norm(l0)**2
		Diam1 = (2/self.minSigma) * (np.sqrt(R1) * np.sqrt(2 / beta1) + np.linalg.norm(self.VarModel.c) + (1/beta1) * np.linalg.norm(l0))

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

			#deltaL = np.abs(Lk1 - Lk0)
			deltaL = Lk1 - Lk0

			#ERR = 2 * np.max([err0, err1])
			ERR = err1

			if (err0 <= beta0 * (deltaL - ERR)):
				PointFound = True
			else:

				xk = xk1
				yk = yk1

				uk = uk1
				vk = vk1

				iter += 1


		print(f"iter Mnon: {iter}")
		print(f"delta L: {deltaL}")
		return (xk1, yk1)
	

	def __backtrackBeta__(self, x0, y0, beta1):

		maxIter = 2000
		iter = 0
		TolImgGap = 1e-6


		l0 = self.lk
		epsilon = self.VarModel.P @ x0 + self.VarModel.Q @ y0 - self.VarModel.c


		theta = np.hstack((self.VarModel.P.T @ epsilon, self.VarModel.Q.T @ epsilon))
		PQtheta = self.VarModel.P @ ( (self.VarModel.P.T) @ epsilon) + \
				  self.VarModel.Q @ ( (self.VarModel.Q.T) @ epsilon)

		
		normTheta = np.linalg.norm(theta)
		tilde = 1 / (normTheta**2)
		l0DotPQtheta = np.dot(l0, PQtheta)

		xk = x0
		yk = y0		

		beta2 = beta1
		ImgErr = np.inf

		while (ImgErr >= (TolImgGap / 2) and (iter <= maxIter)):			

			lBeta = l0 + beta2 * epsilon

			xk1, yk1 = self.VarModel.primalStep(xk, yk, lBeta, beta1)


			rk1 = self.VarModel.P @ xk1 + self.VarModel.Q @ yk1 - self.VarModel.c
			resk1 = np.dot(epsilon, rk1)


			ErrRes = np.abs(resk1) * ( tilde * (self.Lphi * normTheta + np.abs(l0DotPQtheta)) + beta2)

			betaLower = 0
			betaUpper = np.inf

			while (ErrRes >= (TolImgGap / 2)):

				#print("FindingBetaIter")

				if (resk1 < 0):
					betaUpper = beta2
				else:
					betaLower = beta2

				if (not(betaUpper == np.inf)):
					beta2 = (betaUpper + betaLower)/2
				else:
					beta2 *= 1.1

				lBeta = l0 + beta2 * epsilon

				xk1, yk1 = self.VarModel.primalStep(xk, yk, lBeta, beta1)

				rk1 = self.VarModel.P @ xk1 + self.VarModel.Q @ yk1 - self.VarModel.c
				resk1 = np.dot(epsilon, rk1)

				#################################
				#compute residue error

				ErrRes = np.abs(resk1) * ( tilde * (self.Lphi * normTheta + np.abs(l0DotPQtheta)) + beta2)


			################################
			#compute image residue

			Lk = self.__AugmLag__(xk, yk, lBeta, beta1)
			Rk = Lk + (1/(2 * beta1)) * np.linalg.norm(lBeta)**2
			Diamk = (2/self.minSigma) * (np.sqrt(Rk) * np.sqrt(2 / beta1) + np.linalg.norm(self.VarModel.c) + (1/beta1) * np.linalg.norm(lBeta))

			DualResk = beta1 * self.VarModel.P.T @ (self.VarModel.Q @ (yk1 - yk))
			DualResNormk = np.linalg.norm(DualResk)

			ImgErr = max([Diamk, 1]) * DualResNormk
			#ImgErr = Diamk * DualResNormk

			#print(f"ImgErr: {ImgErr}")
			#print(f"DualResNorm: {DualResNormk}")

			xk = xk1
			yk = yk1

			iter += 1

		print(f"iter Backtrack: {iter}")
		print(f"DualRes: {DualResNormk}")
		print(f"Diamk: {Diamk}")

		return xk, yk, beta2, DualResNormk		


		
