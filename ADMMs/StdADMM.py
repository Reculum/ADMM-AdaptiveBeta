
from Solver import *



class StdADMMClass(SolverClass):


	def __init__(self, varMod, x0, y0, l0, beta0):

		super().__init__(varMod, x0, y0, l0, beta0)
		self.IterationStep = self.__StdStep__


	def __StdStep__(self, x, y, l, beta):

		x_k_1, y_k_1 = self.VarModel.primalStep(x, y, l, beta)
		l_k_1 = self.VarModel.dualStep(x_k_1, y_k_1, l, beta)
		return (x_k_1, y_k_1, l_k_1, beta)