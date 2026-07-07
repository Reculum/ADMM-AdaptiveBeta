
from models.VariationalModel import *
from typing import Callable, Tuple


class SolverClass():

	VarModel : VariationalModelClass
	IterationStep : Callable[[np.ndarray, np.ndarray, np.ndarray, float], Tuple[np.ndarray, np.ndarray, np.ndarray, float]]

	xk : np.ndarray
	yk : np.ndarray
	lk : np.ndarray
	betak :float


	def __init__(self, varMod : VariationalModelClass, x0 : np.ndarray, y0 : np.ndarray, l0 : np.ndarray, beta0 : float):

		self.betak = beta0
		self.lk = l0
		self.xk = x0
		self.yk = y0
		
		self.VarModel = varMod

	
	def setIterationStep( self, f = (lambda x, y, l, beta: (None, None, None, 0)) ):
		self.IterationStep = f
