
from models.VariationalModel import *


class SolverClass():

	VarModel : VariationalModelClass
	IterationStep : lambda x, y, l, beta: None

	xk = None
	yk = None
	lk = None
	betak = None


	def __init__(self, varMod : VariationalModelClass, x0 : np.ndarray, y0 : np.ndarray, l0 : np.ndarray, beta0 : float):

		self.betak = beta0
		self.lk = l0
		self.xk = x0
		self.yk = y0
		
		self.VarModel = varMod

	
	def setIterationStep( self, f = (lambda x, y, l, beta: None) ):
		self.IterationStep = f
