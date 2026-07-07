
import numpy as np
import scipy.linalg as la
from typing import Callable, Tuple


class VariationalModelClass():

    fidelity : Callable[[np.ndarray], float]
    regularizer : Callable[[np.ndarray], float]
    mu : float

    primalStep : Callable[[np.ndarray, np.ndarray, np.ndarray, float], Tuple[np.ndarray, np.ndarray]]
    dualStep : Callable[[np.ndarray, np.ndarray, np.ndarray, float], np.ndarray]


    def setFidelity(self, f):
        self.fidelity = f
    
    def setRegularizer(self, f):
        self.regularizer = f

    def setMu(self, mu):
        self.mu = mu

    def setPrimalStep(self, f):
        self.primalStep = f
    
    def setDualStep(self, f):
        self.dualStep = f
    
    def __eval__(self, x):        
        return self.regularizer(x) + (self.mu / 2) * self.fidelity(x)
        
    