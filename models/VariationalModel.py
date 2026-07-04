
from __init__ import *


class VariationalModelClass():

    fidelity = lambda x: None
    regularizer = lambda x: None
    mu = None

    primalStep = lambda x: None
    dualStep = lambda x: None        

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
        return self.regularizer(x) + (self.mu / 2) * self.fidelity
        
    