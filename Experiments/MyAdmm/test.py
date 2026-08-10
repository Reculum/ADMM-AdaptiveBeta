import numpy as np
from LanzaModels import TVL1_1D
from ADMMsRustici import MyBackTrackingSolver
from signalClass import *

np.random.seed(24102001)


n = 1024

#blur matrix construction

a = 0.25
b = 0.5
c = 0.25

diagB = b * np.ones(shape=(n,))
offDiagA = a * np.ones(shape=(n-1,))
offDiagC = c * np.ones(shape=(n-1,))
A = np.diag(diagB, 0) + np.diag(offDiagC, 1) + np.diag(offDiagA, -1)

#apply anti-reflexive BCs

A[0][0] = 2 * a + b
A[0][1] = c - a
A[n-2][n-1] = a - c
A[n-1][n-1] = b + 2 * c

#end blur matrix construction


#begin signal construction

PwSignal = signal(n)
RndSignal = signal(n)
sigma = 0.01

PwSignal.generate_cartoon_sign(2, 100)
RndSignal.generate_GG_realization(0, sigma, 2)

xTrue = PwSignal.get_image()
xCorrupted = (A @ xTrue) + RndSignal.get_image()

#end signal construction

#begin model construction

mu = 0.5
VarModel = TVL1_1D.TVL1_1DClass(A, xCorrupted, mu)

Lfid = mu * np.max(np.linalg.svdvals(A)) * np.sqrt(n)
Lreg = np.sqrt(n)
Lphi = np.sqrt(Lfid**2 + Lreg**2)

#end model construction


#begin solver construction

xk = np.copy(xCorrupted)
yk = np.random.randn(n,)
betak = 1
lk = np.zeros(n)

MySolver = MyBackTrackingSolver.MyBacktrackingSolverClass(VarModel, xk, yk, lk, betak, Lphi)

#end solver construction

#resolution

iters = 2

for iter in range(1, iters + 1):

	xk_1, yk_1, lk_1, betak_1 = MySolver.CallIterationStep(xk, yk, lk, betak)

	print(f"iter: {iter}/{iters}")
	print(f"betak: {betak_1}")
	print(f"primal residue: {np.linalg.norm(VarModel.P @ xk_1 + VarModel.Q @ yk_1 - VarModel.c)}" )

	xk = xk_1
	yk = yk_1
	lk = lk_1
	betak = betak_1


plt.figure(1)
plt.title("Corrupted")
plt.grid()
plt.plot(xCorrupted)

plt.figure(2)
plt.title("Reconstruction")
plt.grid()
plt.plot(xk)

plt.figure(3)
plt.title("Original")
plt.grid()
plt.plot(xTrue)


plt.show()

	

