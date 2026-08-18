import numpy as np
from LanzaModels import TVL1_1D
from ADMMsRustici import MyBackTrackingSolver
from signalClass import *
import time



np.random.seed(24102000)
n = 128

###########################################


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
A[n-1][n-2] = a - c
A[n-1][n-1] = b + 2 * c

#end blur matrix construction


################################################


#begin signal construction

PwSignal = signal(n)
RndSignal = signal(n)
sigma = 0.01

PwSignal.generate_cartoon_sign(2, 15)
RndSignal.generate_GG_realization(0, sigma, 1)

xTrue = PwSignal.get_image()
xCorrupted = (A @ xTrue) + RndSignal.get_image()

#end signal construction

###############################################


mu = 0.5
VarModel = TVL1_1D.TVL1_1DClass(A, xCorrupted, mu)

Lfid = mu * np.max(np.linalg.svdvals(A)) * np.sqrt(n)
Lreg = np.sqrt(n)
Lphi = np.sqrt(Lfid**2 + Lreg**2)

##################################################


#begin solver construction
np.random.seed(24102002)

xk = np.random.randn(n,)
yk = np.random.randn(n,)
betak = 1
lk = np.zeros(n)

x0 = xk.copy()
y0 = yk.copy()

MySolver = MyBackTrackingSolver.MyBacktrackingSolverClass(VarModel, xk, yk, lk, betak, Lphi)

#end solver construction

#############################################################


iters = 15

XsolutionHistory = np.zeros(shape=(iters, n))
YsolutionHistory = np.zeros(shape=(iters, n))

lambdaHistory = np.zeros(shape=(iters, n))

betaHistory = np.zeros(shape=(iters,))

PrimalResidueHistory = np.zeros(shape=(iters,))
DualResidueHistory = np.zeros(shape=(iters,))
ImgHistory = np.zeros(shape=(iters,))

CpuTimes = np.zeros(shape=(iters,))


###############################################################



timer = 0

for iter in range(0, iters):

    print(f"{iter + 1} / {iters}")

    sTime = time.perf_counter_ns()

    xk_1, yk_1, lk_1, betak_1, dualResk = MySolver.CallMyIterationStep(xk, yk, lk, betak)

    eTime = time.perf_counter_ns()

    timer += ( (eTime - sTime) / 1e9 )

    
    primalResidue = np.linalg.norm(VarModel.P @ xk_1 + VarModel.Q @ yk_1 - VarModel.c)
    print(f"primalRes: {primalResidue}")

    XsolutionHistory[iter, :] = xk_1
    YsolutionHistory[iter, :] = yk_1
    lambdaHistory[iter, :] = lk_1
    betaHistory[iter] = betak_1

    PrimalResidueHistory[iter] = primalResidue
    DualResidueHistory[iter] = dualResk
    ImgHistory[iter] = VarModel(xk_1, yk_1)
    CpuTimes[iter] = timer

    xk = xk_1
    yk = yk_1
    lk = lk_1
    betak = betak_1

    if (max(primalResidue, dualResk) <= 1e-9):
        print(iter)
        break


##########################################################################################



np.savez_compressed(
    "./MyADMMTVL1-Laplace.npz",
    Xs = XsolutionHistory,
    Ys = YsolutionHistory,
    Ls = lambdaHistory,
    Betas = betaHistory,

    PrimalRes = PrimalResidueHistory,
    DualRes = DualResidueHistory,
	IMGs = ImgHistory,
    CpuTimes = CpuTimes,

    xTrue = xTrue,
    xCorrupted = xCorrupted,
	x0 = x0,
	y0 = y0
)


	

