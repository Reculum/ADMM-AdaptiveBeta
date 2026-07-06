
from models import TVL2_1D
from ADMMs import StdADMM
from signal_class import *

np.random.seed(24102001)


n = 1024

PwSignal = signal(n)
RndSignal = signal(n)
sigma = 0

PwSignal.generate_cartoon_sign(2, 100)
RndSignal.generate_GG_realization(0, sigma, 2)

diag = 0.5 * np.ones(shape=(n,))
offDiag = 0.25 * np.ones(shape=(n - 1, ))
A = np.diag(diag, 0) + np.diag(offDiag, 1) + np.diag(offDiag, -1) 




xTrue = PwSignal.get_image()
b = (A @ xTrue) + RndSignal.get_image()
mu = 0.5

xk = np.random.randn(n)
yk = np.random.randn(n-1)
betak = 0.1
lk = np.random.randn(n-1)

VarModel = TVL2_1D.TVL2_1DClass(A, b, mu)
StdAdmmSolver = StdADMM.StdADMMClass(VarModel, xk, yk, lk, betak)


for iter in range(0, 100):

	xk_1, yk_1, lk_1, betak_1 = StdAdmmSolver.IterationStep(xk, yk, lk, betak)

	print(f"primal residue: {np.linalg.norm(VarModel.D @ xk_1 - yk_1)}" )
	print(f"dual residue: {np.linalg.norm( betak_1 * VarModel.D.T @ (yk - yk_1))}")

	xk = xk_1
	yk = yk_1
	lk = lk_1
	betak = betak_1

plt.figure(1)
plt.plot(b)

plt.figure(2)
plt.plot(xk)
plt.show()


