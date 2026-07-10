from models import TVL2_1D
from ADMMs import SpectralSolver
from signal_class import *

np.random.seed(24102001)


n = 1024

PwSignal = signal(n)
RndSignal = signal(n)
sigma = 0.01

PwSignal.generate_cartoon_sign(2, 100)
RndSignal.generate_GG_realization(0, sigma, 2)

diag = 0.5 * np.ones(shape=(n,))
offDiag = 0.25 * np.ones(shape=(n - 1, ))
A = np.diag(diag, 0) + np.diag(offDiag, 1) + np.diag(offDiag, -1)


xTrue = PwSignal.get_image()
b = (A @ xTrue) + RndSignal.get_image()
mu = 2

VarModel = TVL2_1D.TVL2_1DClass(A, b, mu)

xk = np.copy(b)
yk = VarModel.D @ xk
betak = 1
lk = np.zeros(n - 1)

SpectralAdmmSolver = SpectralSolver.SpectralSolverClass(VarModel, xk, yk, lk, betak)

iters = 200


for iter in range(0, iters):

	xk_1, yk_1, lk_1, betak_1 = SpectralAdmmSolver.CallIterationStep(xk, yk, lk, betak)

	print(f"iter: {iter}/{iters}")
	print(f"primal residue: {np.linalg.norm(VarModel.D @ xk_1 - yk_1)}" )
	print(f"dual residue: {np.linalg.norm( betak * VarModel.D.T @ (yk - yk_1))}")

	xk = xk_1
	yk = yk_1
	lk = lk_1
	betak = betak_1

plt.figure(1)
plt.title("Corrupted")
plt.grid()
plt.plot(b)

plt.figure(2)
plt.title("Reconstruction")
plt.grid()
plt.plot(xk)

plt.figure(3)
plt.title("Original")
plt.grid()
plt.plot(xTrue)


plt.show()
