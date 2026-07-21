from models import TVL2_1D
from ADMMs import MyLipschitzSolver
from signal_class import *

np.random.seed(24102001)


n = 1024

#begin blur matrix construction
a = 0.25
b = 0.5
c = 0.25

diag = b * np.ones(shape=(n,))
offDiag = a * np.ones(shape=(n - 1, ))
A = np.diag(diag, 0) + np.diag(offDiag, 1) + np.diag(offDiag, -1)

#apply anti-reflexive BCs
A[0][0] = 2 * a + b
A[0][1] = c - a
A[n-2][n-1] = a - c
A[n-1][n-1] = b + 2 * c

#end of blur matrix const


#begin signal construction
PwSignal = signal(n)
RndSignal = signal(n)
sigma = 0.01

PwSignal.generate_cartoon_sign(2, 100)
RndSignal.generate_GG_realization(0, sigma, 2)

xTrue = PwSignal.get_image()
b = (A @ xTrue) + RndSignal.get_image()
#end of signal construction

#begin model construction

mu = 2
VarModel = TVL2_1D.TVL2_1DClass(A, b, mu)

xk = np.copy(b)
yk = np.random.randn(n,)
betak = 1
lk = np.zeros(shape=(n,))

def dPhi(z):
	x, y = z[:VarModel.n], z[VarModel.n:]
	dx = mu * (A @ x - b) @ A
	dy = np.sign(y)
	return np.hstack([dx, dy])

MySolver = MyLipschitzSolver.My_Lipschitz_SolverClass(VarModel, xk, yk, lk, betak, 1, 10, dPhi)

iters = 10

for iter in range(0, iters):

	xk_1, yk_1, lk_1, betak_1 = MySolver.CallIterationStep(xk, yk, lk, betak)

	print(f"iter: {iter}/{iters}")
	print(f"betak: {betak_1}")
	print(f"primal residue: {np.linalg.norm(VarModel.D @ xk_1 - yk_1)}" )
	print(f"dual residue: {
						   np.abs(
							   (VarModel.mu / 2) * (VarModel.fidelity(xk_1) - VarModel.fidelity(xk) ) + \
							   VarModel.regularizer(yk_1) - VarModel.regularizer(yk)							
								)
						  }"
		)

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