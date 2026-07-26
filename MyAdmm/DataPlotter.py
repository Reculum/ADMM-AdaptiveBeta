
import numpy as np
import matplotlib.pyplot as plt
import sys


def Plot(npzPath):

    data = np.load(npzPath)

    Xs = data["Xs"]
    CpuTimes = data["CpuTimes"]
    Betas = data["Betas"]
    PrimalRes = data["PrimalRes"]
    DualRes = data["DualRes"]

    iters, n = np.shape(Xs)

    xSol = Xs[iters - 1, :]
    x0 = Xs[0, :]

    plt.figure(1)
    plt.plot(x0)

    plt.figure(2)
    plt.plot(xSol)

    plt.show()


Plot(sys.argv[1])


