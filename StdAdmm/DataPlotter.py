
import numpy as np
import matplotlib.pyplot as plt
import sys


def Plot(npzPath):

    data = np.load(npzPath)

    Xs = data["Xs"]
    Betas = data["Betas"]

    PrimalRes = data["PrimalRes"]
    DualRes = data["DualRes"]

    CpuTimes = data["CpuTimes"]
    ConvergenceDistance = data["ConvergenceDistance"]

    xTrue = data["xTrue"]
    xCorrupted = data["xCorrupted"]  

    

    iters, n = np.shape(Xs)

    xSol = Xs[iters - 1, :]
    x0 = Xs[0, :]

    plt.figure(1)
    plt.grid()
    plt.title("Corrupted vs Reconstruction vs True")
    plt.plot(xTrue)
    plt.plot(xCorrupted)
    plt.plot(xSol)
    plt.legend(["True", "Corrupted", "Reconstruction"])
    plt.xlabel("compontents")
    

    plt.figure(2)
    plt.grid()
    plt.title("Metrics vs Iters")
    plt.loglog(ConvergenceDistance)
    plt.loglog(PrimalRes)
    plt.loglog(DualRes)
    plt.xlabel("iters")
    plt.legend(["xSolDistance", "PrimalRes", "DualRes"])

    plt.figure(3)
    plt.grid()
    plt.title("Metrics vs Cpu Time")
    plt.semilogy(CpuTimes, ConvergenceDistance)
    plt.semilogy(CpuTimes, PrimalRes)
    plt.semilogy(CpuTimes, DualRes)
    plt.xlabel("Runtime (s)")
    plt.legend(["xSolDistance", "PrimalRes", "DualRes"])

    plt.show()


Plot(sys.argv[1])


