
import numpy as np
import matplotlib.pyplot as plt
import sys


def Plot(npzPath):

    data = np.load(npzPath)

    Xs = data["Xs"]
    Betas = data["Betas"]

    PrimalRes = data["PrimalRes"]
    DualRes = data["DualRes"]
    IMGs = data["IMGs"]

    CpuTimes = data["CpuTimes"]

    xTrue = data["xTrue"]
    xCorrupted = data["xCorrupted"]
    x0 = data["x0"]
    y0 = data["y0"]     

    iters, n = np.shape(Xs)
    cut = int(iters * 0.9)
    
    xReconstr = Xs[-1, :]
    ImgXreconstr = IMGs[-1]

    DomainDistance = np.linalg.norm(Xs - xReconstr, axis=1)
    ImgDistance = np.abs(IMGs - ImgXreconstr)

    DomainRatios = DomainDistance[1:] / DomainDistance[:-1]
    ImageRatios = ImgDistance[1:] / ImgDistance[:-1]

    plt.figure(0)

    plt.subplot(1,2,1)
    plt.title("x0")
    plt.plot(x0)
    plt.xlabel("components")
    plt.grid() 

    plt.subplot(1,2,2)
    plt.title("y0")
    plt.plot(y0)
    plt.xlabel("components")
    plt.grid()

    plt.savefig("startingPoints.pdf", format="pdf", bbox_inches="tight")


    ###########################################

    plt.figure(1)
    plt.grid()
    plt.title("Corrupted vs Reconstruction vs True")
    plt.plot(xTrue)
    plt.plot(xCorrupted)
    plt.plot(xReconstr)
    plt.legend(["True", "Corrupted", "Reconstruction"])
    plt.xlabel("components")
    plt.savefig("reconstr.pdf", format="pdf", bbox_inches="tight")

    ###########################################

    plt.figure(2)
    plt.grid()
    plt.title("Residuals vs Iters")
    plt.semilogy(PrimalRes)
    plt.semilogy(DualRes)
    plt.xlabel("iters")
    plt.legend(["PrimalRes", "DualRes"])
    plt.savefig("residuals.pdf", format="pdf", bbox_inches="tight")

    ###########################################

    plt.figure(3)
    plt.grid()
    plt.title("Pen. Param. vs Iters")
    plt.plot(Betas)
    plt.xlabel("iters")
    plt.legend(["Beta"])
    plt.savefig("betas.pdf", format="pdf", bbox_inches="tight")

    ############################################

    plt.figure(4)
    plt.grid()
    plt.title("Domain: Rate Of Convergence vs Iters")
    plt.plot(DomainRatios[:cut])
    plt.xlabel("iters")
    plt.legend(["Dom. Rate Of Conv."])
    plt.savefig("dom-roc.pdf", format="pdf", bbox_inches="tight")

    ############################################
    
    plt.figure(5)
    plt.grid()
    plt.title("Co-Domain: Rate Of Convergence vs Iters")
    plt.plot(ImageRatios[:cut])
    plt.xlabel("iters")
    plt.legend(["Co-Dom. Rate Of Conv."])
    plt.savefig("co-dom-roc.pdf", format="pdf", bbox_inches="tight")

    ############################################
    
    plt.figure(6)
    plt.grid()
    plt.title("Metrics vs Iters")
    plt.semilogy(DomainDistance[:cut])
    plt.semilogy(ImgDistance[:cut])
    plt.xlabel("iters")
    plt.legend(["Domain Distance", "Image Distance"])
    plt.savefig("metr-vs-iters.pdf", format="pdf", bbox_inches="tight")
    
    ############################################

    plt.figure(7)
    plt.grid()
    plt.title("Metrics vs Cpu Time")
    plt.semilogy(CpuTimes[:cut], DomainDistance[:cut])
    plt.semilogy(CpuTimes[:cut], ImgDistance[:cut])
    plt.xlabel("Runtime (s)")
    plt.legend(["Domain Distance", "Image Distance"])
    plt.savefig("metr-vs-cputime.pdf", format="pdf", bbox_inches="tight")


    #plt.show()


Plot(sys.argv[1])


