
from models import TVL2_1D
from ADMMs import StdADMM
from signal_class import *

import matplotlib.pyplot as plt

np.random.seed(24102001)


PwSignal = signal(1024)
PwSignal.generate_cartoon_sign(10)


plt.plt(PwSignal.get_image())
plt.show()


