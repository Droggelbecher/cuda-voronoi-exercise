
from matplotlib import pyplot as plt
import numpy as np

ns = np.array([128, 512, 1024, 2048])

v01_d2h = np.array([26133881, 31469115, 21171552, 27821482])
v01_h2d = np.array([800,1408,1600,2272])
v01_cmp = np.array([4567771,17562057,34889044,70476143])

# v02_d2h = np.array()

plt.plot(ns, v01_d2h, label="Device to Host")
plt.plot(ns, v01_h2d, label="Host to Device")
plt.plot(ns, v01_cmp, label="Compute")
plt.legend()

plt.xlabel = "Voronoi Centers"
plt.ylabel = "Time [ns]"
plt.grid(True)

plt.savefig("plot.png")

