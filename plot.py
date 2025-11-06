
from matplotlib import pyplot as plt
import numpy as np

ns_to_ms = 10**-6

ns = np.array([128, 512, 1024, 2048])

v01_d2h = np.array([26133881, 31469115, 21171552, 27821482])
v01_h2d = np.array([800,1408,1600,2272])
v01_cmp = np.array([4567771,17562057,34889044,70476143])
v01_mem = v01_d2h + v01_h2d

v02_d2h = np.array([19238409, 25758424, 15814352, 16040690])
v02_cmpv = np.array([17524794, 18799623, 18194658, 18655140])
v02_ini = np.array([179521, 217122, 179457, 179457])
v02_set = np.array([168257, 271969, 436995, 752133])
v02_h2d = np.array([960, 1376, 1856, 2208])

v02_mem = v02_d2h + v02_h2d
v02_cmp = v02_ini + v02_set + v02_cmpv

# v02_d2h = np.array()

# plt.plot(ns, v01_d2h, label="Device to Host")
# plt.plot(ns, v01_h2d, label="Host to Device")
plt.plot(ns, v01_cmp * ns_to_ms, label="v01 Compute")
plt.plot(ns, v01_mem * ns_to_ms, label="v01 Memory Transfers")
plt.plot(ns, v02_cmp * ns_to_ms, label="v02 Compute")
plt.plot(ns, v02_mem * ns_to_ms, label="v02 Memory Transfers")
plt.legend()

plt.xlabel("# Voronoi Centers")
plt.ylabel("Time [ms]")
plt.grid(True)

plt.savefig("plot.png")

