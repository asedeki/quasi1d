import numpy as np
from interaction import Interaction


class bulle:
    def __init__(self, N):
        self.IP = np.ones((N, N, N), float)*1.0
        self.IC = np.ones((N, N, N), float)*1.0


N = 2
g = Interaction(N)
g.initialisation(0.6, 0.6, 0.6)
print(g.g1)
print(g.g2)
print(g.g3)
input("tata")
b = bulle(N)
g.equations_rg(b)
print(g.dg2)
print(g.dg)
