import numpy as np
from cythonfiles import calculbulles as cb
from math import tanh, exp


class Bulles:
    def __init__(self, parametres):
        self.param = parametres

    def resset(self):
        Np = self.param["Np"]
        self.IC = np.zeros((Np, Np, Np), float)
        self.IP = np.zeros((Np, Np, Np), float)
        self.IPsusc = np.zeros((Np, 2), float)

    def calculer(self, T, l):
        self.resset()
        cb.valeursbulles(l, T, self.param, self.IC, self.IP, self.IPsusc)


if __name__ == "__main__":
    r = {"tp": 0, "tp2": 0, "Ef": 3000, "Np": 2}
    bulle = Bulles(r)

    T = 100.0
    l = 10.0
    EF = r['Ef']*exp(-l)
    n = r["Np"]//2
    bulle.calculer(T, l)
    #np.savez("test", C=bulle.IC, P=bulle.IP, P2=bulle.IPsusc)
    # s = np.load("test.npz")
    # IC = s["P"]
    # II = bulle.IP
    for i in range(-n, n):
        for j in range(-n, n):
            for k in range(-n, n):
                print(
                    f"{bulle.IP[i,j,k]*r['Np']} , {bulle.IC[i,j,k]*r['Np']}, {tanh(EF/2.0/T)}")

                # l1 = II[i+n, j+n, k+n]  # -bulle.IC[j, i, k]
                # l2 = IC[i, j, k]
                # if abs(l1-l2) > 1e-19:
                #     print(f"{i , j , k}=={l1}  {l2}  -> {abs(l1-l2)}")
