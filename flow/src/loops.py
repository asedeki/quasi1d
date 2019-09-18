import numpy as np
import calculbulles as cb
import pstats
import cProfile


class Loops:
    def __init__(self, parametres):
        self.param = parametres

    def resset(self):
        Np = self.param["Np"]
        self.Cooper = np.zeros((Np, Np, Np), float)
        self.Peierls = np.zeros((Np, Np, Np), float)
        self.Peierls_susc = np.zeros((Np, 2), float)

    def calculer(self, T, l):
        # print("in bubble")
        self.resset()
        cb.valeursbulles(l, T, self.param, self.Cooper,
                         self.Peierls, self.Peierls_susc)
        # print(f"bulle faite pout {l}")


def test():
    from math import exp
    r = {"tp": 200, "tp2": 20, "Ef": 3000, "Np": 64, "g1": 0.2}
    bulle = Loops(r)

    T = 1
    l = 0.0

    EF = r['Ef']*exp(-l)
    n = r["Np"]//2
    bulle.calculer(T, l)
    return bulle


if __name__ == "__main__":
    import time
    t1 = time.time()
    cProfile.runctx("test()", globals(), locals(), "Profile.prof")

    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats()
    #bulle = test()
    # profile.run(
    #     'cb.valeursLoops(l, T, bulle.param, bulle.IC, bulle.IP, bulle.IPsusc)')
    # np.savez("test", C=bulle.IC, P=bulle.IP, P2=bulle.IPsusc)
    #s = np.load("test.npz")
    #IC = s["C"]
    # II = bulle.IP
    # n = bulle.IP.shape[0]
    # for i in range(-n, n):
    #     print(f"{i}\t{II[i, 0, 0]}")
    # print(f"sum = {np.sum(II[:, 0, 0])}")
    # print(f"temps exec : {time.time()-t1}")
    #print(f" IC0={bulle.IC[:, 0, 0]}")
    # Np = r['Np']
    # for i in range(-n, n):
    #     for j in range(-n, n):
    #         for k in range(-n, n):
    #             e = abs(IC[i, j, k]-II[i, j, k])
    #             if e > 1e-10:
    #                 print(f"{i}, {j}, {k}")
    #                 print(
    #                     f"{IC[i, j, k]-II[i, j, k]} == {II[i, j, k]}=={II[-i, -j, -k]}")

    # l1 = II[i+n, j+n, k+n]  # -bulle.IC[j, i, k]
    # l2 = IC[i, j, k]
    # if abs(l1-l2) > 1e-19:
    #     print(f"{i , j , k}=={l1}  {l2}  -> {abs(l1-l2)}")
