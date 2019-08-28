import numpy as np
from math import pi
import cinteraction as cint
import sys
import concurrent.futures as concfut


# todo reflechir a wrap(), %N ne donne que des valeurs positives


class Interaction:
    def __init__(self, Np, g1=0.0, g2=0.0, g3=0.0):
        self.g1 = np.ones((Np, Np, Np), float) * g1
        self.g2 = np.ones((Np, Np, Np), float) * g2
        self.g3 = np.ones((Np, Np, Np), float) * g3
        self.Np = Np

    def initialisation(self, g1, g2, g3, g1_perp=0, g2_perp=0, g3_perp=0):
        Np = self.Np
        # for k1 in range(-N2, N2):
        #     for k2 in range(-N2, N2):
        #         for k3 in range(-N2, N2):
        #             i = (k1, k2, k3)
        #             self.g1[i] = g1
        #             self.g2[i] = g2
        #             self.g3[i] = g3
        self.g1 = np.ones((Np, Np, Np), float) * g1
        self.g2 = np.ones((Np, Np, Np), float) * g2
        self.g3 = np.ones((Np, Np, Np), float) * g3

    def inipack(self):
        Ng = self.Np**3
        y = np.zeros(3*Ng)
        y[:Ng] = self.g1.reshape((Ng,))
        y[Ng:2*Ng] = self.g2.reshape((Ng,))
        y[2*Ng:3*Ng] = self.g3.reshape((Ng,))
        return y

    def pack(self, dg1, dg2, dg3):
        Ng = self.Np**3
        y = np.zeros(3*Ng)
        y[:Ng] = dg1.reshape((Ng,))
        y[Ng:2*Ng] = dg2.reshape((Ng,))
        y[2*Ng:3*Ng] = dg3.reshape((Ng,))
        return y

    def unpack(self, y):
        Ng = self.Np**3
        self.g1 = y[:Ng].reshape((self.Np, self.Np, self.Np))
        self.g2 = y[Ng:2*Ng].reshape((self.Np,
                                      self.Np, self.Np))
        self.g3 = y[2*Ng:3 *
                    Ng].reshape((self.Np, self.Np, self.Np))

    def equations_rg_seq(self, bulle):
        Np = self.Np
        dg1 = np.zeros((Np, Np, Np), float)
        dg2 = np.zeros((Np, Np, Np), float)
        dg3 = np.zeros((Np, Np, Np), float)

        N2 = Np // 2
        for k1 in range(-N2, N2):
            for k2 in range(-N2, N2):
                qc = (k1+k2) % Np
                for k3 in range(-N2, N2):
                    qp = (k3-k2) % Np
                    qpp = (k1-k3) % Np
                    k4 = (k1+k2-k3) % Np
                    i = (k1, k2, k3)

                    for kp in range(-N2, N2):
                        IP = bulle.IP[k2, kp, qp]
                        IP2 = bulle.IP[k2, kp, -qp]
                        IC = bulle.IC[k1, kp, qc]
                        IPP = bulle.IP[k3, kp, qpp]

                        m1 = (k1, k2, kp)
                        m2 = (kp, (qc-kp) % Np, k3)
                        m3 = (k1, (kp-qp) % Np, kp)
                        m4 = (kp, k2, k3)
                        m5 = (k1, kp, (kp+qp) % Np)
                        m6 = (k2, (kp+qp) % Np, k3)
                        m7 = (k2, (kp+qp) % Np, kp)
                        m8 = ((kp+qp) % Np, k2, k3)
                        m9 = (k2, (kp+qpp) % Np, kp)
                        m10 = (k2, (kp+qpp) % Np, k4)
                        m11 = (k1, kp, (kp+qpp) % Np)
                        m12 = (k1, kp, k3)

                        dg1[i] += -0.5*((self.g2[m1]*self.g1[m2]+self.g1[m1]*self.g2[m2])*IC
                                        - (self.g2[m3]*self.g1[m4]+self.g1[m3]*self.g2[m4]
                                           - 2*self.g1[m3]*self.g1[m4])*IP2)
                        dg1[i] += 0.5*(self.g3[m5]*self.g3[m6] + self.g3[m7]*self.g3[k1, kp, k4]
                                       - 2.0*self.g3[m5]*self.g3[m8])*IP

                        dg2[i] += 0.5*(-(self.g2[m1]*self.g2[m2]+self.g1[m1]*self.g1[m2])*IC
                                       + self.g2[m3]*self.g2[m4] * IP2)
                        dg2[i] += 0.5*self.g3[k1, kp, k4]*self.g3[m6]*IP

                        dg3[i] += 0.5*(self.g3[m5]*self.g2[m7] + self.g3[k1, kp, k4]*self.g1[m7] +
                                       self.g2[m5]*self.g3[m7] + self.g1[m5]*self.g3[m6] -
                                       2*self.g1[m7]*self.g3[m5] - 2*self.g3[m7]*self.g1[m5])*IP
                        dg3[i] += 0.5*(self.g3[m12]*self.g2[m9] +
                                       self.g3[m10]*self.g2[m11])*IPP

        return self.pack(dg1, dg2, dg3)

    def equations_rg_seq(self, bulle):
        Np = self.Np
        dg1 = np.zeros((Np, Np, Np), float)
        dg2 = np.zeros((Np, Np, Np), float)
        dg3 = np.zeros((Np, Np, Np), float)
        cint.equations_rg(Np, self.g1, self.g2,
                          self.g3, bulle.IP, bulle.IC,
                          dg1, dg2, dg3)
        #print("dg=", self.dg2[0, 0, Np//2])
        # sys.exit(1)
        #self.dg = self.pack()
        return self.pack(dg1, dg2, dg3)

    def rg_rows(self, vk1):
        Np = self.Np
        result = []
        for k1 in vk1:
            dg1 = np.zeros((Np, Np), float)
            dg2 = np.zeros((Np, Np), float)
            dg3 = np.zeros((Np, Np), float)
            cint.equations_rgcol(Np, k1, self.g1, self.g2,
                                 self.g3, self.bulle.IP, self.bulle.IC,
                                 dg1, dg2, dg3)
            result.append([k1, dg1, dg2, dg3])
        return result

    def rg_row(self, k1):
        Np = self.Np
        dg1 = np.zeros((Np, Np), float)
        dg2 = np.zeros((Np, Np), float)
        dg3 = np.zeros((Np, Np), float)
        cint.equations_rgcol(Np, k1, self.g1, self.g2,
                             self.g3, self.bulle.IP, self.bulle.IC,
                             dg1, dg2, dg3)

        return k1, dg1, dg2, dg3

    def equations_rg(self, bulle):
        Np = self.Np
        dg1 = np.zeros((Np, Np, Np), float)
        dg2 = np.zeros((Np, Np, Np), float)
        dg3 = np.zeros((Np, Np, Np), float)
        self.bulle = bulle
        kvec = np.arange(Np)
        # for k1 in kvec:
        #     self.rg_row(k1)
        with concfut.ThreadPoolExecutor(max_workers=4) as executor:
            s = np.array_split(kvec, 4)
            jobs = [executor.submit(self.rg_rows, kk) for kk in s]

        for gg in jobs:
            gt = gg.result()
            for g in gt:
                dg1[g[0], :, :] = g[1]
                dg2[g[0], :, :] = g[2]
                dg3[g[0], :, :] = g[3]

        for k1 in range(-Np//2, Np//2):
            for k2 in range(1, Np//2):
                for k3 in range(-Np//2, Np//2):
                    dg1[k1, k2, k3] = dg1[-k1, -k2, -k3]
                    dg2[k1, k2, k3] = dg2[-k1, -k2, -k3]
                    dg3[k1, k2, k3] = dg3[-k1, -k2, -k3]

        return self.pack(dg1, dg2, dg3)
