import numpy as np
from math import pi

# todo reflechir a wrap(), %N ne donne que des valeurs positives


class Interaction:
    def __init__(self, N_patche):
        self.g1 = np.zeros((N_patche, N_patche, N_patche), float)
        self.g2 = np.zeros((N_patche, N_patche, N_patche), float)
        self.g3 = np.zeros((N_patche, N_patche, N_patche), float)
        self.N_patche = N_patche

    def initialisation(self, g1, g2, g3, g1_perp=0, g2_perp=0, g3_perp=0):
        Np = self.N_patche
        for k1 in range(-Np//2, Np//2):
            for k2 in range(-Np//2, Np//2):
                for k3 in range(-Np//2, Np//2):
                    i = (k1, k2, k3)
                    self.g1[i] = g1
                    self.g2[i] = g2
                    self.g3[i] = g3

    def inipack(self):
        Ng = self.N_patche**3
        y = np.zeros(3*Ng)
        y[:Ng] = self.g1.reshape((Ng,))
        y[Ng:2*Ng] = self.g2.reshape((Ng,))
        y[2*Ng:3*Ng] = self.g3.reshape((Ng,))
        return y

    def pack(self):
        Ng = self.N_patche**3
        y = np.zeros(3*Ng)
        y[:Ng] = self.dg1.reshape((Ng,))
        y[Ng:2*Ng] = self.dg2.reshape((Ng,))
        y[2*Ng:3*Ng] = self.dg3.reshape((Ng,))
        return y

    def unpack(self, y):
        Ng = self.N_patche**3
        self.g1 = y[:Ng].reshape((self.N_patche, self.N_patche, self.N_patche))
        self.g2 = y[Ng:2*Ng].reshape((self.N_patche,
                                      self.N_patche, self.N_patche))
        self.g3 = y[2*Ng:3 *
                    Ng].reshape((self.N_patche, self.N_patche, self.N_patche))

    def equations_rg(self, bulle):
        Np = self.N_patche
        self.dg1 = np.zeros((Np, Np, Np), float)
        self.dg2 = np.zeros((Np, Np, Np), float)
        self.dg3 = np.zeros((Np, Np, Np), float)

        for k1 in range(-Np//2, Np//2):
            for k2 in range(-Np//2, Np//2):
                qc = (k1+k2) % Np
                for k3 in range(-Np//2, Np//2):
                    qp = (k3-k2) % Np
                    qpp = (k1-k3) % Np
                    k4 = (k1+k2-k3) % Np
                    i = (k1, k2, k3)

                    for kp in range(-Np//2, Np//2):
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

                        self.dg1[i] += -0.5*((self.g2[m1]*self.g1[m2]+self.g1[m1]*self.g2[m2])*IC
                                             - (self.g2[m3]*self.g1[m4]+self.g1[m3]*self.g2[m4]
                                                - 2*self.g1[m3]*self.g1[m4])*IP2)
                        self.dg1[i] += 0.5*(self.g3[m5]*self.g3[m6] + self.g3[m7]*self.g3[k1, kp, k4]
                                            - 2.0*self.g3[m5]*self.g3[m8])*IP

                        self.dg2[i] += 0.5*(-(self.g2[m1]*self.g2[m2]+self.g1[m1]*self.g1[m2])*IC
                                            + self.g2[m3]*self.g2[m4] * IP2)
                        self.dg2[i] += 0.5*self.g3[k1, kp, k4]*self.g3[m6]*IP

                        self.dg3[i] += 0.5*(self.g3[m5]*self.g2[m7] + self.g3[k1, kp, k4]*self.g1[m7] +
                                            self.g2[m5]*self.g3[m7] + self.g1[m5]*self.g3[m6] -
                                            2*self.g1[m7]*self.g3[m5] - 2*self.g3[m7]*self.g1[m5])*IP
                        self.dg3[i] += 0.5*(self.g3[m12]*self.g2[m9] +
                                            self.g3[m10]*self.g2[m11])*IPP

        self.dg = self.pack()
