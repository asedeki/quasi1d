# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

## cython: profile=True

import numpy as np
cimport numpy as np
from math import pi

# todo reflechir a wrap(), %N ne donne que des valeurs positives


cdef class Interaction:
    cdef public double[:,:,::1] g1, g2, g3
    cdef readonly double[:,:,::1] dg1, dg2, dg3
    cdef int Np
    def __init__(self, int N_patche):
        
        self.g1 = np.zeros((N_patche, N_patche, N_patche), float)
        self.g2 = np.zeros((N_patche, N_patche, N_patche), float)
        self.g3 = np.zeros((N_patche, N_patche, N_patche), float)
        self.Np = N_patche

    cpdef void initialisation(self, double g1, double g2, double g3, double g1_perp=0, 
                        double g2_perp=0, double g3_perp=0):
        cdef:
            int N2, k1, k2, k3

        N2 = self.Np
        for k3 in range(N2):
            for k2 in range(N2):
                for k1 in range(N2):
                    self.g1[k1, k2, k3] = g1
                    self.g2[k1, k2, k3] = g2
                    self.g3[k1, k2, k3] = g3

    cdef double[:] inipack(self):
        cdef:
            int Ng
            double[:] y

        Ng = self.Np**3
        y = np.zeros(3*Ng)
        y[:Ng] = self.g1.reshape((Ng,))
        y[Ng:2*Ng] = self.g2.reshape((Ng,))
        y[2*Ng:3*Ng] = self.g3.reshape((Ng,))
        return y

    cpdef double[:] pack(self):
        cdef:
            int Ng
            double[:] y

        Ng = self.Np**3
        y = np.zeros(3*Ng)
        y[:Ng] = self.dg1.reshape((Ng,))
        y[Ng:2*Ng] = self.dg2.reshape((Ng,))
        y[2*Ng:3*Ng] = self.dg3.reshape((Ng,))
        return y

    cpdef void unpack(self, double[:] y):
        cdef:
            int Ng

        Ng = self.Np**3
        self.g1 = y[:Ng].reshape((self.Np, self.Np, self.Np))
        self.g2 = y[Ng:2*Ng].reshape((self.Np,
                                      self.Np, self.Np))
        self.g3 = y[2*Ng:3 *
                    Ng].reshape((self.Np, self.Np, self.Np))

    cpdef void equations_rg(self, double[:,:,::1] dg1, double[:,:,:] bulle_IP, double[:,:,:] bulle_IC):
        cdef:
            int Np, N2
            int k1, k2, k3, k4
            int qp, qpp, qc, kp
            int qc_kp, kp_qp, kpqp, kpqpp
            
            # tuple i, m1, kp, qc_kp, k3, m3, m4, k1, kp, kpqp, k2, kpqp, k3, k2, kpqp, kp
            # tuple m8, m9, m10, m11, m12
            double IP, IC, IPP, IP2


        Np = self.Np
        

        self.dg1 = np.zeros((Np, Np, Np), float)
        self.dg2 = np.zeros((Np, Np, Np), float)
        self.dg3 = np.zeros((Np, Np, Np), float)
        N2 = Np // 2
        for k1 in range(Np):
            for k2 in range(Np):
                qc = (k1+k2) % Np
                for k3 in range(-N2, N2):
                    qp = k3#(k3-k2) % Np
                    qpp = k1#(k1-k3) % Np
                    k4 = k2#(k1+k2-k3) % Np
                    
                    for kp in range(Np):
                        IP = bulle_IP[k2, kp, qp]
                        IP2 = bulle_IP[k2, kp, -qp]
                        IC = bulle_IC[k1, kp, qc]
                        IPP = bulle_IP[k3, kp, qpp]
                        qc_kp = qc#(qc-kp) % Np
                        kp_qp = qp#(kp-qp) % Np
                        kpqp = kp#(kp+qp) %Np
                        kpqpp = qc#(kp+qpp) %Np

                        # m1 = (k1, k2, kp)
                        # m2 = (kp, qc_kp, k3)
                        # m3 = (k1, kp_qp, kp)
                        # m4 = (kp, k2, k3)
                        # m5 = (k1, kp, kpqp)
                        # m6 = (k2, kpqp, k3)
                        # m7 = (k2, kpqp, kp)
                        # m8 = (kpqp, k2, k3)
                        # m9 = (k2, kpqpp, kp)
                        # m10 = (k2, kpqpp, k4)
                        # m11 = (k1, kp, kpqpp)
                        # m12 = (k1, kp, k3)

                        dg1[k1, k2, k3] += -0.5*((self.g2[k1,k2,kp]*self.g1[kp, qc_kp, k3]
                                            +self.g1[k1,k2,kp]*self.g2[kp, qc_kp, k3])*IC - 
                                            (self.g2[k1, kp_qp, kp]*self.g1[kp, k2, k3] +
                                             self.g1[k1, kp_qp, kp]*self.g2[kp, k2, k3] - 
                                             2*self.g1[k1, kp_qp, kp]*self.g1[kp, k2, k3])*IP2)

                        dg1[k1, k2, k3] += 0.5*(self.g3[k1, kp, kpqp]*self.g3[k2, kpqp, k3] + 
                                                self.g3[k2, kpqp, kp]*self.g3[k1, kp, k4] - 
                                                2.0*self.g3[k1, kp, kpqp]*self.g3[kpqp, k2, k3])*IP

                        self.dg2[k1, k2, k3] += 0.5*(-(self.g2[k1, k2, kp]*self.g2[kp, qc_kp, k3] + 
                                                    self.g1[k1, k2, kp]*self.g1[kp, qc_kp, k3])*IC + 
                                                    self.g2[k1, kp_qp, kp]*self.g2[kp, k2, k3] * IP2)
                        self.dg2[k1, k2, k3] += 0.5*self.g3[k1, kp, k4]*self.g3[k2, kpqp, k3]*IP

                        self.dg3[k1, k2, k3] += 0.5*(self.g3[k1, kp, kpqp]*self.g2[k2, kpqp, kp] + 
                                                    self.g3[k1, kp, k4]*self.g1[k2, kpqp, kp] + 
                                                    self.g2[k1, kp, kpqp]*self.g3[k2, kpqp, kp] + 
                                                    self.g1[k1, kp, kpqp]*self.g3[k2, kpqp, k3] -
                                                    2*self.g1[k2, kpqp, kp]*self.g3[k1, kp, kpqp] - 
                                                    2*self.g3[k2, kpqp, kp]*self.g1[k1, kp, kpqp])*IP

                        self.dg3[k1, k2, k3] += 0.5*(self.g3[k1, kp, k3]*self.g2[k2, kpqpp, kp] +
                                            self.g3[k2, kpqpp, k4]*self.g2[k1, kp, kpqpp])*IPP

        self.dg = self.pack()
