# cython: boundscheck=False
## cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
#distutils: extra_compile_args = -fopenmp 
#distutils: extra_link_args = -fopenmp
##extra_compile_args=['-fopenmp'],
##extra_link_args=['-fopenmp'],
## cython: profile=True

import numpy as np
cimport numpy as np
from math import pi
from cython.parallel import parallel, prange
from cython import boundscheck, wraparound

# todo reflechir a wrap(), %N ne donne que des valeurs positives



cdef double[:] inipack( int Np, double[:,:,::1] g1, double[:,:,::1] g2, double[:,:,::1] g3):
    cdef:
        int Ng
        double[:] y

    Ng = Np**3
    y = np.zeros(3*Ng)
    y[:Ng] = g1.reshape((Ng,))
    y[Ng:2*Ng] = g2.reshape((Ng,))
    y[2*Ng:3*Ng] = g3.reshape((Ng,))
    return y

cpdef double[:] pack(int Np, double[:,:,::1] dg1, double[:,:,::1] dg2, double[:,:,::1] dg3):
    cdef:
        int Ng
        double[:] y

    Ng = Np**3
    y = np.zeros(3*Ng)
    y[:Ng] = dg1.reshape((Ng,))
    y[Ng:2*Ng] = dg2.reshape((Ng,))
    y[2*Ng:3*Ng] = dg3.reshape((Ng,))
    return y

cpdef void unpack(int Np, double[:] y):
    cdef:
        int Ng
        double[:,:,::1] g1, g2, g3

    Ng = Np**3
    g1 = y[:Ng].reshape((Np, Np, Np))
    g2 = y[Ng:2*Ng].reshape((Np,
                                    Np, Np))
    g3 = y[2*Ng:3 *
                Ng].reshape((Np, Np, Np))

cdef inline int wrap(int i, int Np)nogil:
    cdef:
        int N2 = Np//2
    # if (-N2<= i< N2):
    #     return i
    # elif i < -N2:
    #     return i + Np
    # else:
    #     return i - Np
    return i%Np
#@boundscheck(False)
#@wraparound(False)  
cpdef void equations_rg(int Np, double[:,:,::1] g1, double[:,:,::1] g2, double[:,:,::1] g3, 

                        double[:,:,::1] bulle_IP, double[:,:,::1] bulle_IC, 
                        double[:,:,::1] dg1, double[:,:,::1] dg2, double[:,:,::1] dg3):
    cdef:
        int N2, Npp
        int k1, k2, k3, k4
        int qp, qpp, qc, kp
        int qc_kp, kp_qp, kpqp, kpqpp
        # tuple i, m1, kp, qc_kp, k3, m3, m4, k1, kp, kpqp, k2, kpqp, k3, k2, kpqp, kp
        # tuple m8, m9, m10, m11, m12
        double IP, IC, IPP, IP2
    
    #Npp = g1.shape[0]
    #print(np.array(bulle_IP))
    N2 = Np // 2
    # with nogil, parallel(num_threads=1):
    #     for k1 in prange(-N2,N2, schedule="dynamic"):#prange(-N2,N2, nogil=True)
    for k1 in range(-N2,1):
        for k2 in range(-N2,N2):
            qc = wrap(k1+k2,Np)
            for k3 in range(-N2,N2):
                qp = wrap(k3-k2,Np)
                qpp = wrap(k1-k3,Np)
                k4 = wrap(k1+k2-k3,Np)
                dg1[k1, k2, k3] = 0.0
                dg2[k1, k2, k3] = 0.0
                dg3[k1, k2, k3] = 0.0
                
                
                with nogil, parallel():
                    for kp in range(-N2,N2):
                        #print(qp,qc,qpp,wrap(-qp,Np))
                        IP = bulle_IP[k2][kp][qp]
                        IP2 = bulle_IP[k2][kp][wrap(-qp,Np)]
                        IC = bulle_IC[k1, kp, qc]
                        IPP = bulle_IP[k3, kp, qpp]
                        qc_kp = wrap(qc-kp,Np)
                        kp_qp = wrap(kp-qp,Np)
                        kpqp = wrap(kp+qp,Np)
                        kpqpp = wrap(kp+qpp,Np)
                        #print(k2, kp, wrap(-qp,Np),IP2)
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

                        dg1[k1, k2, k3] += -0.5*((g2[k1,k2,kp]*g1[kp, qc_kp, k3]
                                            +g1[k1,k2,kp]*g2[kp, qc_kp, k3])*IC - 
                                            (g2[k1, kp_qp, kp]*g1[kp, k2, k3] +
                                                g1[k1, kp_qp, kp]*g2[kp, k2, k3] - 
                                                2*g1[k1, kp_qp, kp]*g1[kp, k2, k3])*IP2)

                        dg1[k1, k2, k3] += 0.5*(g3[k1, kp, kpqp]*g3[k2, kpqp, k3] + 
                                                g3[k2, kpqp, kp]*g3[k1, kp, k4] - 
                                                2.0*g3[k1, kp, kpqp]*g3[kpqp, k2, k3])*IP

                        dg2[k1, k2, k3] += 0.5*(-(g2[k1, k2, kp]*g2[kp, qc_kp, k3] + 
                                                    g1[k1, k2, kp]*g1[kp, qc_kp, k3])*IC + 
                                                    g2[k1, kp_qp, kp]*g2[kp, k2, k3] * IP2)
                        dg2[k1, k2, k3] += 0.5*g3[k1, kp, k4]*g3[k2, kpqp, k3]*IP

                        dg3[k1, k2, k3] += 0.5*(g3[k1, kp, kpqp]*g2[k2, kpqp, kp] + 
                                                    g3[k1, kp, k4]*g1[k2, kpqp, kp] + 
                                                    g2[k1, kp, kpqp]*g3[k2, kpqp, kp] + 
                                                    g1[k1, kp, kpqp]*g3[k2, kpqp, k3] -
                                                    2*g1[k2, kpqp, kp]*g3[k1, kp, kpqp] - 
                                                    2*g3[k2, kpqp, kp]*g1[k1, kp, kpqp])*IP

                        dg3[k1, k2, k3] += 0.5*(g3[k1, kp, k3]*g2[k2, kpqpp, kp] +
                                            g3[k2, kpqpp, k4]*g2[k1, kp, kpqpp])*IPP

    for k1 in range(1,N2):
        for k2 in range(-N2,N2):
            for k3 in range(-N2,N2):
                dg1[k1, k2, k3] = dg1[-k1, -k2, -k3]                 
                dg2[k1, k2, k3] = dg2[-k1, -k2, -k3]                 
                dg3[k1, k2, k3] = dg3[-k1, -k2, -k3]                 


cpdef void equations_rgcol(int Np, int k1, double[:,:,::1] g1, double[:,:,::1] g2, double[:,:,::1] g3, 
                        double[:,:,::1] bulle_IP, double[:,:,::1] bulle_IC, 
                        double[:,::1] dg1, double[:,::1] dg2, double[:,::1] dg3):
    cdef:
        int N2, Npp
        int k2, k3, k4
        int qp, qpp, qc, kp
        int qc_kp, kp_qp, kpqp, kpqpp
        # tuple i, m1, kp, qc_kp, k3, m3, m4, k1, kp, kpqp, k2, kpqp, k3, k2, kpqp, kp
        # tuple m8, m9, m10, m11, m12
        double IP, IC, IPP, IP2
    
    #Npp = g1.shape[0]
    #print(np.array(bulle_IP))
    N2 = Np // 2

    for k2 in range(-N2,1):
        qc = wrap(k1+k2,Np)
        for k3 in range(-N2,N2):
            qp = wrap(k3-k2,Np)
            qpp = wrap(k1-k3,Np)
            k4 = wrap(k1+k2-k3,Np)
            dg1[k2, k3] = 0.0
            dg2[k2, k3] = 0.0
            dg3[k2, k3] = 0.0
            
            for kp in range(-N2,N2):
                #print(qp,qc,qpp,wrap(-qp,Np))
                IP = bulle_IP[k2][kp][qp]
                IP2 = bulle_IP[k2][kp][wrap(-qp,Np)]
                IC = bulle_IC[k1, kp, qc]
                IPP = bulle_IP[k3, kp, qpp]
                qc_kp = wrap(qc-kp,Np)
                kp_qp = wrap(kp-qp,Np)
                kpqp = wrap(kp+qp,Np)
                kpqpp = wrap(kp+qpp,Np)
                #print(k2, kp, wrap(-qp,Np),IP2)
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

                dg1[k2, k3] += -0.5*((g2[k1,k2,kp]*g1[kp, qc_kp, k3]
                                    +g1[k1,k2,kp]*g2[kp, qc_kp, k3])*IC - 
                                    (g2[k1, kp_qp, kp]*g1[kp, k2, k3] +
                                        g1[k1, kp_qp, kp]*g2[kp, k2, k3] - 
                                        2*g1[k1, kp_qp, kp]*g1[kp, k2, k3])*IP2)

                dg1[k2, k3] += 0.5*(g3[k1, kp, kpqp]*g3[k2, kpqp, k3] + 
                                        g3[k2, kpqp, kp]*g3[k1, kp, k4] - 
                                        2.0*g3[k1, kp, kpqp]*g3[kpqp, k2, k3])*IP

                dg2[k2, k3] += 0.5*(-(g2[k1, k2, kp]*g2[kp, qc_kp, k3] + 
                                            g1[k1, k2, kp]*g1[kp, qc_kp, k3])*IC + 
                                            g2[k1, kp_qp, kp]*g2[kp, k2, k3] * IP2)
                dg2[k2, k3] += 0.5*g3[k1, kp, k4]*g3[k2, kpqp, k3]*IP

                dg3[k2, k3] += 0.5*(g3[k1, kp, kpqp]*g2[k2, kpqp, kp] + 
                                            g3[k1, kp, k4]*g1[k2, kpqp, kp] + 
                                            g2[k1, kp, kpqp]*g3[k2, kpqp, kp] + 
                                            g1[k1, kp, kpqp]*g3[k2, kpqp, k3] -
                                            2*g1[k2, kpqp, kp]*g3[k1, kp, kpqp] - 
                                            2*g3[k2, kpqp, kp]*g1[k1, kp, kpqp])*IP

                dg3[k2, k3] += 0.5*(g3[k1, kp, k3]*g2[k2, kpqpp, kp] +
                                    g3[k2, kpqpp, k4]*g2[k1, kp, kpqpp])*IPP
           
