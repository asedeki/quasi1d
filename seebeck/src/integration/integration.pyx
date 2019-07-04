# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
## cython: profile=True
## distutils: extra_compile_args = -DVECTORISED

cimport numpy as np
import numpy as np
from libc cimport math
import cython
from cython.operator cimport dereference as deref
from libc.stdlib cimport malloc, free, calloc
import sys
import scipy
cdef double eperp(double kperp, double tp, double tp2):
    return  -2.0 * tp * math.cos(kperp) - 2.0 * tp2 * math.cos(2.0 * kperp)

cdef double sum_eperp(unsigned int nkperp, double* kperp, double tp, double tp2):
    cdef:
        double sum = 0.0
        unsigned int i
    for i in range(nkperp):
        sum += eperp(kperp[i], tp, tp2)
    return sum

cdef double sigma(double sig_beta, double e_beta):
    cdef:
        double sig_sur_T
        double sigma_value
        double inf1
        double inf2
    sig_sur_T = sig_beta -0.5 * e_beta
    sigma_value = 0.0
    inf1 = math.fabs(sig_beta)
    inf2 = math.fabs(sig_sur_T)
    if(inf1==0):
        sigma_value = 0.5/math.cosh(sig_sur_T)
    else:
        sigma_value = sig_beta/math.sinh(2*sig_beta)/math.cosh(sig_sur_T)
    # if not (inf1 >= 20 or inf2 >=20):
    #     if (inf1==0.0):
    #         sigma_value = 0.5/math.cosh(sig_sur_T)
    #     else:
    #         sigma_value = sig_beta/math.sinh(2*sig_beta)/math.cosh(sig_sur_T)
    return sigma_value