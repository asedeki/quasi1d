# import pdb
import numpy as np
cimport numpy as np
#ctypedef struct param
from Seebeck.src.utils.system.structure cimport param


cdef class MatriceDiffusion:

    cdef readonly param arg
    cdef double[:,:,:] g3
    #cpdef void initialisation(self, dict arg,  )
    cdef inline double eperp(self, long k)
    cdef double sigma(self, double sum_eperp)
    cdef void get_sigma(self, double[:,:,::1], double[:,:,::1])
    cpdef double[:,:] get_collision_matrix(self)
    cdef double[:] get_row_collision_matrix(self,
                    double[:,:,::1] mu_1, double[:,:,::1] mu_2, long k1)
    cdef double get_ek_deriv(self, double e, double eta, double tp)