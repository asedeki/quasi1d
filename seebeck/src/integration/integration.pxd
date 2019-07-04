from  seebeck.src.structure cimport param

ctypedef struct Pdata:
    double k1
    double k2
    param* P
    double ebeta


cdef double eperp(double kperp, double tp, double tp2)
cdef double sum_eperp(unsigned int nkperp, double* kperp, double tp, double tp2)
cdef double sigma(double sum_eperp, double ebeta)