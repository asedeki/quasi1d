ctypedef struct pquasi1d:
    double tp
    double tp2
    double Ef
    int Np

ctypedef struct parametres:
    double kf
    double qp
    double sgn
    double T
    pquasi1d* pq1d

ctypedef double (*integrand) (double k_perp, void * params) nogil

cdef double theta(double) nogil
cdef double eperp(double , double, pquasi1d*, double) nogil
cdef double eperpc(double, double, double, pquasi1d*, double) nogil
cdef double gradient(double, double, double) nogil
cdef double deriv(double, void * ) nogil
cdef double derivsusc(double, void * ) nogil
cdef double vbulle(integrand, pquasi1d*, double, double, double, double, double)
cpdef void valeursbulles(double, double, pquasi1d, double[:,:,::1], double[:,:,::1] , double[:,::1])
