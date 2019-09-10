from Seebeck.src.utils.system.structure cimport param
cdef extern from './cubature/cubature.h':
    # values for the error_norm parameter
    ctypedef enum error_norm:
        ERROR_INDIVIDUAL = 0
        ERROR_PAIRED
        ERROR_L2
        ERROR_L1
        ERROR_LINF

    ctypedef int (*integrand_v) (unsigned ndim, size_t npt, const double *x,
                                 void *fdata, unsigned fdim, double *fval)

    int hcubature_v(unsigned fdim, integrand_v f, void *fdata,
                    unsigned ndim, const double *xmin, const double *xmax,
                    size_t maxEval, double reqAbsError, double reqRelError,
                    error_norm norm, double *val, double *err) nogil

    int pcubature_v(unsigned fdim, integrand_v f, void *fdata,
                    unsigned ndim, const double *xmin, const double *xmax,
                    size_t maxEval, double reqAbsError, double reqRelError,
                    error_norm norm, double *val, double *err) nogil
# Vectorized version with user-supplied buffer to store points and values.
# The buffer *buf should be of length *nbuf * dim on entry (these parameters
# are changed upon return to the final buffer and length that was used).
# The buffer length will be kept <= max(max_nbuf, 1) * dim.
#
# Also allows the caller to specify an array m[dim] of starting degrees
# for the rule, which upon return will hold the final degrees.  The
# number of points in each dimension i is 2^(m[i]+1) + 1.
    int pcubature_v_buf(unsigned fdim, integrand_v f, void *fdata,
                    unsigned ndim, const double *xmin, const double *xmax,
                    size_t maxEval, double reqAbsError, double reqRelError,
                    error_norm norm, unsigned *m,
                    double **buf, size_t *nbuf, size_t max_nbuf,
                    double *val, double *err)

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
    


############# New ##########################
ctypedef int (*integration_v)(integrand_v , void *, double*,double*, double*, double*, unsigned)



cdef int integration_p_v(integrand_v , void *, double*,double*, double*, double*, unsigned)nogil
cdef int integration_h_v(integrand_v , void *,double*,double*, double*, double*, unsigned) nogil


cdef double theta(double) 
cdef double eperp(double , double, pquasi1d*, double) 
cdef double eperpc(double, double, double, pquasi1d*, double)  
cdef double gradient(double, double, double) 
cdef int deriv(unsigned ndim, size_t npt, const double *x,
                        void *fdata, unsigned fdim, double *fval)
cdef int derivsusc(unsigned ndim, size_t npt, const double *x,
                        void *fdata, unsigned fdim, double *fval)

cdef double vbulle(integrand_v, pquasi1d*, double, double, double, double, double) nogil
cpdef void valeursbulles(double, double, pquasi1d, double[:,:,::1], double[:,:,::1] , double[:,::1])


