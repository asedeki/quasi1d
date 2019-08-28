
cdef extern from "gsl/gsl_math.h":
  ctypedef struct gsl_function:
    double (* function) (double x, void * params)
    void * params

cdef extern from "gsl/gsl_integration.h":
  ctypedef struct gsl_integration_cquad_workspace
  ctypedef struct gsl_integration_workspace
  
  gsl_integration_cquad_workspace *  gsl_integration_cquad_workspace_alloc (size_t n)
  void  gsl_integration_cquad_workspace_free (gsl_integration_cquad_workspace * w)

  gsl_integration_workspace *  gsl_integration_workspace_alloc (size_t n)
  void  gsl_integration_workspace_free (gsl_integration_workspace * w)

  int gsl_integration_cquad (gsl_function * f, double a, double b, double epsabs, 
            double epsrel, gsl_integration_cquad_workspace * workspace, double * result, double * abserr
            , size_t * nevals)
  int gsl_integration_qng(const gsl_function * f, double a, double b, double epsabs, double epsrel, 
                          double * result, double * abserr, size_t * neval)
  int gsl_integration_qag(const gsl_function * f, double a, double b, double epsabs, double epsrel, 
      size_t limit, int key, gsl_integration_workspace * workspace, double * result, double * abserr)


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
cdef double vbulle(integrand, pquasi1d*, double, double, double, double, double)nogil
cpdef void valeursbulles(double, double, pquasi1d, double[:,:,::1], double[:,:,::1] , double[:,::1])
