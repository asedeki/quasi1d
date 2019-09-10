# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
## cython: profile=True
## distutils: extra_compile_args = -DVECTORISED

from libc cimport math
from libc.math cimport M_PI, tanh, exp, cosh, fabs, cos
import cython
from cython.parallel import parallel, prange
from libc.stdio cimport printf

cdef double theta(double x):
    cdef:
        double out
    out = 1.0
    if(x==0):
        out = 0.5
    elif(x<0):
        out = 0.0
    return out

cdef double eperp(double kperp, double qperp, pquasi1d *pq, double sign) :
    cdef:
        double res, k, q
    
    k = 2.0 * kperp*M_PI / pq.Np
    q = 2.0 * qperp*M_PI / pq.Np

    res = 2.0 * pq.tp * cos(k) + 2.0 * pq.tp2 * cos(2.0 * k) 
    res += sign * (2.0 * pq.tp * cos(q + sign * k)+2.0 * pq.tp2 * cos(2.0 * (q + sign * k)))
    return res

cdef double eperpc(double kperp, double kfermi, double qperp, pquasi1d *pq, double sgn):
    return eperp(kperp, qperp, pq, sgn)-eperp(kfermi, qperp, pq, sgn)

cdef double gradient(double A, double temperature, double fermiE) :
    cdef:
        double mu = 1.0, yp=0.0, res
        double arg1, arg2, D
        int i
    
    arg1 = 0.5 * fermiE / temperature;
    for i in range(2):
        arg2 = arg1 + mu * 0.5 * A / temperature
        D = 1.0 + arg2 / arg1
        res = (tanh(arg1) + tanh(arg2)) / D
        yp += theta(fabs(fermiE + mu * A) - fermiE) * res
        mu = -1.0
        
    return yp

cdef int derivsusc(unsigned ndim, size_t npt, const double *x,
                void *fdata, unsigned fdim, double *fval):
    cdef:
        unsigned int j
        parametres *pm = <parametres*> fdata
        double A
    for j in range(npt):
        A = eperp(x[j], pm.qp, pm.pq1d, pm.sgn)
        fval[j]= gradient(A, pm.T, pm.pq1d.Ef)
    return 0

cdef int deriv(unsigned ndim, size_t npt, const double *x,
                void *fdata, unsigned fdim, double *fval):
    cdef:
        unsigned int j
        parametres *pm = <parametres*> fdata
        double A

    for j in range(npt):
        A = eperpc(x[j], pm.kf, pm.qp, pm.pq1d, pm.sgn)
        fval[j]= gradient(A, pm.T, pm.pq1d.Ef)
    return 0

cdef int integration_h_v(integrand_v fg, void *fdata, double* val,
                    double* err, double* xmin, double* xmax, unsigned xdim) nogil:
    cdef:
        unsigned fdim=1
        double reqRelError=1e-6
        size_t maxEval = 0
        double reqAbsError = 0.0
    hcubature_v(fdim, fg, fdata, xdim, xmin, xmax, maxEval,
                reqAbsError, reqRelError, error_norm.ERROR_INDIVIDUAL, val, err)
    return 0


cdef int integration_p_v(integrand_v fg, void *fdata, double* val,
                            double* err, double* xmin, double* xmax, unsigned xdim) nogil:
    cdef:
        unsigned fdim=1
        double reqRelError=1e-6
        size_t maxEval = 0
        double reqAbsError = 0.0
        
    
    pcubature_v(fdim, fg, fdata, xdim, xmin, xmax, maxEval,
                reqAbsError, reqRelError, error_norm.ERROR_INDIVIDUAL, val, err)
    return 0

cdef double vbulle(integrand_v derivee, pquasi1d *pq1d, double kp, 
                    double kf, double qp, double sgn, double T) nogil:
    cdef:
        parametres Params
        void* fdata
        double xmin[1]
        double xmax[1]
        unsigned xdim = 1
        double val, err
        integration_v f_integration

    #f_integration = integration_h_v
    f_integration = integration_p_v
    Params.pq1d = pq1d
    
    Params.kf = kf
    Params.qp = qp
    Params.sgn = sgn
    Params.T = T

    fdata = &Params
       
    xmin[0] = kp - 0.5
    xmax[0] = kp + 0.5
    integration_p_v(derivee, fdata, &val, &err, xmin,xmax,xdim)
    
    return val/float(pq1d.Np)


cpdef void valeursbulles(double x, double T, pquasi1d param,
    double[:,:,::1] IC, double[:,:,::1] IP, double[:,::1] IPsusc):
    cdef:
        int kp, qp, k1
        int N2 = param.Np//2
        int Np = param.Np
        pquasi1d param1 = param
    
    param1.Ef = param.Ef * exp(-x)
    with nogil, parallel():
        for k1 in prange(Np):
            IPsusc[k1,0] = vbulle(&derivsusc, &param1, float(k1), float(k1), 0.0, +1,T)
            IPsusc[k1,1] = vbulle(&derivsusc, &param1, float(k1), float(k1), float(N2), +1,T)
            for kp in range(Np): 
                for qp in range(N2+1):
                        IP[k1, kp, qp] = vbulle(&deriv, &param1, float(k1), float(kp), float(qp), +1,T)
                        IC[k1, kp, qp] = vbulle(&deriv, &param1, float(k1), float(kp), float(qp), -1,T)
    
    
    for k1 in range(Np):
        for kp in range(Np):
            for qp in range(N2+1,Np):
                IP[k1, kp, qp] = IP[(Np-k1)%Np, (Np-kp)%Np, (Np-qp)%Np] 
                IC[k1, kp, qp] = IC[(Np-k1)%Np, (Np-kp)%Np, (Np-qp)%Np]