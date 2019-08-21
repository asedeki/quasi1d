# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

from libc.math cimport M_PI, tanh, exp
from cython_gsl cimport *


cdef double theta(double x) nogil:
    cdef:
        double out
    if(x==0):
        out = 0.5
    elif(x<0):
        out = -1.0
    else:
        out = +1.0

    return out

cdef double eperp(double kperp, double qperp, pquasi1d *pq, double sign) nogil:
    cdef:
        double res, k, q
    
    k = 2.0 * kperp*M_PI / pq.Np
    q = 2.0 * qperp*M_PI / pq.Np

    res = 2.0 * pq.tp * cos(k) + 2.0 * pq.tp2 * cos(2.0 * k) 
    res += sign * (2.0 * pq.tp * cos(q + sign * k)+2.0 * pq.tp2 * cos(2.0 * (q + sign * k)))
    return res

cdef double eperpc(double kperp, double kfermi, double qperp, pquasi1d *pq, double sgn) nogil:
    return eperp(kperp, qperp, pq, sgn)-eperp(kfermi, qperp, pq, sgn)

cdef double gradient(double A, double temperature, double fermiE) nogil:
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

cdef double deriv(double k_perp, void * params) nogil:
    cdef:
        parametres *pm = <parametres*> params
        double A, result
    A = eperpc(k_perp, pm.kf, pm.qp, pm.pq1d, pm.sgn)
    result = gradient(A, pm.T, pm.pq1d.Ef)
    return  result

cdef double derivsusc(double k_perp, void * params) nogil:
    cdef:
        parametres *pm = <parametres*> params
        double A, result
    A = eperp(k_perp, pm.qp, pm.pq1d, pm.sgn)
    result = gradient(A, pm.T, pm.pq1d.Ef)
    return  result

cdef double vbulle(integrand derivee, pquasi1d *pq1d, double kp, double kf, double qp, double sgn, double T):
    cdef:
        double result, error
        double k_ini, k_end
        parametres param
        gsl_integration_workspace * w
        gsl_function F

    w = gsl_integration_workspace_alloc (1000)
    param.kf = kf
    param.qp = qp
    param.kf = kf
    param.sgn = sgn
    param.T = T
    param.pq1d = pq1d
    
    k_ini = kp-0.5 
    k_end = kp+0.5

    F.function = derivee
    F.params = &param
    
    gsl_integration_qags (&F, k_ini, k_end, 0, 1e-7, 1000, w, &result, &error)

    return result/float(pq1d.Np)

# cdef void calculbulles(double x, double T, pquasi1d param,
#     double[:,:,::1] IC, double[:,:,::1] IP, double[:,::1] IPsusc):
#     cdef:
#         int kp, qp, k1
#         int N2 = param.Np
#         pquasi1d param1 = param
    
#     param1.Ef = param.Ef * exp(-x)

#     for qp in range(N2):
#         IPsusc[qp,0] = vbulle(&derivsusc, &param1, float(qp), float(qp), 0.0, +1,T)
#         IPsusc[qp,1] = vbulle(&derivsusc, &param1, float(qp), float(qp), float(N2), +1,T)
#         for kp in range(N2):
#             for k1 in range(N2):
#                 IP[k1, kp, qp] = vbulle(&deriv, &param1, float(kp), float(k1), float(qp), +1,T)
#                 IC[k1, kp, qp] = vbulle(&deriv, &param1, float(kp), float(k1), float(qp), -1,T)

cpdef void valeursbulles(double x, double T, pquasi1d param,
    double[:,:,::1] IC, double[:,:,::1] IP, double[:,::1] IPsusc):
    cdef:
        int kp, qp, k1
        int N2 = param.Np//2
        pquasi1d param1 = param
    
    param1.Ef = param.Ef * exp(-x)

    for qp in range(-N2,N2):
        IPsusc[qp+N2,0] = vbulle(&derivsusc, &param1, float(qp), float(qp), 0.0, +1,T)
        IPsusc[qp+N2,1] = vbulle(&derivsusc, &param1, float(qp), float(qp), float(N2), +1,T)
        for kp in range(-N2,N2):
            for k1 in range(-N2,N2):
                IP[k1+N2, kp+N2, qp+N2] = vbulle(&deriv, &param1, float(kp), float(k1), float(qp), +1,T)
                IC[k1+N2, kp+N2, qp+N2] = vbulle(&deriv, &param1, float(kp), float(k1), float(qp), -1,T)

