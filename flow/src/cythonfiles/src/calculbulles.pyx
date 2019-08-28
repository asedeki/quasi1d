# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
#distutils: extra_compile_args = -fopenmp -O2
#distutils: extra_link_args = -fopenmp


from libc.math cimport M_PI, tanh, exp, cosh, fabs, cos
from cython_gsl cimport *
from libc.stdio cimport printf
from cython.parallel import parallel, prange
#cimport openmp
cimport cython

cdef double theta(double x)nogil:
    cdef:
        double out
    if(x==0):
        out = 0.5
    elif(x<0):
        out = 0.0
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
        #printf("A=%f, %f, %f, %f\n",A, fermiE,arg1,arg2)
        D = 1.0 + arg2 / arg1
        res = (tanh(arg1) + tanh(arg2)) / D
        # if fabs(D) >= 1e-10:
        #     res = (tanh(arg1) + tanh(arg2)) / D
        # else:
        #     res = arg1 * 1.0/cosh(arg1)**2

        yp += theta(fabs(fermiE + mu * A) - fermiE) * res
        
        mu = -1.0
        #printf("r,A,yp=%f  ,%f, %f, %f\n",mu,res, A, yp)
    return yp

cdef double deriv(double k_perp, void * params) nogil:
    cdef:
        parametres *pm = <parametres*> params
        double A, result
    A = eperpc(k_perp, pm.kf, pm.qp, pm.pq1d, pm.sgn)
    
    result = gradient(A, pm.T, pm.pq1d.Ef)
    
    return  result

cdef double derivsusc(double k_perp, void * params)nogil:
    cdef:
        parametres *pm = <parametres*> params
        double A, result
    A = eperp(k_perp, pm.qp, pm.pq1d, pm.sgn)
    result = gradient(A, pm.T, pm.pq1d.Ef)
    return  result

# cdef double vbulle(integrand derivee, pquasi1d *pq1d, double kp, double kf, double qp, double sgn, double T):
#     cdef:
#         double result, error
#         double k_ini, k_end
#         parametres param
#         gsl_integration_workspace * w
#         gsl_function F

#     w = gsl_integration_workspace_alloc (100000)
#     param.kf = kf
#     param.qp = qp 
#     param.kf = kf
#     param.sgn = sgn
#     param.T = T
#     param.pq1d = pq1d

#     k_ini = kp-0.5 
#     k_end = kp+0.5

#     F.function = derivee
#     F.params = &param
#     gsl_integration_qags (&F, k_ini, k_end, 0, 1e-5, 100000, w, &result, &error)
#     gsl_integration_workspace_free(w)
#     return result/float(pq1d.Np)

cdef double vbulle(integrand derivee, pquasi1d *pq1d, double kp, double kf, double qp, double sgn, double T) nogil:
    cdef:
        double result, error
        double k_ini, k_end
        parametres param
        gsl_function F
        size_t nevals
        int key = 1 # Gauss key =1,2,...,6 corresponding to the 15, 21, 31, 41, 51 and 61 point Gauss-Kronrod rules. 
        #The higher-order rules give better accuracy for smooth 
        #functions, while lower-order rules save time when the function contains 
        #local difficulties, such as discontinuities.
        double epsabs = 0.0, epsrel=1e-4
        #gsl_integration_workspace *w
        gsl_integration_cquad_workspace *w 
    w = gsl_integration_cquad_workspace_alloc (100)
    #w = gsl_integration_workspace_alloc (100)
    #printf("%f",param.kf)
    param.kf = kf
    param.qp = qp 
    param.sgn = sgn
    param.T = T
    param.pq1d = pq1d

    k_ini = kp-0.5 
    k_end = kp+0.5

    F.function = derivee
    F.params = &param
    
    gsl_integration_cquad(&F, k_ini, k_end, epsabs, epsrel, w, &result, &error, &nevals)
    gsl_integration_cquad_workspace_free (w) 

    # gsl_integration_qag(&F, k_ini, k_end, epsabs, epsrel, 100, key, w, &result, &error)
    # gsl_integration_workspace_free (w)  
     
    #gsl_integration_qng(&F, k_ini, k_end, epsabs, epsrel, &result, &error, &nevals)
    return result/float(pq1d.Np)

# cpdef void valeursbulles(double x, double T, pquasi1d param,
#     double[:,:,::1] IC, double[:,:,::1] IP, double[:,::1] IPsusc):
#     cdef:
#         int kp, qp, k1
#         int N2 = param.Np // 2
#         pquasi1d param1 = param
    
#     param1.Ef = param.Ef * exp(-x)

#     #for qp in prange(-N2, N2,nogil=True):
#     for k1 in range(-N2, N2):
#         IPsusc[k1,0] = vbulle(&derivsusc, &param1, float(k1), float(k1), 0.0, +1,T)
#         IPsusc[k1,1] = vbulle(&derivsusc, &param1, float(k1), float(k1), float(N2), +1,T)
#         for kp in range(-N2,N2):
#             for qp in range(-N2,N2):
#             #for qp in prange(-N2, N2,nogil=True):
#                 #print(f"kp= {kp}  k1={k1}  qp={qp}")
#                 IP[k1, kp, qp] = vbulle(&deriv, &param1, float(kp), float(k1), float(qp), +1,T)
#                 IC[k1, kp, qp] = vbulle(&deriv, &param1, float(kp), float(k1), float(qp), -1,T)

#     # for k1 in range(-N2,N2):
#     #     for kp in range(1,N2):
#     #         for qp in range(-N2,N2):
#     #             IP[k1, kp, qp] = IP[-k1, -kp, -qp] 
#     #             IC[k1, kp, qp] = IC[-k1, -kp, -qp]


cpdef void valeursbulles(double x, double T, pquasi1d param,
    double[:,:,::1] IC, double[:,:,::1] IP, double[:,::1] IPsusc):
    cdef:
        int kp, qp, k1
        int N2 = param.Np//2
        int Np = param.Np
        pquasi1d param1 = param
    
    param1.Ef = param.Ef * exp(-x)

    #for qp in prange(-N2, N2,nogil=True):

    #with nogil, parallel(num_threads=2):
    #    for k1 in prange(Np, schedule="dynamic"):        
    for k1 in range(Np):
        IPsusc[k1,0] = vbulle(&derivsusc, &param1, float(k1), float(k1), 0.0, +1,T)
        IPsusc[k1,1] = vbulle(&derivsusc, &param1, float(k1), float(k1), float(N2), +1,T)
        for kp in range(Np): 
            for qp in range(N2+1):
                    IP[k1, kp, qp] = vbulle(&deriv, &param1, float(kp), float(k1), float(qp), +1,T)
                    IC[k1, kp, qp] = vbulle(&deriv, &param1, float(kp), float(k1), float(qp), -1,T)
    for k1 in range(Np):
        for kp in range(Np):
            for qp in range(N2+1,Np):
                IP[k1, kp, qp] = IP[(Np-k1)%Np, (Np-kp)%Np, (Np-qp)%Np] 
                IC[k1, kp, qp] = IC[(Np-k1)%Np, (Np-kp)%Np, (Np-qp)%Np]


