import numpy as np
from numpy import cos, sin
from types import MethodType
from bulles import Bulles
from interaction import Interaction


def rg_derivative_cbdw(self, buble: Bulles, couplage: Interaction):
    z = np.zeros((self.dim1, self.dim2))
    chi = np.zeros(self.dim1)

    Np = buble.IP.shape[0]
    i_sup = Np//2 - 1
    i_inf = -i_sup - 1
    kc = 0
    kp = i_inf
    qperp = [0, 1]
    

    for i in range(2):
        chi[i] = sum((self.z[i, :]**2) * buble.IPsusc[:, qperp[i]])

    for kpp in range(i_inf, i_sup):
        for i in range(i_inf, i_sup):
            kpp_kp = (kpp-kp) % Np
            i_kp = (i-kp) % Np
            Ipc = buble.IP[kpp, i, kc]
            Ipp = buble.IP[kpp, i, kp]

            z[0, kpp] -= 0.5*(2*couplage.g1[i, kpp, kpp]-couplage.g2[i, kpp, kpp]
                              - 2*couplage.g3[i, kpp, kpp] + couplage.g3[i, kpp, i])*self.z[0, i]*Ipc

            z[1, kpp] -= 0.5*(2*couplage.g1[i, kpp, kpp_kp]-couplage.g2[i, kpp, kpp_kp]
                              - 2*couplage.g3[i, kpp, kpp_kp] + couplage.g3[i, kpp, i_kp])*self.z[1, i]*Ipp

    return self.pack(z, chi)


def rg_derivative_csdw(self, buble: Bulles, couplage:Interaction):
    z = np.zeros((self.dim1, self.dim2))
    chi = np.zeros(self.dim1)
    Np = buble.IP.shape[0]
    i_sup = Np//2 - 1
    i_inf = -i_sup - 1

    kc = 0
    kp = i_inf
    qperp = [0, 1]

    for i in range(self.dim1):
        chi[i] = sum((self.z[i, :]**2) * buble.IPsusc[:, qperp[i]])

    for kpp in range(i_inf, i_sup):
        for i in range(i_inf, i_sup):

            kpp_kp = (kpp-kp) % Np
            i_kp = (i-kp) % Np
            Ipc = buble.IP[kpp, i, kc]
            Ipp = buble.IP[kpp, i, kp]

            z[0, kpp] -= 0.5*(2*couplage.g1[i, kpp, kpp] - couplage.g2[i, kpp, kpp]
                              + 2*couplage.g3[i, kpp, kpp]-couplage.g3[i, kpp, i])*self.z[0, i]*Ipc

            z[1, kpp] -= 0.5*(2*couplage.g1[i, kpp, kpp_kp]-couplage.g2[i, kpp, kpp_kp]
                              + 2*couplage.g3[i, kpp, kpp_kp]-couplage.g3[i, kpp, i_kp])*self.z[1, i]*Ipp

    return self.pack(z, chi)


def rg_derivative_sbdw(self, buble: Bulles, couplage: Interaction):
    z = np.zeros((self.dim1, self.dim2))
    chi = np.zeros(self.dim1)
    Np = buble.IP.shape[0]
    i_sup = Np//2 - 1
    i_inf = -i_sup - 1
    kc = 0
    kp = i_inf
    qperp = [0,1]
    for i in range(self.dim1):
        chi[i] = sum((self.z[i, :]**2)* buble.IPsusc[:, qperp[i]])

    for kpp in range(i_inf, i_sup):
        for i in range(i_inf, i_sup):
            kpp_kp = (kpp-kp) % Np
            i_kp = (i-kp) % Np
            Ipc = buble.IP[kpp, i, kc]
            Ipp = buble.IP[kpp, i, kp]
            z[0,kpp] += 0.5*(couplage.g2[i,kpp,kpp] - couplage.g3[i,kpp,i])*self.z[0,i]*Ipc
            z[1,kpp] += 0.5*(couplage.g2[i,kpp,kpp_kp] - couplage.g3[i,kpp,i_kp])*self.z[1,i]*Ipp

    return self.pack(z, chi)

def rg_derivative_ssdw(self, buble: Bulles, couplage: Interaction):
    #print(self.dim1, self.dim2)
    z = np.zeros((self.dim1, self.dim2))
    chi = np.zeros(self.dim1)
    Np = buble.IP.shape[0]
    i_sup = Np//2 - 1
    i_inf = -i_sup - 1
    kc = 0
    kp = i_inf 
    qperp=[0, 1]

    for i in range(self.dim1):
        chi[i] = sum((self.z[i, :]**2) * buble.IPsusc[:, qperp[i]])

    
    for kpp in range(i_inf, i_sup):
        for i in range(i_inf, i_sup):
            kpp_kp = (kpp-kp) % Np
            i_kp = (i-kp) % Np
            Ipc = buble.IP[kpp, i, kc]
            Ipp = buble.IP[kpp, i, kp]
            z[0,kpp] += 0.5*(couplage.g2[i,kpp,kpp] + couplage.g3[i,kpp,i])*self.z[0,i]*Ipc
            z[1,kpp] += 0.5*(couplage.g2[i,kpp,kpp_kp] + couplage.g3[i,kpp,i_kp])*self.z[1,i]*Ipp

    return self.pack(z, chi)
def rg_derivative_supra_singlet(self, buble: Bulles, couplage: Interaction):
    z = np.zeros((self.dim1, self.dim2))
    chi = np.zeros(self.dim1)

    Np = buble.IP.shape[0]
    i_sup = Np//2 - 1
    i_inf = -i_sup - 1
    kc = 0
    for i in range(self.dim1):
        chi[i] = sum((self.z[i, : ]**2) * buble.IC[0, :, 0])

    kp = -i_inf
    for kpp in range(i_inf, i_sup):
        for i in range(i_inf, i_sup):
            mkpp = (-kpp) % Np
            Ic = buble.IC[kpp, i, kc]
            for j in range(self.dim1):
                z[j,kpp] -= 0.5*(couplage.g1[kpp,mkpp,i]
                            + couplage.g2[kpp,mkpp,i])*self.z[j,i]*Ic
    return self.pack(z, chi)

def rg_derivative_supra_triplet(self, buble: Bulles, couplage: Interaction):
    z = np.zeros((self.dim1, self.dim2))
    chi = np.zeros(self.dim1)
    Np = buble.IP.shape[0]
    i_sup = Np//2 - 1
    i_inf = -i_sup - 1
    kc = 0
    kp = -i_inf

    for i in range(self.dim1):
        chi[i] = sum((self.z[i, : ]**2) * buble.IC[0, :, 0])

    for kpp in range(i_inf, i_sup):
        for i in range(i_inf, i_sup):
            mkpp = (-kpp) % Np
            Ic = buble.IC[kpp, i, kc]
            for j in range(self.dim1):
                z[j, kpp] += 0.5*(couplage.g1[kpp, mkpp, i]
                                - couplage.g2[kpp, mkpp, i])*self.z[j, i]*Ic

    return self.pack(z, chi)

class ResponseFunction():
    __TYPE_RESPONSE_FUNCTION = {
        "csdw": {"dim1": 2, "func_ini": [], "rg": rg_derivative_csdw},
        "cbdw": {"dim1": 2, "func_ini": [], "rg": rg_derivative_cbdw},
        "ssdw": {"dim1": 2, "func_ini": [], "rg": rg_derivative_ssdw},
        "sbdw": {"dim1": 2, "func_ini": [], "rg": rg_derivative_sbdw},
        "supra_triplet": {"dim1": 4, "func_ini": ["", "1*sin", "2*cos", "1*cos"],
                          "rg": rg_derivative_supra_triplet},
        "supra_singlet": {"dim1": 5, "func_ini": ["", "1*sin", "1*cos", "2*sin", "3*cos"],
                          "rg": rg_derivative_supra_singlet}
    }

    def __init__(self, N_patche):
        self.dim2 = N_patche
        self.Neq = None
        self.response_type = self.__TYPE_RESPONSE_FUNCTION.keys()

    def add_response_function_type(self, name: str):
        rp_type = self.__TYPE_RESPONSE_FUNCTION[name]
        self.dim1 = rp_type["dim1"]
        self.Neq = self.dim1*(self.dim2 + 1)
        self.initialize(rp_type["func_ini"])
        self.rg_derivative = MethodType(rp_type["rg"], self)

    def pack(self, z: np.ndarray, chi: np.ndarray):
        y = np.zeros(self.Neq, float)
        y[:self.dim1] = chi
        y[self.dim1:] = z.reshape(self.dim2*self.dim1)
        return y

    def unpack(self, y: np.ndarray):
        self.chi = y[:self.dim1]
        self.z = y[self.dim1:].reshape(self.dim1, self.dim2)

    def initialize(self, string_function: str):
        self.z = np.zeros((self.dim1, self.dim2), float)
        self.chi = np.zeros(self.dim2, float)
        v = 2*np.pi/float(self.dim2)
        k_perp = np.arange(self.dim2) * v
        if len(string_function) == 0:
            self.z[:, :] = 1.0
        else:
            self.z[0, :] = 1.0
            for i in range(1, self.dim1):
                list_function = string_function[i].split("*")
                CONSTANTE = float(list_function[0])
                function = list_function[1]
                self.z[i, :] = np.sqrt(2) * eval(function)(CONSTANTE * k_perp)


if __name__ == "__main__":
    rf = ResponseFunction(8)
    g = Interaction(8, 0.1,0.2,0.3)
    
    r = {"tp": 200, "tp2": 20, "Ef": 3000, "Np": 8, "g1": 0.2}
    b = Bulles(r)
    b.resset()
    b.calculer(1, 2)
    rf.add_response_function_type("csdw")
    y = rf.rg_derivative(b, g)
    print(y)

    for ty in rf.response_type:
        rf2 = ResponseFunction(8)
        print(ty)
        rf2.add_response_function_type(ty)
        y = rf2.rg_derivative(b, g)
        print(y)
