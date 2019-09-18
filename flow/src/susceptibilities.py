import numpy as np
from numpy import cos, sin
from types import MethodType
from loops import Loops
from interaction import Interaction


def rg_derivative_cbdw(self, loops: Loops, couplage: Interaction):
    z = np.zeros((self.dim1, self.dim2))
    chi = np.zeros(self.dim1)

    Np = loops.Peierls.shape[0]
    i_sup = Np//2 - 1
    i_inf = -i_sup - 1
    kc = 0
    kp = i_inf
    qperp = [0, 1]

    for i in range(2):
        chi[i] = sum((self.vertex[i, :]**2) * loops.Peierls_susc[:, qperp[i]])

    for kpp in range(i_inf, i_sup):
        for i in range(i_inf, i_sup):
            kpp_kp = (kpp-kp) % Np
            i_kp = (i-kp) % Np
            Ipc = loops.Peierls[kpp, i, kc]
            Ipp = loops.Peierls[kpp, i, kp]

            z[0, kpp] -= 0.5*(2*couplage.g1[i, kpp, kpp]
                              - couplage.g2[i, kpp, kpp]
                              - 2*couplage.g3[i, kpp, kpp]
                              + couplage.g3[i, kpp, i]) * self.vertex[0, i]*Ipc

            z[1, kpp] -= 0.5*(2*couplage.g1[i, kpp, kpp_kp]
                              - couplage.g2[i, kpp, kpp_kp]
                              - 2*couplage.g3[i, kpp, kpp_kp]
                              + couplage.g3[i, kpp, i_kp])*self.vertex[1, i]*Ipp

    return self.pack(z, chi)


def rg_derivative_csdw(self, loops: Loops, couplage: Interaction):
    z = np.zeros((self.dim1, self.dim2))
    chi = np.zeros(self.dim1)
    Np = loops.Peierls.shape[0]
    i_sup = Np//2 - 1
    i_inf = -i_sup - 1

    kc = 0
    kp = i_inf
    qperp = [0, 1]

    for i in range(self.dim1):
        chi[i] = sum((self.vertex[i, :]**2) * loops.Peierls_susc[:, qperp[i]])

    for kpp in range(i_inf, i_sup):
        for i in range(i_inf, i_sup):

            kpp_kp = (kpp-kp) % Np
            i_kp = (i-kp) % Np
            Ipc = loops.Peierls[kpp, i, kc]
            Ipp = loops.Peierls[kpp, i, kp]

            z[0, kpp] -= 0.5*(2*couplage.g1[i, kpp, kpp] - couplage.g2[i, kpp, kpp]
                              + 2*couplage.g3[i, kpp, kpp]-couplage.g3[i, kpp, i])*self.vertex[0, i]*Ipc

            z[1, kpp] -= 0.5*(2*couplage.g1[i, kpp, kpp_kp]-couplage.g2[i, kpp, kpp_kp]
                              + 2*couplage.g3[i, kpp, kpp_kp]-couplage.g3[i, kpp, i_kp])*self.vertex[1, i]*Ipp

    return self.pack(z, chi)


def rg_derivative_sbdw(self, loops: Loops, couplage: Interaction):
    z = np.zeros((self.dim1, self.dim2))
    chi = np.zeros(self.dim1)
    Np = loops.Peierls.shape[0]
    i_sup = Np//2 - 1
    i_inf = -i_sup - 1
    kc = 0
    kp = i_inf
    qperp = [0, 1]
    for i in range(self.dim1):
        chi[i] = sum((self.vertex[i, :]**2) * loops.Peierls_susc[:, qperp[i]])

    for kpp in range(i_inf, i_sup):
        for i in range(i_inf, i_sup):
            kpp_kp = (kpp-kp) % Np
            i_kp = (i-kp) % Np
            Ipc = loops.Peierls[kpp, i, kc]
            Ipp = loops.Peierls[kpp, i, kp]
            z[0, kpp] += 0.5*(couplage.g2[i, kpp, kpp] -
                              couplage.g3[i, kpp, i])*self.vertex[0, i]*Ipc
            z[1, kpp] += 0.5*(couplage.g2[i, kpp, kpp_kp] -
                              couplage.g3[i, kpp, i_kp])*self.vertex[1, i]*Ipp

    return self.pack(z, chi)


def rg_derivative_ssdw(self, loops: Loops, couplage: Interaction):
    #print(self.dim1, self.dim2)
    z = np.zeros((self.dim1, self.dim2))
    chi = np.zeros(self.dim1)
    Np = loops.Peierls.shape[0]
    i_sup = Np//2 - 1
    i_inf = -i_sup - 1
    kc = 0
    kp = i_inf
    qperp = [0, 1]

    for i in range(self.dim1):
        chi[i] = sum((self.vertex[i, :]**2) * loops.Peierls_susc[:, qperp[i]])

    for kpp in range(i_inf, i_sup):
        for i in range(i_inf, i_sup):
            kpp_kp = (kpp-kp) % Np
            i_kp = (i-kp) % Np
            Ipc = loops.Peierls[kpp, i, kc]
            Ipp = loops.Peierls[kpp, i, kp]
            z[0, kpp] += 0.5*(couplage.g2[i, kpp, kpp] +
                              couplage.g3[i, kpp, i])*self.vertex[0, i]*Ipc
            z[1, kpp] += 0.5*(couplage.g2[i, kpp, kpp_kp] +
                              couplage.g3[i, kpp, i_kp])*self.vertex[1, i]*Ipp

    return self.pack(z, chi)


def rg_derivative_supra_singlet(self, loops: Loops, couplage: Interaction):
    z = np.zeros((self.dim1, self.dim2))
    chi = np.zeros(self.dim1)

    Np = loops.Peierls.shape[0]
    i_sup = Np//2 - 1
    i_inf = -i_sup - 1
    kc = 0
    for i in range(self.dim1):
        chi[i] = sum((self.vertex[i, :]**2) * loops.Cooper[0, :, 0])

    kp = -i_inf
    for kpp in range(i_inf, i_sup):
        for i in range(i_inf, i_sup):
            mkpp = (-kpp) % Np
            Ic = loops.Cooper[kpp, i, kc]
            for j in range(self.dim1):
                z[j, kpp] -= 0.5*(couplage.g1[kpp, mkpp, i]
                                  + couplage.g2[kpp, mkpp, i])*self.vertex[j, i]*Ic
    return self.pack(z, chi)


def rg_derivative_supra_triplet(self, loops: Loops, couplage: Interaction):
    z = np.zeros((self.dim1, self.dim2))
    chi = np.zeros(self.dim1)
    Np = loops.Peierls.shape[0]
    i_sup = Np//2 - 1
    i_inf = -i_sup - 1
    kc = 0
    kp = -i_inf

    for i in range(self.dim1):
        chi[i] = sum((self.vertex[i, :]**2) * loops.Cooper[0, :, 0])

    for kpp in range(i_inf, i_sup):
        for i in range(i_inf, i_sup):
            mkpp = (-kpp) % Np
            Ic = loops.Cooper[kpp, i, kc]
            for j in range(self.dim1):
                z[j, kpp] += 0.5*(couplage.g1[kpp, mkpp, i]
                                  - couplage.g2[kpp, mkpp, i])*self.vertex[j, i]*Ic

    return self.pack(z, chi)


class Scusceptibility():
    __SUSCEPTIBILITY_TYPE = {
        "csdw": {"dim1": 2,
                 "func_ini": [],
                 "rg": rg_derivative_csdw,
                 "type": ["Site_Charge"]},
        "cbdw": {"dim1": 2,
                 "func_ini": [],
                 "rg": rg_derivative_cbdw,
                 "type": ["Bond_Charge"]},
        "ssdw": {"dim1": 2,
                 "func_ini": [],
                 "rg": rg_derivative_ssdw,
                 "type": ["Site_Spin"]},
        "sbdw": {"dim1": 2,
                 "func_ini": [],
                 "rg": rg_derivative_sbdw,
                 "type": ["Bond_Spin"]},
        "supra_triplet": {"dim1": 4,
                          "func_ini": ["", "1*sin", "2*cos", "1*cos"],
                          "rg": rg_derivative_supra_triplet,
                          "type": ["px", "py", "h", "f"]},
        "supra_singlet": {"dim1": 5,
                          "func_ini": ["", "1*sin", "1*cos", "2*sin", "3*cos"],
                          "rg": rg_derivative_supra_singlet,
                          "type": ["s", "dxy", "dx2y2", "g", "i"]}
    }

    def __init__(self, parameters: dict):
        self.dim2 = parameters["Np"]
        self.Neq = None
        self.susceptibilities = self.__SUSCEPTIBILITY_TYPE.keys()

    def set_susceptibility_type(self, name: str):
        susceptibility_type = self.__SUSCEPTIBILITY_TYPE[name]
        self.dim1 = susceptibility_type["dim1"]
        self.Neq = self.dim1*(self.dim2 + 1)
        self.initialize(susceptibility_type["func_ini"])
        self.rg_derivative = MethodType(susceptibility_type["rg"], self)
        self.types = susceptibility_type["type"]

    def pack(self, z: np.ndarray, chi: np.ndarray):
        y = np.zeros(self.Neq, float)
        y[:self.dim1] = chi
        y[self.dim1:] = z.reshape(self.dim2*self.dim1)
        return y

    def unpack(self, y: np.ndarray):
        self.scusceptibility = y[:self.dim1]
        self.vertex = y[self.dim1:].reshape(self.dim1, self.dim2)

    def initialize(self, string_function: list):
        self.vertex = np.zeros((self.dim1, self.dim2), float)
        self.scusceptibility = np.zeros(self.dim2, float)
        v = 2*np.pi/float(self.dim2)
        k_perp = np.arange(self.dim2) * v
        if len(string_function) == 0:
            self.vertex[:, :] = 1.0
        else:
            self.vertex[0, :] = 1.0
            for i in range(1, self.dim1):
                list_function = string_function[i].split("*")
                CONSTANTE = float(list_function[0])
                function = list_function[1]
                self.vertex[i, :] = np.sqrt(
                    2) * eval(function)(CONSTANTE * k_perp)


class Susceptibilities:
    def __init__(self, parameters: dict, susceptibilities_nmes: list):
        self.susceptibilities = {}
        self.susceptibilities_names = susceptibilities_nmes
        self.susceptibilities_names.sort()
        self.initialize(parameters)
        self.Neq = self.get_variables_number()

    def initialize(self, parameters: dict):
        for name in self.susceptibilities_names:
            susc = Scusceptibility(parameters)
            susc.set_susceptibility_type(name)
            self.susceptibilities[name] = susc

    def get_variables_number(self):
        Neq = 0
        for susc in self.susceptibilities.values():
            Neq += susc.Neq
        return Neq

    def derivative(self, loops: Loops, interaction: Interaction):
        derivs = {}
        for name, susc in self.susceptibilities.items():
            derivs[name] = susc.rg_derivative(loops, interaction)
        return self.pack(derivs)

    def pack(self, derivs: dict):
        y = np.zeros(self.Neq, float)
        indice = 0
        for name in self.susceptibilities_names:
            neq = self.susceptibilities[name].Neq
            y[indice:indice + neq] = derivs[name]
            indice += neq
        return y

    def unpack(self, y: np.ndarray):
        indice = 0
        for name in self.susceptibilities_names:
            neq = self.susceptibilities[name].Neq
            self.susceptibilities[name].unpack(y[indice:indice+neq])
            indice += neq


if __name__ == "__main__":

    g = Interaction(8, 0.1, 0.2, 0.3)
    r = {"tp": 200, "tp2": 20, "Ef": 3000, "Np": 8, "g1": 0.2}
    rf = Scusceptibility(r)
    b = Loops(r)
    b.resset()
    b.calculer(1, 2)
    rf.set_susceptibility_type("csdw")
    y = rf.rg_derivative(b, g)
    print(y)

    for ty in rf.susceptibilities:
        rf2 = Scusceptibility(r)
        print(ty)
        rf2.set_susceptibility_type(ty)
        y = rf2.rg_derivative(b, g)
        print(y)
