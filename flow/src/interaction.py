'''
created on august 25, 2019
@author: Abdelouahab Sedeki
@company: Saida University (Algeria)
'''

import numpy as np
import cinteraction as cint
import concurrent.futures as concfut
from numba import jit
from loops import Loops

# todo reflechir a wrap(), %N ne donne que des valeurs positives


class Interaction:
    '''
        Class representing the goology modele interaction:
            g1, g2, umklapp g3
    '''

    def __init__(self, parameters):
        g1 = parameters["g1"]
        g2 = parameters["g2"]
        g3 = parameters["g3"]
        self.Np = parameters["Np"]
        self.initialisation(g1, g2, g3)
        self.loops = None

    def initialisation(self, g1: float, g2: float, g3: float,
                       g1_perp: float = 0, g2_perp: float = 0,
                       g3_perp: float = 0):
        Np = self.Np
        self.g1 = np.ones((Np, Np, Np), float) * g1
        self.g2 = np.ones((Np, Np, Np), float) * g2
        self.g3 = np.ones((Np, Np, Np), float) * g3

    def inipack(self):
        '''
            Summary line
            Extended description of the function

            Parameters:
                self: Interaction

            return:
                numpy array
                    representing concactenation of the interaction arrays (g1,g2,g3)
        '''
        Ng = self.Np**3
        y = np.concatenate(
            (self.g1.reshape((Ng,)), self.g2.reshape((Ng,)), self.g3.reshape((Ng,))))
        return y

    # @jit
    def pack(self, dg1, dg2, dg3):
        '''
            Summary line
            Extended description of the function

            Parameters:
            ---------------------------------------
                dg1: float numpy array
                dg2: float numpy array
                dg3: float numpy array 

            return:
            ----------------------------------------
                numpy array
                    representing concactenation of the arrays (dg1,dg2,dg3)
        '''
        Ng = self.Np**3
        # y = np.zeros(3*Ng)
        # y[:Ng] = dg1.reshape((Ng,))
        # y[Ng:2*Ng] = dg2.reshape((Ng,))
        # y[2*Ng:3*Ng] = dg3.reshape((Ng,))
        y = np.concatenate(
            (dg1.reshape((Ng,)), dg2.reshape((Ng,)), dg3.reshape((Ng,))))
        return y

    # @jit
    def unpack(self, y: np.ndarray):
        '''
            Summary line
            Extended description of the function

            Parameters:
            ---------------------------------------
                y: float numpy array

            return:
            ----------------------------------------
                None
                    execute the spliting of the y array (input) in three arrays 
                    self.g1, self.g2, self.g3
        '''
        Ng = self.Np**3
        self.g1 = y[:Ng].reshape((self.Np, self.Np, self.Np))
        self.g2 = y[Ng:2*Ng].reshape((self.Np,
                                      self.Np, self.Np))
        self.g3 = y[2*Ng:3*Ng].reshape((self.Np, self.Np, self.Np))

        # g1, g2, g3 = np.split(y, [Ng, 2*Ng])
        # self.g1 = g1.reshape((self.Np, self.Np, self.Np))
        # self.g2 = g2.reshape((self.Np, self.Np, self.Np))
        # self.g3 = g3.reshape((self.Np, self.Np, self.Np))

    def equations_rg_py(self, loops: Loops):
        '''
            It's a pure python function calculating 
            the RG derivatives of the goology modele
            interactions.            
            Parameters:
            ---------------------------------------
                loops: Loops class
                    contains the values of the Cooper and Peierls loops
                    values.

            return:
            ----------------------------------------
                numpy array

                    Calculate the rg derivatives for the g1, g2, g3
                    goology interaction. The result is concatenated in 
                    an numpy array via the pack function.
        '''
        Np = self.Np
        dg1 = np.zeros((Np, Np, Np), float)
        dg2 = np.zeros((Np, Np, Np), float)
        dg3 = np.zeros((Np, Np, Np), float)

        N2 = Np // 2
        for k1 in range(-N2, N2):
            for k2 in range(-N2, N2):
                qc = (k1+k2) % Np
                for k3 in range(-N2, N2):
                    qp = (k3-k2) % Np
                    qpp = (k1-k3) % Np
                    k4 = (k1+k2-k3) % Np
                    i = (k1, k2, k3)

                    for kp in range(-N2, N2):
                        IP = loops.Peierls[k2, kp, qp]
                        IP2 = loops.Peierls[k2, kp, -qp]
                        IC = loops.Cooper[k1, kp, qc]
                        IPP = loops.Peierls[k3, kp, qpp]

                        m1 = (k1, k2, kp)
                        m2 = (kp, (qc-kp) % Np, k3)
                        m3 = (k1, (kp-qp) % Np, kp)
                        m4 = (kp, k2, k3)
                        m5 = (k1, kp, (kp+qp) % Np)
                        m6 = (k2, (kp+qp) % Np, k3)
                        m7 = (k2, (kp+qp) % Np, kp)
                        m8 = ((kp+qp) % Np, k2, k3)
                        m9 = (k2, (kp+qpp) % Np, kp)
                        m10 = (k2, (kp+qpp) % Np, k4)
                        m11 = (k1, kp, (kp+qpp) % Np)
                        m12 = (k1, kp, k3)

                        dg1[i] += -0.5*((self.g2[m1]*self.g1[m2]+self.g1[m1]*self.g2[m2])*IC
                                        - (self.g2[m3]*self.g1[m4]+self.g1[m3]*self.g2[m4]
                                           - 2*self.g1[m3]*self.g1[m4])*IP2)
                        dg1[i] += 0.5*(self.g3[m5]*self.g3[m6] + self.g3[m7]*self.g3[k1, kp, k4]
                                       - 2.0*self.g3[m5]*self.g3[m8])*IP

                        dg2[i] += 0.5*(-(self.g2[m1]*self.g2[m2]+self.g1[m1]*self.g1[m2])*IC
                                       + self.g2[m3]*self.g2[m4] * IP2)
                        dg2[i] += 0.5*self.g3[k1, kp, k4]*self.g3[m6]*IP

                        dg3[i] += 0.5*(self.g3[m5]*self.g2[m7] + self.g3[k1, kp, k4]*self.g1[m7] +
                                       self.g2[m5]*self.g3[m7] + self.g1[m5]*self.g3[m6] -
                                       2*self.g1[m7]*self.g3[m5] - 2*self.g3[m7]*self.g1[m5])*IP
                        dg3[i] += 0.5*(self.g3[m12]*self.g2[m9] +
                                       self.g3[m10]*self.g2[m11])*IPP

        return self.pack(dg1, dg2, dg3)

    def equations_rg_cython(self, loops: Loops):
        '''
            A function calcuating the RG derivatives of the goology modele
            interactions. The main calculations are done by a cython function
            for time consideration.            
            Parameters:
            ---------------------------------------
                loops: Loops class
                    contains the values of the Cooper and Peierls loops
                    values.

            return:
            ----------------------------------------
                numpy array

                    Calculate the rg derivatives for the g1, g2, g3
                    goology interaction. The main calculations are done 
                    by a cython function "equations_rg" in the "cinteraction"
                    module.
                    The result is concatenated in an numpy array via the 
                    pack function.
        '''
        Np = self.Np
        dg1 = np.zeros((Np, Np, Np), float)
        dg2 = np.zeros((Np, Np, Np), float)
        dg3 = np.zeros((Np, Np, Np), float)
        cint.equations_rg(Np, self.g1, self.g2,
                          self.g3, loops.Peierls, loops.Cooper,
                          dg1, dg2, dg3)

        return self.pack(dg1, dg2, dg3)

    def rg_rows(self, vk1):
        Np = self.Np
        result = []
        for k1 in vk1:
            dg1 = np.zeros((Np, Np), float)
            dg2 = np.zeros((Np, Np), float)
            dg3 = np.zeros((Np, Np), float)
            cint.equations_rgcol(Np, k1, self.g1, self.g2,
                                 self.g3, self.loops.Peierls, self.loops.Cooper,
                                 dg1, dg2, dg3)
            result.append([k1, dg1, dg2, dg3])
        return result

    def equations_rg(self, loops: Loops):
        Np = self.Np
        dg1 = np.zeros((Np, Np, Np), float)
        dg2 = np.zeros((Np, Np, Np), float)
        dg3 = np.zeros((Np, Np, Np), float)
        self.loops = loops
        kvec = np.arange(Np)
        # for k1 in kvec:
        #     self.rg_row(k1)
        with concfut.ThreadPoolExecutor(max_workers=8) as executor:
            s = np.array_split(kvec, 8)
            jobs = [executor.submit(self.rg_rows, kk) for kk in s]

        for gg in jobs:
            gt = gg.result()
            for g in gt:
                dg1[g[0], :, :] = g[1]
                dg2[g[0], :, :] = g[2]
                dg3[g[0], :, :] = g[3]

        for k1 in range(-Np//2, Np//2):
            for k2 in range(1, Np//2):
                for k3 in range(-Np//2, Np//2):
                    dg1[k1, k2, k3] = dg1[-k1, -k2, -k3]
                    dg2[k1, k2, k3] = dg2[-k1, -k2, -k3]
                    dg3[k1, k2, k3] = dg3[-k1, -k2, -k3]

        return self.pack(dg1, dg2, dg3)


if __name__ == "__main__":

    param = {
        "Np": 4,
        "g1": 0.1,
        "g2": 0.2,
        "g3": 0.3
    }

    g = Interaction(param)
    g.g1[:, :, -1] = 0.984
    y = g.inipack()
    g.unpack(y)
    print(g.g1[:, :, -1])
