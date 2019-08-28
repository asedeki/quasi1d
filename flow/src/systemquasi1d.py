from scipy.integrate import odeint, solve_ivp
from interaction import Interaction
from bulles import Bulles
import sys
import warnings
import numpy as np


class System:
    def __init__(self, parametres):
        N_patche = parametres["Np"]
        self.param = parametres
        self.interaction = Interaction(N_patche)
        self.fonctionsreponse = None
        self.bulles = Bulles(parametres)
        self.T0 = 0.0

    def initialisation(self):
        self.interaction.initialisation(
            self.param["g1"], self.param["g2"], self.param["g3"])
        self.bulles.resset()

    def inipack(self):
        y = self.interaction.inipack()
        return y

    def pack(self):
        pass

    def unpack(self, y):
        self.interaction.unpack(y)

    def derivee(self, l, T):
        #print("in bulle l , T=", l, T)
        self.bulles.calculer(T, l)
        #print("out bulle")

        dy = self.interaction.equations_rg(self.bulles)
        return dy

    def evolutiontemperature(self, T):
        def rg(l, y):
            # input(type(y))
            self.unpack(y)
            dy = self.derivee(l, T)
            return dy
        l = [0, 100]
        y0 = self.inipack()
        # y = odeint(rg, y0, l)
        # self.unpack(y[1])

        sol = solve_ivp(rg, l, y0, t_eval=l, method='RK45',
                        rtol=1e-3, vectorized=False)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if (sol.success):
                self.T0 = T
                self.unpack(sol.y[:, -1])
            else:
                pass
                # print(sol.message)
                # print(sol)
        return sol.success

    def evolutionl(self, li, lf):
        def rg(l, y):
            self.unpack(y)
            dy = self.derivee(l, 1e-80)

            return dy
        l = [li, lf]
        y0 = self.inipack()
        with warnings.catch_warnings():
            sol = solve_ivp(rg, l, y0, t_eval=[li, lf])
            if (sol.success):
                self.unpack(sol.y[:, -1])
                return True
            else:
                # print(sol.message)
                return False
        # with warnings.catch_warnings():
        #     y = odeint(rg, y0, l, full_output=1)
        #     if y[1]["message"] == "Integration successful.":
        #         self.unpack(y[0][1])
        #         return True
        #     else:
        #         print(f'erreur dans odeint: {y[1]["message"]}')
        #         return False
