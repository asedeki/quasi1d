from scipy.integrate import odeint
from interaction import Interaction
from bulles import Bulles


class System:
    def __init__(self, N_patche):
        self.interaction = Interaction(N_patche)
        self.fonctionsreponse = None
        self.bulles = Bulles(N_patche)

    def inipack(self):
        y = self.interaction.inipack()
        return y

    def pack(self):
        pass

    def unpack(self, y):
        self.interaction.unpack(y)

    def evolution(self, T):
        def rg(y, l):
            self.interaction.unpack(y)
            self.bulles.calculer(T, l)
            self.interaction.equations_rg(self.bulles)
            return self.interaction.dg
        l = [0, 100]
        y0 = self.inipack()
        y = odeint(rg, y0, l)
        self.unpack(y[1])
