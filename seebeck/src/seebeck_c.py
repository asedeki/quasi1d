import numpy as np
from seebeck.src.system import System
from seebeck.src.matricediffusion.matricediffusion import MatriceDiffusion as MDf
import math


class Seebeck:
    def __init__(self, parametres=[], g3=[], temperatures=[], energies=[], integration=False):
        self.temperatures = temperatures
        self.energies = energies
        if integration:
            print("a venir")
        else:
            self.MatriceDiffusion = MDf()
        self.parametres = parametres
        self.g3 = g3
        self.ScaterringTime = {}

    def get_collision_matrix(self, t, e):
        self.parametres["T"] = t
        self.parametres["E"] = e
        self.parametres["beta"] = 1.0 / t
        self.parametres["v"] = 2 * math.pi / float(self.parametres["Np"])
        self.MatriceDiffusion.initialisation(self.parametres, self.g3[t])
        return self.MatriceDiffusion.get_collision_matrix()

    def scatering_time(self, t, e):
        # import scipy
        # import scipy.linalg.lapack as lapack
        L = self.get_collision_matrix(t, e)
        try:
            b = np.ones([self.parametres["Np"]])
            #lu, piv, x, info = lapack.dgesv(L, b)
            #x, resid, rank, s = scipy.linalg.lstsq(L, b)
            x = np.linalg.solve(L, b)
            err = np.linalg.norm(np.dot(L, x) - b)

            s = sum(x)/math.cosh(e / 2.0 / t)
            return s, err
        except Exception as e:
            print(f"Probleme pour T={t}, E={e}")
            print(e)
            return np.nan, np.nan

    def coefficient_seebeck(self, t):
        # e = 0.01
        # ve = [-3*e, -2*e, -e, e, 2*e, 3*e]
        # cst = [-1.0/60.0, 3.0/20.0, -3.0/4.0 , 3.0/4.0,-3/20.0, 1.0/60.0]
        # seebeck = 0.0
        # for i in range(6):
        #     s1, err1 = self.scatering_time(t, ve[i])
        #     seebeck +=cst[i]*math.log(s1)
        # return seebeck/e , err1

        e = 0.01
        s1, err1 = self.scatering_time(t, -e)
        s2, err2 = self.scatering_time(t, e)
        seebeck = -math.log(s2 / s1)/2.0/e
        return seebeck, max(err1, err2)

    def set_temps_diffusion(self, temperatures=None):
        if temperatures is None:
            temperatures = self.temperatures
        else:
            self.temperatures = temperatures
        for t in temperatures:
            self.ScaterringTime[t] = {}
            for e in self.energies:
                self.ScaterringTime[t][e] = self.scatering_time(t, e)

    def __iter__(self):
        # import pdb; pdb.set_trace()
        for t in self.temperatures:
            for e in self.energies:
                s = self.ScaterringTime[t][e]
                yield t, e, s[0], s[1]

    def save_csv(self, file, keys=None, data=None):
        """
            Enregistre les valeurs contenus dans data avec comme
            cles keys dans le fichier file au format csv.
        """
        import csv
        import os
        file += f'_g3_{self.parametres["g3i"]}_tp2_{self.parametres["tp2"]}'
        file += f'_Np_{self.parametres["Np"]}'
        file = f"{file}.csv"

        if keys is None:
            keys = ["Temperature", "Energie", "TempsDiffusion", "Erreur"]
        if data is None:
            data = self

        typeWrite = "w"
        if os.path.isfile(file):
            typeWrite = "a"
        with open(file, typeWrite) as csv_file:
            writer = csv.writer(csv_file)
            if typeWrite == "w":
                writer.writerow(keys)
            [writer.writerow([f"{l}" for l in d])
             for d in data]
