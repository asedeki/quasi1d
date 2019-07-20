#!/usr/local/bin/python3
from datetime import datetime
import posix
import posixpath
import time
import numpy as np
import concurrent.futures as concfut
import csv
import pathlib
import os
import sys
sys.path.append("..")
import myoptik
from myoptik import scan
from seebeck.src.system import System
from seebeck.src.seebeck_c import Seebeck


def read_g_file(fileg="g.dat"):
    """
    Lecture du fichier contenant le flow des constantes d'interaction. 
    Renvoie les differents parametres du systeme ainsi que le flow de g3.
    Si ce n'est pas le cas, réecriture de toutes les données contenu dans le
    fichier dans un autre fichier plus facile d'acces.
    """
    f1 = pathlib.Path("interactions.npy")
    if not f1.is_file():
        sys = System(fileg)
        sys.set_interaction()
        data = {
            "p": sys.parametres,
            "g": sys.g
        }
        np.save("interactions", data)
        return sys.parametres, sys.g[3]
    else:
        data = np.load("interactions.npy")
        parametres = data[()]["p"]
        g1 = data[()]["g"][1]
        g2 = data[()]["g"][2]
        g3 = data[()]["g"][3]
        return parametres, g3


def run():
    Ts = _O.temperatures
    output = _O.output
    energies = _O.energies
    parametres, g3 = read_g_file()
    if Ts is None:
        Temps = np.array(list(g3.keys()), dtype=np.double)
        Temps = Temps[Temps >= 1]
    else:
        Temps = Ts
    s = Seebeck(parametres=parametres, g3=g3,
                temperatures=Temps, energies=energies,
                integration=_O.integration)
    if _O.seebeck:
        Qa = 0.00077  # fit article Myriam
        seebeck_coef = []
        for t in Temps:
            try:
                s_cof, err = s.coefficient_seebeck(t)
                seebeck_coef.append((t, (s_cof+Qa)*t, err))
                print(f"{t}\t{(s_cof+Qa)*t}")
                #seebeck_coef.append((t,s_cof, err))
            except Exception as e:
                print(e)
        s.save_csv(output+"_Seebeck", keys=["Temperature",
                                            "CoefficientSeebeck", "Erreur"], data=seebeck_coef)

    if _O.temps_diffusion:
        if Ts is not None:
            TT = []
            for t in Ts:
                m = np.min(np.abs(Temps-t))
                TT.append(Temps[np.abs(Temps-t) == m][0])
            Temps = TT
        s.set_temps_diffusion(temperatures=Temps)
        s.save_csv(output+"_diffusion")


def run_dir(d):
    d = d.split("/")
    d = "/".join(d[:-1])
    print(d)
    posix.chdir(d)
    run()
    posix.chdir(_O.dirini)


def main(_O):
    dirini = os.getcwd()
    _O.dirini = dirini
    if _O.scan:
        scan_files = scan(".")
        if(len(scan_files) > 0):
            scan_files.sort()
            if _O.parallel:
                with concfut.ProcessPoolExecutor() as executor:
                    future1 = executor.map(run_dir, scan_files)
            else:
                [run_dir(d) for d in scan_files]
    else:
        run()


if __name__ == '__main__':
    print(datetime.now().strftime('#$%Y-%m-%d %H:%M:%S'))
    _O = myoptik.opt()
    t1 = time.time()
    main(_O)
    print("#Temps Exec=%s" % round(time.time() - t1, 3))
    print(datetime.now().strftime('#$%Y-%m-%d %H:%M:%S'))
