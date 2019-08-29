import numpy as np

def tognuplot(file,title="", genre="pm"):
    f=open(file+".gp","w")
    f.write("#!/usr/bin/gnuplot -persist\n")
    f.write("set grid \n")
    f.write("set format z '%3.1f' \n")
    f.write("set xtics (\"{/Symbol p}\"1,\"-{/Symbol p}\"-1,'0'0)\n")
    f.write("set ytics (\"{/Symbol p}\"1,\"-{/Symbol p}\"-1,'0'0)\n")
    
    f.write("set view 60,30\n")
    if genre[:1]=="c":
        f.write("set contour base\n")
    elif genre[:1]=="p":
        f.write("unset ztics \n")
        f.write("set nokey\n")
        f.write("set pm3d at b\n")
    f.write("set term post enhanced color 18\n")
    zz = "1:2:($3+$4)"
    zlab= zz.split(":");zlab=zlab[2][1:-1]
    zlab=zlab.replace("$3","g_1")
    zlab=zlab.replace("$4","g_2")
    zlab=zlab.replace("$5","g_3")
    f.write(f'set title {title}\n')
    f.write('set zlabel "%s"\n'%zlab) 
    f.write("set xlabel '{/URWPalladioL-Ital k}_{perp}'\n")
    f.write("set ylabel '{/URWPalladioL-Ital k^{,}}_{perp}'\n")
    f.write('set output "%s.ps" \n'%file)
    f.write("set nokey \n")
    f.write(f"sp '{file}.dat' u 1:2:(-$3-$4) w l lw 2\n")
    f.close()

def getlines(r):
    N = r["g1"].shape[0]//2
    v = np.pi/float(N)
    line = ""
    for i in range(-N,N+1):
        for j in range(-N,N+1):
            line += f'{i*v}\t{j*v}\t{r["g1"][i,-i,j]}\t{r["g2"][i,-i,j]}\t{r["g3"][i,-i,j]}\n'
        line += "\n"
    return line

def todictionnary(d ="data/"):
    import pathlib
    path = pathlib.Path(d)
    files = path.rglob('*.npz')
    result = {}
    for f in files:
        r = np.load(f)["g"]
        # input(r["g1"])
        result = r.item()
        #result[float(r["T"])] = {"g1": r["g1"], "g2": r["g2"], "g3": r["g3"]}
    Temp = list(result.keys())
    #Temp.sort()
    #Temp = Temp[-1::-1]
    return Temp, result
if __name__ == "__main__":
    import os
    from tempfile import NamedTemporaryFile
    Temps, result = todictionnary()
    pdfs = ""
    for fg in Temps:
        tmpfile = NamedTemporaryFile().name
        l = getlines(result[fg])
        f = open(f"{tmpfile}.dat","w")
        f.write(l)
        f.close()
        tognuplot(f"{tmpfile}", title=f"'T={fg}'")
        os.system(f'gnuplot {tmpfile}.gp')
        os.system(f'ps2pdf {tmpfile}.ps {tmpfile}.pdf')
        pdfs += f'{tmpfile}.pdf '
    os.system(f"pdfunite {pdfs} out.pdf")

