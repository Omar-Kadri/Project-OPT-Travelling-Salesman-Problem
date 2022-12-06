import HeuristicMoy as Moy
import HeuristicReduct as Red
import HeuristicPPVoisin as PPV
#import AlgorithmeGenetique as AG
import RechercheTabou as RT
import BranchAndBound as bb
import time
import numpy as np

NB_TOWNS = 48
f = open("dataset2.tsp", "r")
coord = np.empty((NB_TOWNS, 2))
#f.seek(128)
for i in range(NB_TOWNS):
    line = f.readline()
    l = line.split()
    coord[i][0] = float(l[1])
    coord[i][1] = float(l[2])
dist = RT.calculate_dist(coord,NB_TOWNS)
start_time = time.time()
print("Le cout est de :", RT.recherche_tabou(1000,dist,NB_TOWNS))
print("\n\n _______________________temps execution : %s seconds " % (time.time() - start_time))