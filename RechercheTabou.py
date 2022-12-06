from copy import deepcopy
from dis import dis
from operator import index
import time
import numpy as np
from itertools import combinations
import math
import random
import os 
from scipy import rand

from HeuristicMoy import NB_TOWNS
 
taille = 100

def permut(A, b, c):
    tmp = A[b] 
    A[b] = A[c]
    A[c] = tmp 

def solution_realisable(chemin):
    # realisable = True
    # for i in range(NB_TOWNS):
    #     if dist[i,i+1] == math.inf:
    #         realisable = False
    #         break
    return True

def cout(chemin,dist,NB_TOWNS):
    res = 0 
    for i in range(0,NB_TOWNS):
        
        res += dist[chemin[i]][chemin[i+1]]
    return res


def random_chemin(NB_TOWNS): 
    l = list(range(NB_TOWNS))
    random.shuffle(l)
    l.insert(0,0)
    l.append(0)


    return l 



def voisinage(chemin ,boucle, dist,NB_TOWNS):
    
    voisins = list()
    couts = list()
    for i in range(boucle) : 
        tmp = deepcopy(chemin)
       
        random1 = random.randint(1, NB_TOWNS-1)
        random2 = random.randint(1, NB_TOWNS-1)
        
        indx1 = tmp.index(random1)
        indx2 = tmp.index(random2)
        permut(tmp,indx1,indx2)
        if solution_realisable(tmp) == 1 :
            voisins.append(tmp) 
    for i in voisins : 
        couts.append(cout(i,dist,NB_TOWNS))
    return voisins , couts 
def recherche_tabou(iteration,dist,NB_TOWNS):
    # solution lawla 
    # s0 = [0, 7, 10, 8, 9, 1, 13, 11, 5, 6, 12, 4, 3, 2, 0]
    s0 = random_chemin(NB_TOWNS)
    s = s0 
    lt = list() 
    n = 0 
    sm = s0
    while( n < iteration):
        x , y = voisinage(sm , 50, dist, NB_TOWNS)
        ind = np.argmin(y)
        sm = x[ind] 
       
        if sm not in lt : 
            if len(lt) > taille : 
                lt.pop(0) 
                lt.append(sm)
            else:
                lt.append(sm)
            if cout(sm,dist,NB_TOWNS) < cout(s,dist,NB_TOWNS) :  s = sm 
        n+=1

    return s , cout(s,dist,NB_TOWNS)

def calculate_dist(coord, NB_TOWNS):
  dist = np.zeros((NB_TOWNS,NB_TOWNS))
  for i in range(NB_TOWNS):
      x1 = coord[i][0]
      y1 = coord[i][1]
      for j in range(NB_TOWNS):
          x2 = coord[j][0]
          y2 = coord[j][1]
          if i == j:
              dist[i][j] = math.inf
          else:
              dist[i][j] = math.sqrt(pow((x2-x1),2) + pow((y2-y1),2))
  return dist
                
# f = open("dataset.tsp", "r")
# coord = np.empty((NB_TOWNS, 2))
# f.seek(128)
# for i in range(NB_TOWNS):
#     line = f.readline()
#     l = line.split()
#     coord[i][0] = float(l[1])
#     coord[i][1] = float(l[2])
# calculate_dist(coord)
# bruh=0
# min=0
# ind=0
# for i in range(10,100):
#     hit = 0
#     for j in range(50):
#         tmp=recherche_tabou(i)[1]
#         if(tmp<=33):
#             min = tmp
#             hit += 1
#     if(hit>bruh):
#         bruh=hit
#         ind=i
#     print("cout=",min,"   pour i=",i,"   hit=",hit)
#     if(hit == 40):
#         break

# print(min,ind,hit)