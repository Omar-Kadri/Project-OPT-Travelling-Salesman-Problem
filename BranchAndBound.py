import numpy as np
import math
from sympy import reduced_totient
import os
import time


class noeud:
    def __init__(self,matrice,path,niv,i,j):
      self.cost = 0
      self.path=path.copy()
      self.niv =niv
      self.matrice =matrice.copy()
      self.ville=j     
      if(niv!=0):
       self.path.append(j)
       self.matrice[:,j]=math.inf
       self.matrice[i]=math.inf
       self.matrice[j][0]=math.inf

def reduire(matrix):
 cost=0
 for i in range(matrix.__len__()):
   if (not matrix[i].__contains__(0)) :
     if (min(matrix[i]) != math.inf):
       cost+=min(matrix[i])
       matrix[i] = matrix[i]-min(matrix[i])
 for j in range(matrix.__len__()):
   if (not matrix[:,j].__contains__(0)) :
     if (min(matrix[:,j]) != math.inf): 
       cost+=min(matrix[:,j])
       matrix[:,j]=matrix[:,j]-min(matrix[:,j])
 return cost

def TSP(matrice):
 N = matrice.__len__()
 listP = []
 path = []
 solution = False
 root = noeud(matrice,path,0,-1,0)
 root.cost = reduire(root.matrice)
 listP.append((root.cost,root))
 while listP:
   actif = listP.pop()[1]
   i = actif.ville
   print("niveau    ",actif.niv)
   if (actif.niv == N-1) and (matrice[i][0] != math.inf):
     solution = True
     actif.path.append(0)
     print(actif.path)
     return actif.cost
   for j in range(N):
     if (actif.matrice[i][j] != math.inf):
        fils = noeud(actif.matrice,actif.path,actif.niv + 1,i,j)
        fils.cost = actif.cost + actif.matrice[i][j] + reduire(fils.matrice)
        print("cout   ",fils.cost)
        listP.append((fils.cost,fils))
        listP.sort(reverse = True,key = lambda x: x[0])
 if not solution:
   print('Pas de solution')
   return math.inf


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
# start_time = time.time()
# print("le cout est de :",TSP(dist))
# print("\n\n _______________________temps execution : %s seconds " % (time.time() - start_time))

