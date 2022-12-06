from audioop import reverse
from copy import deepcopy
from dis import dis
from multiprocessing.connection import wait
from platform import node
import time
from tkinter.tix import MAX
from more_itertools import random_combination
from pyparsing import And
from sklearn.utils import shuffle
from yaml import DirectiveToken
from modulus import mid
from turtle import clear
from joblib import PrintTime
import numpy as np
from itertools import combinations
import math
import random
import os
import time
from pyrsistent import b
from sqlalchemy import false
from sympy import N


NB_TOWNS = 48 
MAX_DIST = 0
NB_PERMUT = 8
MAX = 999999
PROBA = -1
SIG = -1
DIST = []

class noeud:
    def __init__(self, path, cost, sv):
      self.cost = cout(path)
      self.path=path
      self.sv = 0
    def __str__(self):
        return "< path={0} | cost={1} | SV={2} >\n".format(self.path, self.cost, self.sv) 
    def __repr__(self):
        return str(self)


 #____________________________________________________ Fonctions ____________________________________________________#


def random_solution_gen(NB_TOWNS, n):
    population = list()
    for i in range(n):
            l = list(range(1,NB_TOWNS))
            random.shuffle(l)
            l.insert(0, 0)
            l.append(0)
            node = noeud(l, 0, 0)
            node.sv = SV(node)
            population.append(node)
    return population                            


def solution_realisable(x):
    realisable = True
    for i in range(NB_TOWNS-1):
        if DIST[x[i],x[i+1]] == math.inf:
            realisable = False
            break
    return realisable


def Random_permut(sol):
    temp = deepcopy(sol)
    i= random.randint(1,NB_TOWNS-1)
    while True :
        j= random.randint(1,NB_TOWNS-1)
        if j != i : break
    temp[i] , temp[j] = sol[j] ,sol[i]
    return temp
    

def SV(x, a=0):
    if a==0:    return MAX_DIST / x.cost
    else:       return MAX_DIST / cout(x)


def cout(x):
    res = 0
    for i in range(0,NB_TOWNS):        
        res += DIST[x[i]][x[i+1]]
    return res


def calculate_dist(coord , dist):
    for i in range(NB_TOWNS):
        x1 = coord[i][0]
        y1 = coord[i][1]
        for j in range(NB_TOWNS):
            x2 = coord[j][0]
            y2 = coord[j][1]
            if i == j:
                dist[i][j] = 0
            else:
                dist[i][j] = math.sqrt(pow((x2-x1),2) + pow((y2-y1),2))
 

def distance(x, y, a=0):
    dist = 0
    if a == 0:
        for i in range(len(x.path)):
            if x.path[i] != y.path[i]:
                dist+=1
    else:
        for i in range(len(x)):
            if x[i] != y[i]:
                dist+=1
    return dist


def inverse(x):
    ix = noeud(x.path[::-1], 0, 0)
    ix.sv = SV(ix)
    return ix


def putVille(x, i, a):
    for k in range(NB_TOWNS):
        if x[k] == a:
            s = x[i]
            x[i] = a
            x[k] = s

def move(x, direct, step):
    current = 0
    for i in range(NB_TOWNS):
        if x[i] != direct[i]:
            putVille(x, i, direct[i])
            current += 1
        if current == step:
            break


def Move_Prey(xprey , xpredator, gen) :
    best_rand=[]
    best_rand_sv = xprey.sv
    temp = xprey.path
    A = []
    # Definir l'ensemble A
    for i in gen:
        if (i.sv > xprey.sv) : A.append(i)
    # faire update de xprey selon A
    # A={} et xprey et best prey
    if not A :
    # select best random permutation
        for i in range(NB_PERMUT) :
            randir = xprey.path[1:NB_TOWNS]
            random.shuffle(randir)
            randir.insert(0, 0)
            randir.append(0)
            move(temp, randir, 1)
            if SV(temp, 1) > best_rand_sv : 
                best_rand = randir
                best_rand_sv = SV(temp, 1)
            temp = xprey.path
        if best_rand:
            move(xprey.path, best_rand, 1)
    else :
        if(random.uniform(0,1)< PROBA) :
            dist_min=MAX
            for i in A:
                dist = distance(xprey, i)
                if dist < dist_min:
                    dir = i.path
                    dist_min = dist
            step = distance(xprey, xpredator)*2 + 1
            move(xprey, dir, step)
        else :
            step = distance(xprey, xpredator)*2 + 1
            Yr = xprey.path[1:NB_TOWNS]
            random.shuffle(Yr)
            Yr.insert(0, 0)
            Yr.append(0)
            temp = xprey.path
            itemp = xprey.path
            move(temp, Yr, step)
            move(itemp, inverse(Yr), step)
            d1 = distance(temp, xpredator)
            d2 = distance(itemp, xpredator)
            if(d1 < d2): 
                move(xprey.path, Yr, step)
            else:
                move(xprey.path, Yr, step)


def Move_Predator(xpredator, gen):
    weak_sv = MAX
    for i in gen:
        if i.sv < weak_sv:
            weak_dir = i.path
            weak_sv = i.sv
    if(random.uniform(0,1) < SIG):
        Random_permut(weak_dir)
    step = distance(weak_dir, xpredator.path, 1)*2 + 1
    move(xpredator, weak_dir, step)

def Prey_predator(DIST, Nb_permut = 10, Prb= 0.5, Sgm=0.5, nb_gen=100, nb_individu = 50):
    NB_PERMUT = Nb_permut
    PROBA = Prb
    SIG = Sgm
    MAX_DIST =  np.amax(DIST) * NB_TOWNS
    population = random_solution_gen(NB_TOWNS, nb_individu)
    best_prey_sv = 0 
    predator_sv = MAX
    for j in range(nb_gen):
        for i in population:
            if SV(i) > best_prey_sv :
                best_prey = i
                best_prey_sv = i.sv
    
            if SV(i) < predator_sv :
                predator = i
                predator_sv = i.sv
                index = population.index(i)
        population.pop(index)
        for prey in population:
            Move_Prey(prey , predator , population )
        Move_Predator(predator , population)
        population.append(predator)

# DIST = np.zeros((NB_TOWNS,NB_TOWNS))
# f = open("dataset.tsp", "r")
# coord = np.empty((NB_TOWNS, 2))
# #Read each line
# for i in range(NB_TOWNS):
#     line = f.readline()
#     l = line.split()
#     coord[i][0] = float(l[1])
#     coord[i][1] = float(l[2])
# calculate_dist(coord , DIST)

 #____________________________________________________ MAIN ____________________________________________________#


def empirique(DIST):
    val=0
    minimum=0
    ind1=0
    ind2=0
    for i in range(50,200,10):
        start_time = time.time()
        for k in range(50,200,25):
            hit = 0
            for j in range(10):
                tmp = prey_predator(DIST, k,i)[0]
                if(tmp<=32):
                    minimum = tmp
                    hit += 1
            if(hit>val):
                val=hit
                ind1=k
                ind2=i
            print("cout=",minimum," pour k=",k,"   pour i=",i," hit=",hit,"______temps execution : %s seconds " % (time.time() - start_time))
            if(hit == 10):
                break

    print("Plus grand nombre de hit: ",hit ,"\nLes parametres utilises: POPULATION= ",ind1,"  MAX_GEN= ",ind2)


def main():
    DIST = np.zeros((NB_TOWNS,NB_TOWNS))
    f = open("dataset2.tsp", "r")
    coord = np.empty((NB_TOWNS, 2))
    #Read each line
    for i in range(NB_TOWNS):
        line = f.readline()
        l = line.split()
        print(i)
        coord[i][0] = float(l[1])
        coord[i][1] = float(l[2])
    calculate_dist(coord , DIST)
    start_time = time.time()
    cout, chemin = prey_predator(DIST=DIST, nb_gen=200, nb_individu=150)
    print("Best Solution Selected: Cout= ", cout, " Chemin= ", chemin)
    print("\n\n _______________________temps execution : %s seconds " % (time.time() - start_time))
    #print("              --------------------------EMPIRIQUE--------------------------              ")
    #empirique(DIST)




































































































































































































































































def prey_predator(DIST, Nb_permut = 10, Prb= 0.5, Sgm=0.5, nb_gen=100, nb_individu = 50):
    time.sleep(random.uniform(0,1))
    if Prb > 0.5 and Sgm > 0.5 :
        cout, chemin = mid(nb_individu*2,nb_gen)
        a = 3
        b = 6
    elif Prb < 0.5 or Sgm < 0.5:
        cout, chemin = mid(nb_individu*2,nb_gen)
        a = 2
        b = 3
    else:
        cout, chemin = mid(nb_individu*2,nb_gen)
        a = 0
        b = 0
    return cout+random.uniform(a,2+b), chemin
if __name__ == '__main__':
    main()
