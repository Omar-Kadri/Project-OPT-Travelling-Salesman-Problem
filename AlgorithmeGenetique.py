from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from typing import List
import random
import numpy
import math
import os
import time 

NB_TOWNS = 14
f = open("dataset.tsp", "r")
CITY_COORDINATES = numpy.empty((NB_TOWNS, 2))
for i in range(NB_TOWNS):
    line = f.readline()
    l = line.split()
    CITY_COORDINATES[i][0] = float(l[1])
    CITY_COORDINATES[i][1] = float(l[2])
TOTAL_CHROMOSOME = len(CITY_COORDINATES) - 1
MUTATION_RATE = 0.2
WEAKNESS_THRESHOLD = 35

class Genome():
    def __init__(self):
        self.chromosome = []
        self.fitness = 0

    def __str__(self):
        return "Chromosome: {0} Fitness: {1}\n".format(self.chromosome, self.fitness) 
    
    def __repr__(self):
        return str(self)

def create_genome() -> Genome:
    genome = Genome()
    
    genome.chromosome = random.sample(range(1, TOTAL_CHROMOSOME + 1), TOTAL_CHROMOSOME)
    genome.fitness = eval_chromosome(genome.chromosome)
    return genome

def distance(a, b) -> float:
    dis = math.sqrt(((a[0] - b[0])**2) + ((a[1] - b[1])**2))
    return numpy.round(dis, 2)

def get_fittest_genome(genomes: List[Genome]) -> Genome:
    genome_fitness = [genome.fitness for genome in genomes]
    return genomes[genome_fitness.index(min(genome_fitness))]

def eval_chromosome(chromosome: List[int]) -> float:
    # Add 0 to beginning and ending of chromosome
    arr = [0] * (len(chromosome) + 2)
    arr[1:-1] = chromosome

    fitness = 0
    for i in range(len(arr) - 1):
        p1 = CITY_COORDINATES[arr[i]]
        p2 = CITY_COORDINATES[arr[i + 1]]
        fitness += distance(p1, p2)
    return numpy.round(fitness, 2)

def tournament_selection(population:List[Genome], k:int) -> List[Genome]:
    selected_genomes = random.sample(population, k)
    selected_parent = get_fittest_genome(selected_genomes)
    return selected_parent

def order_crossover(parents: List[Genome]) -> Genome:
    child_chro = [-1] * TOTAL_CHROMOSOME

    subset_length = random.randrange(2, 5)
    crossover_point = random.randrange(0, TOTAL_CHROMOSOME - subset_length)

    child_chro[crossover_point:crossover_point+subset_length] = parents[0].chromosome[crossover_point:crossover_point+subset_length]

    j, k = crossover_point + subset_length, crossover_point + subset_length
    while -1 in child_chro:
        if parents[1].chromosome[k] not in child_chro:
            child_chro[j] = parents[1].chromosome[k]
            j = j+1 if (j != TOTAL_CHROMOSOME-1) else 0
        
        k = k+1 if (k != TOTAL_CHROMOSOME-1) else 0

    child = Genome()
    child.chromosome = child_chro
    child.fitness = eval_chromosome(child.chromosome)
    return child

def scramble_mutation(genome: Genome) -> Genome:
    subset_length = random.randint(2, 6)
    start_point = random.randint(0, TOTAL_CHROMOSOME - subset_length)
    subset_index = [start_point, start_point + subset_length]

    subset = genome.chromosome[subset_index[0]:subset_index[1]]
    random.shuffle(subset)

    genome.chromosome[subset_index[0]:subset_index[1]] = subset
    genome.fitness = eval_chromosome(genome.chromosome)
    return genome

def reproduction(population: List[Genome]) -> Genome:
    parents = [tournament_selection(population, 20), random.choice(population)] 

    child = order_crossover(parents)
    
    if random.random() < MUTATION_RATE:
        scramble_mutation(child)

    return child

def visualize(all_fittest: List[Genome]):
    fig = plt.figure(tight_layout=True, figsize=(10, 6))
    gs = gridspec.GridSpec(1, 1)

    ax = fig.add_subplot(111)
    all_fitness = [genome.fitness for genome in all_fittest]
    ax.plot(all_fitness, color="orange")
    
    at = AnchoredText(
        "Meilleur cout: {0}".format(all_fittest[-1].fitness), prop=dict(size=10), 
        frameon=True, loc='upper left')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)
    
    ax.set_title("Fitness par generation")
    ax.set_ylabel("Fitness")
    ax.set_xlabel("Generations")
    
    fig.align_labels()
    plt.grid(True)
    plt.show()

def ag(pop,gen):
    POPULATION_SIZE = pop
    MAX_GENERATION = gen
    generation = 0
    population = [create_genome() for x in range (POPULATION_SIZE)]
    all_fittest = []
    while generation != MAX_GENERATION:
        generation += 1

        fittest_genome = get_fittest_genome(population)

        childs = []
        for x in range(int(POPULATION_SIZE * 0.2)):
            child = reproduction(population)
            childs.append(child)

        population.extend(childs)
        # Eliminer les resultats faible
        for genome in population:
            if genome.fitness > WEAKNESS_THRESHOLD:
                population.remove(genome)

        all_fittest.append(get_fittest_genome(population))
    #visualize(all_fittest)
    return [fittest_genome.fitness, fittest_genome.chromosome]
start_time = time.time()
cout, chemin = ag(375,250)
print("Best Solution Selected: Cout= ", cout, " Chemin= ", chemin)
print("\n\n _______________________temps execution : %s seconds " % (time.time() - start_time))
# val=0
# minimum=0
# ind1=0
# ind2=0
# for i in range(50,300,10):
#     start_time = time.time()
#     for k in range(100,400,25):
#         hit = 0
#         for j in range(100):
#             tmp = ag(k,i)[0]
#             if(tmp<=32):
#                 minimum = tmp
#                 hit += 1
#         if(hit>val):
#             val=hit
#             ind1=k
#             ind2=i
#         print("cout=",minimum," pour k=",k,"   pour i=",i," hit=",hit,"______temps execution : %s seconds " % (time.time() - start_time))
#         if(hit == 10):
#             break

# print("Plus grand nombre de hit: ",hit ,"\nLes parametres utilises: POPULATION= ",ind1,"  MAX_GEN= ",ind2)
# print("\ncout= 30.87    pour k= 150    pour i= 240    hit= 47 _______________________temps execution : 276.4898011684418 seconds \ncout= 31.45    pour k= 175    pour i= 240    hit= 58 _______________________temps execution : 407.77050280570984 seconds \ncout= 31.22    pour k= 200    pour i= 240    hit= 66 _______________________temps execution : 559.9592773914337 seconds \ncout= 31.21    pour k= 225    pour i= 240    hit= 59 _______________________temps execution : 731.5191967487335 seconds \ncout= 31.96    pour k= 250    pour i= 240    hit= 69 _______________________temps execution : 924.2597403526306 seconds \ncout= 31.22    pour k= 275    pour i= 240    hit= 72 _______________________temps execution : 1135.484478712082 seconds \ncout= 31.21    pour k= 300    pour i= 240    hit= 72 _______________________temps execution : 1368.1199088096619 seconds \ncout= 31.81    pour k= 325    pour i= 240    hit= 73 _______________________temps execution : 1620.7261354923248 seconds \ncout= 30.87    pour k= 350    pour i= 240    hit= 77 _______________________temps execution : 1893.658276796341 seconds \ncout= 31.21    pour k= 375    pour i= 240    hit= 78 _______________________temps execution : 2190.424998521805 seconds \ncout= 31.21    pour k= 100    pour i= 250    hit= 41 _______________________temps execution : 77.00842642784119 seconds \ncout= 30.87    pour k= 125    pour i= 250    hit= 43 _______________________temps execution : 173.24294781684875 seconds \ncout= 31.95    pour k= 150    pour i= 250    hit= 46 _______________________temps execution : 290.63057112693787 seconds \ncout= 30.87    pour k= 175    pour i= 250    hit= 53 _______________________temps execution : 429.53260564804077 seconds \ncout= 31.21    pour k= 200    pour i= 250    hit= 66 _______________________temps execution : 589.2707855701447 seconds \ncout= 30.87    pour k= 225    pour i= 250    hit= 66 _______________________temps execution : 770.2983441352844 seconds \ncout= 31.78    pour k= 250    pour i= 250    hit= 71 _______________________temps execution : 971.9258825778961 seconds \ncout= 30.87    pour k= 275    pour i= 250    hit= 78 _______________________temps execution : 1196.981167793274 seconds \ncout= 30.87    pour k= 300    pour i= 250    hit= 76 _______________________temps execution : 1447.0927786827087 seconds \ncout= 30.87    pour k= 325    pour i= 250    hit= 74 _______________________temps execution : 1769.7151033878326 seconds \ncout= 31.62    pour k= 350    pour i= 250    hit= 78 _______________________temps execution : 2111.6572427749634 seconds \ncout= 30.87    pour k= 375    pour i= 250    hit= 85 _______________________temps execution : 2438.3357837200165 second")
# print("Plus grand nombre de hit: ",85 ,"\nLes parametres utilises: POPULATION= ",375,"  MAX_GEN= ",250)