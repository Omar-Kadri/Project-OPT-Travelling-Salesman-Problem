from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from typing import List
import random
import numpy
import math
import os
import time

NB_TOWNS = 48
f = open("dataset2.tsp", "r")
CITY_COORDINATES = numpy.empty((NB_TOWNS, 2))
for i in range(NB_TOWNS):
    line = f.readline()
    l = line.split()
    print(i)
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

def mid(pop,gen):
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