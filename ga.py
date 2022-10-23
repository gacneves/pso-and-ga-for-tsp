from utils import *
from random import sample, random
import math


class Chromosome:
    def __init__(self, route):
        """
        Individual of the Genetic Algorithm

        :param route: Initial route of the gene
        """
        self.route = route
        self.fitness = math.inf

    def mutate(self):
        city1, city2 = sample(range(1, len(self.route)), 2)
        self.route[city1], self.route[city2] = self.route[city2], self.route[city1]


def tournament(parent_list, n):
    """
    Parents tournament selection

    :param parent_list: Possible parent list
    :param n: Tournament size
    :return: Parent with best fitness chosen in tournament
    """
    candidates = sample(parent_list, n)
    return sorted(candidates, key=lambda chromosome: chromosome.fitness, reverse=False)[0]


def crossover(parent1, parent2):
    """
    Make a crossover between two parents of solutions of TSP. Basically an interval is generated randomly and each child
    receives a slice of one of the parents based on these values, then it is completed with the missing cities in the
    order they appear in the other parent.

    :param parent1: Parent one
    :param parent2: Parent two
    :return: Two chromosomes generated from the crossover of parent1 and parent2
    """
    length = len(parent1.route)
    child1_route = [None] * length
    child2_route = [None] * length

    indexes = sample(range(length), 2)
    start, end = min(indexes), max(indexes)
    end += 1
    child1_route[start:end], child2_route[start:end] = parent2.route[start:end], parent1.route[start:end]

    parent_order_index = 0
    for child_city in range(length):
        if child1_route[child_city] is None:
            while parent1.route[parent_order_index] in child1_route:
                parent_order_index += 1
            child1_route[child_city] = parent1.route[parent_order_index]

    parent_order_index = 0
    for child_city in range(length):
        if child2_route[child_city] is None:
            while parent2.route[parent_order_index] in child2_route:
                parent_order_index += 1
            child2_route[child_city] = parent2.route[parent_order_index]

    return Chromosome(child1_route), Chromosome(child2_route)


class GA:
    def __init__(self, table_of_distances, population_size, generations, mutation_rate, elitism, tournament_size,
                 fitness_function):
        """
        Genetic Algorithm to solve TSP

        :param table_of_distances: Distances of each city pair
        :param population_size: Number of individuals in population
        :param generations: Maximum number of generations
        :param mutation_rate: Probability of an individual of the population to mutate
        :param elitism: Number of the best individuals to be carried over to the next generation
        :param tournament_size: Size of the tournament selection
        :param fitness_function: Function to evaluate each individual of the population
        """
        self.table_of_distances = table_of_distances
        self.generations = generations
        self.population_size = population_size
        self.population = []
        self.__initialize_population()
        self.ranking = []
        self.elitism = elitism
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.fitness_function = fitness_function

    def __initialize_population(self):
        cities = list(self.table_of_distances.index.values)
        for individual in range(self.population_size):
            initial_route = sample(cities, len(cities))
            self.population.append(Chromosome(initial_route))

    def evaluate_population(self):
        self.ranking = []
        for chromosome in self.population:
            chromosome.fitness = self.fitness_function(chromosome.route, self.table_of_distances)
            if not self.ranking:
                self.ranking.append(chromosome)
                continue
            for position in self.ranking:
                if chromosome.fitness < position.fitness:
                    self.ranking.insert(self.ranking.index(position), chromosome)
                    break
            else:
                self.ranking.append(chromosome)

    def generate_new_population(self):
        new_population = []

        # Elitism warranty
        new_population += self.ranking[:self.elitism]

        # Parent selection
        parent_list = []
        for _ in range(self.population_size - self.elitism):
            parent = tournament(self.ranking, self.tournament_size)
            parent_list.append(parent)

        # Crossover
        length = len(parent_list)
        for i in range(math.ceil(length / 2)):
            parent1 = parent_list[i]
            parent2 = parent_list[length - i - 1] if i != length - i - 1 else sample(parent_list[:i - 1] +
                                                                                     parent_list[i + 1:], 1)[0]
            child1, child2 = crossover(parent1, parent2)
            new_population.append(child1)
            new_population.append(child2)

        # Mutation
        for individual in new_population[self.elitism:]:
            probability = random()
            if probability <= self.mutation_rate:
                individual.mutate()

        self.population = new_population

    def evolve(self):
        for _ in range(self.generations):
            self.evaluate_population()
            self.generate_new_population()
        print('Best route is', self.ranking[0].route, 'with a distance equal to', self.ranking[0].fitness)


if __name__ == '__main__':
    cities_distance = read_table_of_distances('distances.csv')
    population = 50
    generations = 1000
    mutation_rate = 0.01
    elitism = 5
    tournament_selection = 3
    fitness_func = distance_of_route
    for _ in range(15):
        ga = GA(cities_distance, population, generations, mutation_rate, elitism, tournament_selection, fitness_func)
        ga.evolve()
