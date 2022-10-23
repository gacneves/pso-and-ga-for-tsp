from utils import *
from random import sample, random, randint
import pandas as pd
import math


class SwapSequence:
    def __init__(self):
        self.queue = []

    def enqueue(self, exchange_tuple):
        self.queue.append(exchange_tuple)

    def dequeue(self):
        return self.queue.pop(0)

    def enqueue_ss(self, ss_queue):
        self.queue.extend(ss_queue)


def swap_cities(route, i, j):
    """
    Swap element of index i with element of index j in array route

    :param route: Array of elements to be swapped
    :param i: Index of the first element
    :param j: Index of the second element
    """
    route[i], route[j] = route[j], route[i]


def routes_subtraction(route_one, route_two):
    """
    Generate a SwapSequence for the route_two to become the route_one (SS = A - B => B + SS = A)

    :param route_one: Left operand of the subtraction
    :param route_two: Right operand of the subtraction
    :return: SwapSequence for route_two
    """
    swappable_route_two = route_two.copy()
    result_ss = SwapSequence()
    length = len(route_one)
    index = 0
    while index < length:
        if route_one[index] != swappable_route_two[index]:
            swap_index = swappable_route_two.index(route_one[index])
            result_ss.enqueue((index, swap_index))
            swap_cities(swappable_route_two, index, swap_index)
        index += 1

    return result_ss


class Particle:
    def __init__(self, route):
        """
        Individual of the swarm

        :param route: Initial route of the particle
        """
        self.route = route  # Particle position
        self.swap_sequence = SwapSequence()  # Particle velocity
        self.__generate_initial_swap_sequence()  # Initial velocity
        self.fitness = math.inf
        self.my_best_fitness = math.inf
        self.my_best_route = []

    def __generate_initial_swap_sequence(self):
        cities_number = len(self.route)
        swap_no = randint(0, cities_number)
        for swap in range(swap_no):
            city_i = randint(0, cities_number - 1)
            city_j = randint(0, cities_number - 1)
            self.swap_sequence.enqueue((city_i, city_j))


class PSO:
    def __init__(self, table_of_distances: pd.DataFrame, population_size, max_epochs, fitness_function, phi_one, phi_two):
        """
        Particle Swarm Optimization to solve TSP

        :param table_of_distances: Distances of each city pair
        :param population_size: Number of individuals in swarm
        :param max_epochs: Max number of epochs
        :param fitness_function: Function to evaluate each particle of the swarm
        :param phi_one: Probability of the particle to go in direction of the GLOBAL best
        :param phi_two: Probability of the particle to go in direction of the LOCAL best
        """
        self.table_of_distances = table_of_distances
        self.max_epochs = max_epochs
        self.population_size = population_size
        self.population = []
        self.__initialize_population()
        self.fitness_function = fitness_function
        self.global_best_solution_route = []
        self.global_best_solution_fitness = math.inf
        self.phi_one = phi_one
        self.phi_two = phi_two

    def __initialize_population(self):
        cities = list(self.table_of_distances.index.values)
        for individual in range(self.population_size):
            initial_route = sample(cities, len(cities))
            self.population.append(Particle(initial_route))
        self.global_best_solution_route = self.population[0].route
        self.global_best_solution_fitness = self.population[0].fitness

    def evaluate_population(self):
        for particle in self.population:
            fitness = self.fitness_function(particle.route, self.table_of_distances)
            particle.fitness = fitness
            if fitness < particle.my_best_fitness:
                particle.my_best_fitness, particle.my_best_route = fitness, particle.route
            if fitness < self.global_best_solution_fitness:
                self.global_best_solution_route, self.global_best_solution_fitness = particle.route.copy(), \
                                                                                     particle.fitness

    def update_population(self):
        for particle in self.population:
            # Updating velocity
            # V(t) = V(t-1) + phi_one * (best_global - current_route) + phi_two * (best_local - current_route)

            # phi_one * (best_global - current_route)
            global_subtraction = routes_subtraction(self.global_best_solution_route, particle.route)
            aux_global_subtraction_queue = global_subtraction.queue.copy()
            for swap in global_subtraction.queue:
                probability = random()
                if probability <= self.phi_one:
                    aux_global_subtraction_queue.remove(swap)
            global_subtraction.queue = aux_global_subtraction_queue

            # phi_two * (best_local - current_route)
            local_subtraction = routes_subtraction(particle.my_best_route, particle.route)
            aux_local_subtraction_queue = local_subtraction.queue.copy()
            for swap in local_subtraction.queue:
                probability = random()
                if probability <= self.phi_two:
                    aux_local_subtraction_queue.remove(swap)
            local_subtraction.queue = aux_local_subtraction_queue

            # Update position
            # X(t) = X(t-1) + V(t-1)
            for swap in particle.swap_sequence.queue:
                swap_cities(particle.route, swap[0], swap[1])

            # Update V(t)
            particle.swap_sequence.enqueue_ss(global_subtraction.queue)
            particle.swap_sequence.enqueue_ss(local_subtraction.queue)

    def evolve(self):
        for epoch in range(self.max_epochs):
            self.evaluate_population()
            self.update_population()
        print('Best route is', self.global_best_solution_route, 'with a distance equal to',
              self.global_best_solution_fitness)


if __name__ == '__main__':
    cities_distance = read_table_of_distances('distances.csv')
    population = 30
    epochs = 1000
    fitness_func = distance_of_route
    global_probability = 0.4
    local_probability = 0.6
    for _ in range(15):
        pso = PSO(cities_distance, population, epochs, fitness_func, global_probability, local_probability)
        pso.evolve()
