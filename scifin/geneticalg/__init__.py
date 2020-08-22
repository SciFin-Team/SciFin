# __init__.py
__version__ = "0.1.0"
__author__ = "Fabien Nugier"

"""
The :mod:`scifin.geneticalg` module includes methods for genetic algorithms.
"""

from .geneticalg import Individual, Population, \
                        generate_random_population, get_generation, roulette, selection, \
                        pairing, non_adjacent_random_list, mating_pair, get_offsprings, \
                        mutate_individual, mutation_set, mutate_population, \
                        next_generation, get_elite_and_individuals, \
                        fitness_similarity_check, sum_top_fitness, plot_compare_genomes

