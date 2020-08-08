# __init__.py
__version__ = "0.0.8"
__author__ = "Fabien Nugier"

"""
The :mod:`scifin.geneticalg` module includes methods for genetic algorithms.
"""

from .geneticalg import individual, population, get_generation, roulette, selection, \
                        pairing, non_adjacent_random_list, mating_pair, get_offsprings, \
                        mutate_individual, mutation_set, mutate_population, \
                        next_generation, get_elite_and_individuals, \
                        fitness_similarity_check, sum_top_fitness, plot_compare_genomes

