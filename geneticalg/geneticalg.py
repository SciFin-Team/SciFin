# Created on 2020/7/22

# This module is for importing, transforming and visualizing market data.

import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
import random as random
import matplotlib.pyplot as plt

import marketdata.simuldata as sd


def individual(number_of_genes, upper_limit, lower_limit, sum_target):
    """
    Function that creates an individual from random values that we call genes.
    These genes can represent the investment into a market or any other value in a pool of possible values.
    
    Arguments:
    - number_of_genes: it is the number of genes (but it can also be the number of assets in the market).
    - upper_limit: that's the maximum value taken by the genes (or amount we invest), before normalization.
    - lower_limit: that's the minimum value taken by the genes (or amount we invest), before normalization.
    - sum_target: that's the sum value of the genes (or the total investment that will be reached), after normalization.
    
    Note: this function has a little bug that turns out funny. If the sum of all the values is negative, then the normalization will reverse the sign,
    and we will end up having a portfolio which has flipped signes, hence specified long positions become short, and conversely. So I just put a small test.
    """
    individual = [random.random() * (upper_limit-lower_limit) + lower_limit for x in range(number_of_genes)]
    normalization = sum(individual) / sum_target
    if normalization < 0:
        raise Exception("Shorting too many assets. Not allowed for now.")
    normalized_individual = np.array(individual) / normalization
    
    return normalized_individual



def population(number_of_individuals, number_of_genes, upper_limit, lower_limit, sum_target, birth_date, name_indiv="Indiv Portfolio"):
    """
    Function that creates a population of individuals from the function `individual`.
    
    Arguments:
    number_of_individuals: the number of individuals we want in this creation of a population
    number_of_genes: the number of genes each of these individuals have
    upper_limit: the higher limit of genes, i.e. largest amount of long positions
    lower_limit: the lowest limit of genes, i.e. the lowest amount we invest in a given asset. Value can be negative (short positions).
    sum_target: the sum of all these positions.
    birth_date: a date to specify at which time the individuals of that population were created.
    name_indiv: the name we choose for the individuals.
    """
    
    # Building a data frame of individuals
    pop = pd.DataFrame([individual(number_of_genes, upper_limit, lower_limit, sum_target) for _ in range(number_of_individuals)])
    pop.columns = ["Asset " + str(i) for i in range(number_of_genes)]
    
    # Setting the birthdate
    pop["Born"] = [birth_date for _ in range(number_of_individuals)]
    
    # Setting the row names
    pop.index = [name_indiv + str(i+1) for i in range(number_of_individuals)]
    pop.index.names = ["Individuals"]
    
    return pop




def get_generation(population, market, current_eval_date, next_eval_date, lamb=0.5, fitness_method="Max Return and Vol", return_propag=False, date_format="%Y-%m"):
    """
    Takes a population, propagate its elements to the next evaluation event, and compute their fitness
    
    Arguments:
    - population: the population to evolve
    - market: the market which serves as a basis for propagation
    - current_eval_date: the present date on which we evaluate the portfolios
    - next_eval_date: the target date until which we want to make the portfolios evolve
    - return_propag: an option to return the propagations
    - date_format: format of the dates in the data frames
    
    Note: we define a generation as a population for which the fitness has been computed and who is sorted according to it.
    
    Note: - population has rows which are the names of the portfolio, and columns which are the assets.
          - propagation has rows which are the time stamps, and columns which are the names of the portfolios.
    """
    
    # Make sure all the individual portfolios were born before the current date:
    for birth_date in population['Born']:
        if birth_date >= next_eval_date:
            raise Exception("Individuals can't be born at/after the evaluation date.")
    # Make sure the next evaluation date is later than the current date:
    if current_eval_date >= next_eval_date:
        raise Exception("Current date can't be after the evaluation date.")
    
    # Getting the date just before the next evaluation date (at which reproduction will happen)
    date_before_eval = sd.find_tick_before_eval(market.index, next_eval_date)
    
    # Propagate individuals
    propagation = sd.limited_propagation(population, market, current_eval_date, date_before_eval)
    
    # Create the generation from the population copy
    try:
        generation = population.copy()
    except:
        print(population)
    
    # Remove any fitness column if it existed already (otherwise we will sum it/them after... potentially disrupting the fitness calculation)
    # This is in case we use a generation (i.e. a population with computed fitness and ranked by it) as a population argument
    # if len(generation.filter(regex='Fit').columns) != 0:
    #     generation.drop(columns=generation.filter(regex='Fit').columns.tolist(), inplace=True)
    # Not needed if we want to keep the fitness columns with dates, which sounds like a better solution for the moment
        
    # Compute the fitness of individuals and sort them by fitness
    fitness_name = "Fitness " + date_before_eval.strftime(date_format)
    generation[fitness_name] = sd.fitness_calculation(population, propagation, market, current_eval_date, next_eval_date, lamb, fitness_method)
    generation.sort_values(by=[fitness_name], ascending=False, inplace=True)
    
    if return_propag == True:
        return generation, propagation
    else:
        return generation

    
def roulette(cum_sum, chance):
    """
    The function for roulette wheel selection takes the cumulative sums and the randomly generated value
    for the selection process and returns the number of the selected individual. By calculating the cumulative sums,
    each individual has a unique value between 0 and 1. To select individuals, a number between 0 and 1 is randomly
    generated and the individual that is closes to the randomly generated number is selected.
    """
    variable = list(cum_sum.copy())
    variable.append(chance)
    variable = sorted(variable)
    return variable.index(chance)


def selection(generation, method='Fittest Half'):
    """
    Function that operates the selection among a generation, based on the last fitness.
    Different methods can be used among 'Fittest Half', 'Random' and 'Roulette Wheel'.
    
    Methods:
    'Fittest Half': the first half of the top fitness is kept
    'Random': rows are picked up at random, but they can't be the same
    'Roulette Wheel': rows are picked up at random, but with a preference for high cumulative sum (high fitness).
    """
    
    # Initialization
    N = generation.shape[0]
    
    # Sorting:
    # We use the last fitness column and normalize it
    last_fitness = generation.filter(regex='Fit').columns[-1]
    generation['Normalized Fitness'] = [generation[last_fitness][x]/sum(generation[last_fitness]) for x in range(len(generation[last_fitness]))]

    # We sort the values from smallest to highest for computing the cumulative sum
    generation.sort_values(by='Normalized Fitness', ascending=True, inplace=True)
    
    # Computing the cumulative sum
    cumsum_name = "CumSum " + last_fitness.split(" ")[1]
    generation[cumsum_name] = np.array(generation['Normalized Fitness']).cumsum()
    
    # We sort the values back, from highest fitness to lowest
    generation.sort_values(by=cumsum_name, ascending=False, inplace=True)
    
    # We get rid of the normalized fitness and just use the cumulative sum
    generation.drop(columns=["Normalized Fitness"], inplace=True)
    
    
    # Selection:
    if method == 'Fittest Half':
        select_rows = [x for x in range(N//2)]
        selected_individuals = generation.iloc[select_rows,:]
        
    elif method == 'Random':
        select_rows = [x for x in range(N)]
        random.shuffle(select_rows)
        selected_individuals = generation.iloc[select_rows[:N//2],:]
        
    elif method == 'Roulette Wheel':
        selected = []
        for x in range(N//2):
            selected.append(roulette(generation[cumsum_name], random.random()))
            while len(set(selected)) != len(selected):
                selected[x] = roulette(generation[cumsum_name], random.random())
        select_rows = [int(selected[x]) for x in range(N//2)]
        selected_individuals = generation.iloc[select_rows,:]

    return pd.DataFrame(selected_individuals)


    
    



































