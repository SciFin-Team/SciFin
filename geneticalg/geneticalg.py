# Created on 2020/7/22

# This module is for importing, transforming and visualizing market data.

import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
import random as random
import matplotlib.pyplot as plt


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








































