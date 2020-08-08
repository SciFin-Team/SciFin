# Created on 2020/7/22

# This module is for importing, transforming and visualizing market data.

# Standard library imports
from datetime import datetime
from datetime import timedelta
import random as random

# Third party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Local application imports
from .. import marketdata


#---------#---------#---------#---------#---------#---------#---------#---------#---------#


def individual(number_of_genes, upper_limit, lower_limit, sum_target):
    """
    Creates an individual from random values called genes.    
    
    Parameters
    ----------
    number_of_genes : int
      Number of genes making up the individual.
    upper_limit : float
      Maximum value taken by the genes, before normalization.
    lower_limit : float
      Minimum value taken by the genes, before normalization. Can be negative.
    sum_target : float  
      Target value for the sum of the genes after normalization.
    
    Returns
    -------
    Numpy Array
      Array containing the values of the genes making up the individual.
      
    Notes
    -----
      These genes can represent the investment into a market
      or any other value in a pool of possible values.
      If the gene value is positive, it corresponds to a long position,
      if negative, it corresponds to a short position.
      
      In the case of a portfolio, number_of_genes can be the number of assets
      to consider, upper_limit / lower_limit the respective maximum / minimum
      asset allocations, and sum_target the total investment value after normalization.
      
      This function has a little bug that turns out to be funny.
      If the sum of all the values is negative, then the normalization will
      reverse the sign and we will end up having a genes which have flipped signs.
      
      For a portfolio application, this means long positions become short, and conversely.
      To prevent this from happening, an exception is raised.
    """
    
    # Checks
    assert(isinstance(number_of_genes, int))
    
    # Generating an individual and computing normalization
    individual = [ random.random() * (upper_limit-lower_limit) 
                   + lower_limit for x in range(number_of_genes) ]
    normalization = sum(individual) / sum_target
    
    # Check the normalization is positive
    if normalization < 0:
        raise Exception("Negative normalization not allowed.")
        
    normalized_individual = np.array(individual) / normalization
    
    return normalized_individual



def population(number_of_individuals, number_of_genes, upper_limit, lower_limit,
               sum_target, birth_date, name_indiv="Indiv"):
    """
    Creates a population of individuals from the function `individual`.
    
    Parameters
    ----------
    number_of_individuals : int
      Number of individuals we want in this creation of a population.
    number_of_genes : int
      Number of genes each of these individuals have.
    upper_limit : float
      Maximum value taken by the genes, before normalization.
    lower_limit : float
      Minimum value taken by the genes, before normalization. Can be negative.
    sum_target : float  
      Target value for the sum of the genes after normalization.
    birth_date : str
      Date specifying the time individuals of that population were created.
    name_indiv : str
      Name we choose for the individuals of this population.
      
    Returns
    -------
    DataFrame
      Pandas data frame containing the individuals of the populations as rows
      and genes composing them as columns.
    """
    
    # Checks
    assert(isinstance(number_of_individuals,int))
    assert(isinstance(number_of_genes,int))
    
    # Building a data frame of individuals
    pop = pd.DataFrame([ individual(number_of_genes, upper_limit, lower_limit, sum_target)
                         for _ in range(number_of_individuals) ])
    pop.columns = ["Asset " + str(i) for i in range(number_of_genes)]
    
    # Setting the birth date of the individuals
    pop["Born"] = [birth_date for _ in range(number_of_individuals)]
    
    # Setting the row names
    pop.index = [name_indiv + str(i+1) for i in range(number_of_individuals)]
    pop.index.names = ["Individuals"]
    
    return pop


def get_generation(population, environment, current_eval_date, next_eval_date,
                   lamb=0.5, fitness_method="Last Return and Vol",
                   return_propag=False, date_format="%Y-%m"):
    """
    Takes a population, propagate its elements to the next evaluation event, and compute their fitness.
    A generation is defined as a population with computed fitness and ranked by it.
    
    Different fitness methods can be used:
    - "Last Return": uses the value of return at the end of propagation,
                     i.e. the value just before evaluation date.
    - "Last Return and Vol": combines last return with estimation of volatility,
                             the dosage between the two being tuned by lambda.
    - "Avg Return and Vol": uses average return during the propagation and the
                            estimation of volatility, also using lambda.
    - "Sharpe Ratio": uses the ratio of average return over estimation of volatility.
    
    Parameters
    ----------
    population : DataFrame
      Population to evolve.
    environment : DataFrame
      Environment which serves as a basis for propagation.
    current_eval_date : Period date
      Present date on which we evaluate the individuals.
    next_eval_date : Period date
      Target date until which we want to make the portfolios evolve.
    lamb : float in [0,1]
      Parameter of the fitness calculation to decide between returns and volatility.
    fitness_method : str
      Name of the fitness method chosen among:
      {"Last Return", "Last Return and Vol", "Avg Return and Vol", "Sharpe Ratio"}
    return_propag : bool
      An option to return or not the propagations.
    date_format : str
      Format of the dates in the data frames.
    
    Returns
    -------
    DataFrame
      Pandas data frame represented the propagated population, i.e. the generation
      with individuals as rows and {genes, birth date, fitness} as columns.
    DataFrame
      Eventually returns also the propagation of the individuals.
    
    Raises
    ------
    Exception
      Imposes that individuals are born before the evaluation date.
    Exception
      Checks that the current evaluation date is before the next evaluation date.
    
    Notes
    -----
     We define a generation as a population for which the fitness
     has been computed and who is sorted according to it.
    
     In the context of portfolios, a 'population' has rows which are the names
     of the portfolio and columns which are the assets.
     
     The function can also return 'propagation' that has rows which are the time stamps
     and columns which are the names of the portfolios.
     
     For date formats please refer to the following:
     https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.DatetimeIndex.strftime.html
     
    Examples
    --------
      None
    """
    
    # Checks
    # Make sure all the individual portfolios were born before the current date:
    for birth_date in population['Born']:
        if birth_date >= next_eval_date:
            raise Exception("Individuals can't be born at/after the evaluation date.")
    # Make sure the next evaluation date is later than the current date:
    if current_eval_date >= next_eval_date:
        raise Exception("Current date can't be after the evaluation date.")
    
    # Getting the date just before the next evaluation date
    # (date at which reproduction will happen)
    date_before_eval = marketdata.find_tick_before_eval(environment.index, next_eval_date)
    
    # Propagate individuals
    propagation = marketdata.limited_propagation(population, environment,
                                                 current_eval_date, date_before_eval)

    # Create the generation from the population copy
    try:
        generation = population.copy()
    except:
        print(population)
        
    # Compute the fitness of individuals
    fitness_name = "Fitness " + date_before_eval.strftime(date_format)
    generation[fitness_name] = marketdata.fitness_calculation(population, propagation, environment,
                                                              current_eval_date, next_eval_date,
                                                              lamb, fitness_method)
    # Sort individuals by fitness
    generation.sort_values(by=[fitness_name], ascending=False, inplace=True)
    
    # Return generation (and eventually propagation)
    if return_propag == True:
        return generation, propagation
    else:
        return generation

    
def roulette(cum_sum, chance):
    """
    Takes the cumulative sums and the randomly generated value for the selection process
    and returns the number of the selected individual.
    
    By calculating the cumulative sums, each individual has a unique value between 0 and 1.
    
    To select individuals, a number between 0 and 1 is randomly generated
    and the individual that is closes to it is selected.
    
    Parameters
    ----------
    cum_sum : Panda.Series
      Series of cumulative sum of fitness for each individuals.
    chance : float in [0,1]
      Number to select the individual.
    
    Returns
    -------
    int
      Number refering to selected row.
    """

    variable = list(cum_sum.copy())
    variable.append(chance)
    variable = sorted(variable)

    return variable.index(chance)


def selection(generation, method='Fittest Half'):
    """
    Operates the selection among a generation based on the last fitness.
    
    Different methods can be used:
    - 'Fittest Half': the first half of the top fitness is kept.
    - 'Random': rows are picked up at random, but can't be the same.
    - 'Roulette Wheel': rows are picked up at random, but with a preference
                        for high cumulative sum (high fitness).
                        
    Parameters
    ----------
    generation : DataFrame
      Generation to do the selection form.
    method : str
      Method of selection.
    
    Returns
    -------
    DataFrame
      Data frame of selected individuals.    
    """
    
    # Initialization
    N = generation.shape[0]
    
    # Sorting:
    # We use the last fitness column and normalize it
    last_fitness = generation.filter(regex='Fit').columns[-1]
    generation['Normalized Fitness'] = [ generation[last_fitness][x]/sum(generation[last_fitness])
                                         for x in range(len(generation[last_fitness])) ]

    # We sort the values from smallest to highest for computing the cumulative sum
    generation.sort_values(by='Normalized Fitness', ascending=True, inplace=True)
    
    # Computing the cumulative sum
    cumsum_name = "CumSum " + last_fitness.split(" ")[1]
    generation[cumsum_name] = np.array(generation['Normalized Fitness']).cumsum()
    
    # We sort the values back, from highest fitness to lowest
    generation.sort_values(by=cumsum_name, ascending=False, inplace=True)
    
    # We get rid of the normalized fitness and just use the cumulative sum
    generation.drop(columns=["Normalized Fitness"], inplace=True)
    
    
    # Doing the selection:
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



def pairing(elite, selected, method = 'Fittest'):
    """
    Establishes the pairing of the selected population and elite all together.
    
    Different methods can be used:
    - 'Fittest': the top fitness individuals are paired in order, this does not check
                 if the elite is fit however (we let evolution do that).
                 The selected individuals should be ranked from fittest to less fit
                 if things have been done in the proper order before.
    - 'Random': the pairing is done at random, but no individual
                can reproduce more than once.
    - 'Weighted Random': the pairing is done with higher probability among high fitness
                         individuals. Some individuals may reproduce several times.

    Parameters
    ----------
    elite : DataFrame
      Elite population.
    selected : DataFrame
      Selected population.
    method : str
      Method to be used.
    
    Returns
    -------
    List of lists of 2 strings
      List of pairs of parents mating together.
    List of DataFrames
      List of the parents genes.
    """
    
    # Combining elite and previously selected individuals
    cumsumcols_toremove = selected.filter(regex="CumSum").columns
    individuals = pd.concat([elite, selected.drop(columns=cumsumcols_toremove)])
    individuals.reindex(np.random.permutation(individuals.index))
    
    # Getting the fitness of all of them (not necessarily ordered)
    last_fitness = selected.filter(regex="Fit").columns[-1]
    Fitness = individuals[last_fitness]

    # Initialization for pairing
    M = individuals.shape[0]
    
    # odd_indiv = False
    if M%2 != 0:
        raise Exception("NUMBER OF INDIVIDUALS IS ODD ! PROBLEM WITH PAIRING !")
    
    # Start the pairing
    pairs = []
    if method == 'Fittest':
        parents_pairs = [[individuals.index[x], individuals.index[x+1]] for x in range(0,M,2)]
        parents_values = [[individuals.iloc[x], individuals.iloc[x+1]] for x in range(0,M,2)]
        
    if method == 'Random':
        pairs = [x for x in range(M)]
        random.shuffle(pairs)
        parents_pairs = []
        parents_values = []
        for x in range(0,M,2):
            parents_pairs.append([individuals.index[pairs[x]], individuals.index[pairs[x+1]]])
            parents_values.append([individuals.iloc[pairs[x]], individuals.iloc[pairs[x+1]]])
                
    # Note: This method does not allow a parent to reproduce with himself,
    #       but allows some parents to reproduce several times!
    if method == 'Weighted Random':
        Normalized_Fitness = sorted([Fitness[x]/sum(Fitness) for x in range(len(Fitness))], reverse = True)
        Cumulative_Sum = np.array(Normalized_Fitness).cumsum()
        parents_pairs = []
        parents_values = []
        for x in range(M//2):
            pair = [roulette(Cumulative_Sum, random.random()), roulette(Cumulative_Sum, random.random())]
            parents_pairs.append([individuals.index[pair[0]], individuals.index[pair[1]]])
            parents_values.append([individuals.iloc[pair[0]], individuals.iloc[pair[1]]])
            while parents_pairs[x][0] == parents_pairs[x][1]:
                new_element = roulette(Cumulative_Sum, random.random())
                parents_pairs[x][1] = individuals.index[new_element]
                parents_values[x][1] = individuals.iloc[new_element]
                
    return parents_pairs, parents_values



def non_adjacent_random_list(Nmin, Nmax, Npoints):
    """
    Generates a list of Npoints non-adjacent numbers at random,
    taken from a list of integers between Nmin and Nmax.
    
    Extreme ends of the list Nmin, Nmin+1, ..., Nmax-1, Nmax are excluded.
    
    Parameters
    ----------
    Nmin : int
      Minimal value.
    Nmax : int
      Maximal value.
    Npoints : int
      Number of points.
    
    Returns
    -------
    List of int
      The list of non-adjacent numbers.
      
    Notes
    -----
    This function is written to be used in the function `mating_pair`.
    """
    
    # Initial tests
    if Nmin%1!=0 or Nmax%1!=0 or Npoints%1!=0:
        raise Exception("Nmin, Nmax and Npoints must be integers.")

    if Npoints > int((Nmax-Nmin-2)/3):
        print("For Nmin =",Nmin,"and Nmax =",Nmax,"we must have Npoints <=",int((Nmax-Nmin-2)/5))
        raise Exception("Please take a lower value to avoid slowness.")

    # Initialization
    list_pts=[]
    count = 0

    # Process
    while count < Npoints:
        pt = random.randint(Nmin+1, Nmax-1) 
        pt_is_fine = True
        for p in list_pts:
            if np.abs(p-pt)<2:
                pt_is_fine = False
                continue
        if pt_is_fine:
            list_pts.append(pt)
            count +=1
            
    # For visual check
    # fig = plt.figure(figsize=(15,5))
    # plt.hist(list_pts, bins=Nmax-Nmin)
    
    return sorted(list_pts)



def mating_pair(pair_of_parents, mating_date, method='Single Point', n_points=None):
    """
    Takes a pair of parents and makes a reproduction of them to produce two offsprings.
    This is done by exchanging sequences of genes between parents.
    
    Different methods can be used:
    - 'Single Point': using only one pivot value for exchange of genes.
    - 'Two Points': using two pivot values for exchange of genes.
    - 'Multi Points': using `n_points` pivot values for the exchange of genes.
    
    Parameters
    ----------
    pair_of_parents : List of 2 DataFrames
      List of the two parents to mate.
    mating_date : str or Period date
      Mating date.
    method : str
      Methods of mating.
    n_points : int
      Number of points pivot values for method 'Multi Points'.
    
    Returns
    -------
    List of 2 DataFrames
      List of the 2 offsprings resulting from the mating of the 2 parents.
      
    Notes
    -----
      Considering that one application of this function is for portfolios,
      we need to consider the following:
      
      - Since the echange of genes have changed the sum of each portfolio values,
      we are enforced to renormalize the values.
      
      - This makes the exchange of genes harder to compare when looking at values,
      but it is necessary to avoid portfolios overall investment to change.
    """
    
    # Check that there is only 2 parents here
    if len(pair_of_parents) != 2:
        raise Exception("Only a pair of parents allowed here!")
    
    # Selecting only the columns with Asset allocations, i.e. the genes
    parents = pd.DataFrame(pair_of_parents).filter(regex="Asset")
    Ngene = parents.shape[1]
    
    # Check that the parents have the same sum
    if parents.iloc[0].sum() - parents.iloc[1].sum() > 1E-5 :
        print(parents.iloc[0].sum(), parents.iloc[1].sum())
        raise Exception("Parents must have the same sum of assets allocations.")
    parents_sum = parents.iloc[0].sum()
        
    # Creating offsprings - Method 1
    if method == 'Single Point':
        pivot_point = random.randint(1, Ngene-2)
        
        offspring1 = parents.iloc[0,0:pivot_point].append(parents.iloc[1,pivot_point:])
        if offspring1.sum() < 0:
            print("An offspring got the sum of asset allocations negative before renormalization.")
        offspring1_renorm = offspring1 * parents_sum / offspring1.sum()
        offsprings = [offspring1_renorm]
        
        offspring2 = parents.iloc[1,0:pivot_point].append(parents.iloc[0,pivot_point:])
        if offspring2.sum() < 0:
            print("An offspring got the sum of asset allocations negative before renormalization.")
        offspring2_renorm = offspring2 * parents_sum / offspring2.sum()
        offsprings.append(offspring2_renorm)
    
    
    # Creating offsprings - Method 2
    if method == 'Two Points':
        pivot_point_1 = random.randint(1, Ngene-1)
        pivot_point_2 = random.randint(1, Ngene)
        
        while pivot_point_2 < pivot_point_1:
            pivot_point_2 = random.randint(1, Ngene)
            
        offspring1 = parents.iloc[0,0:pivot_point_1].append(parents.iloc[1,pivot_point_1:pivot_point_2]).append(parents.iloc[0,pivot_point_2:])
        
        if offspring1.sum() < 0:
            print(  "An offspring got the sum of asset \
                    allocations negative before renormalization.")
        offspring1_renorm = offspring1 * parents_sum / offspring1.sum()
        offsprings = [offspring1_renorm]
        
        offspring2 = parents.iloc[1,0:pivot_point_1].append(parents.iloc[0,pivot_point_1:pivot_point_2]).append(parents.iloc[1,pivot_point_2:])
        
        if offspring2.sum() < 0:
            print(  "An offspring got the sum of asset \
                    allocations negative before renormalization.")
        offspring2_renorm = offspring2 * parents_sum / offspring2.sum()
        offsprings.append(offspring2_renorm)
    
    
    # Creating offsprings - Method 3
    if method == 'Multi Points':
        if (n_points is None) or (n_points == 0):
            raise Exception("n_points must be specified.")
        if n_points%1!=0:
            raise Exception("n_points must be integer.")
        
        # Create a set of pivot points
        pivots = non_adjacent_random_list(0, Ngene, n_points)
        
        # Case where n_points is odd
        if n_points%2==1:
            offspring1 = parents.iloc[0,0:pivots[0]]
            offspring2 = parents.iloc[1,0:pivots[0]]
            for i in range(n_points-1):
                offspring1 = offspring1.append(parents.iloc[int((1+(-1)**i)/2), pivots[i]:pivots[i+1]])
                offspring2 = offspring2.append(parents.iloc[int((1+(-1)**(i+1))/2), pivots[i]:pivots[i+1]])
            offspring1 = offspring1.append(parents.iloc[1, pivots[n_points-1]:Ngene])
            offspring2 = offspring2.append(parents.iloc[0, pivots[n_points-1]:Ngene])
        
        # Case where n_points is even
        elif n_points%2==0:
            offspring1 = parents.iloc[0,0:pivots[0]]
            offspring2 = parents.iloc[1,0:pivots[0]]
            for i in range(n_points-1):
                offspring1 = offspring1.append(parents.iloc[int((1+(-1)**i)/2), pivots[i]:pivots[i+1]])
                offspring2 = offspring2.append(parents.iloc[int((1+(-1)**(i+1))/2), pivots[i]:pivots[i+1]])
            offspring1 = offspring1.append(parents.iloc[0, pivots[n_points-1]:Ngene])
            offspring2 = offspring2.append(parents.iloc[1, pivots[n_points-1]:Ngene])
        
        # For visual check
        # print(n_points)
        # print(pivots)
        # for d in range(parents.shape[1]):
        #     print(d, " ", parents.iloc[0,d], parents.iloc[1,d], offspring1[d], offspring2[d])
        
        
        # Renormalizations
        if offspring1.sum() < 0:
            print("An offspring got the sum of asset allocations negative before renormalization.")
        offspring1_renorm = offspring1 * parents_sum / offspring1.sum()
        offsprings = [offspring1_renorm]
        if offspring2.sum() < 0:
            print("An offspring got the sum of asset allocations negative before renormalization.")
        offspring2_renorm = offspring2 * parents_sum / offspring2.sum()
        offsprings.append(offspring2_renorm)
        
    
    # Check that the sums of asset allocations are the same as before
    sum_allocations = parents.iloc[0].sum()
    if sum_allocations - parents.iloc[1].sum() > 1E-5 :
        print([parents.iloc[0], parents.iloc[1], offsprings[0], offsprings[1]])
        print(parents.iloc[0].sum(), parents.iloc[1].sum(), offsprings[0].sum(), offsprings[1].sum())
        raise Exception("Parents must all have the same sum of assets allocations.")
    if sum_allocations - offsprings[0].sum() > 1E-5 :
        print([parents.iloc[0], parents.iloc[1], offsprings[0], offsprings[1]])
        print(parents.iloc[0].sum(), parents.iloc[1].sum(), offsprings[0].sum(), offsprings[1].sum())
        raise Exception("Offsprings and parents must have the same sum of assets allocations.")
    if sum_allocations - offsprings[1].sum() > 1E-5 :
        print([parents.iloc[0], parents.iloc[1], offsprings[0], offsprings[1]])
        print(parents.iloc[0].sum(), parents.iloc[1].sum(), offsprings[0].sum(), offsprings[1].sum())
        raise Exception("Offsprings and parents must have the same sum of assets allocations.")
        
    # Adding the mating date, which is also the birth date of the offsprings (no gestation period here).
    offsprings[0]["Born"] = mating_date
    offsprings[1]["Born"] = mating_date
    
    return offsprings



def get_offsprings(parents_values, mating_date, method='Single Point', n_points=None, name_indiv="Offspring"):
    """
    Takes all the pairs of parents and proceeds to their mating in order
    to produce two offsprings for each, putting all of them in a dataframe.
    
    Different methods can be used:
    - 'Single Point': using only one pivot value for exchange of genes.
    - 'Two Points': using two pivot values for exchange of genes.
    - 'Multi Points': using `n_points` pivot values for the exchange of genes.
    
    Parameters
    ----------
    parent_values : List of DataFrames
      List of data frames containing the parents genes.
    mating_date : str or Period date
      Mating date.
    method : str
      Methods of mating.
    n_points : int
      Number of points pivot values for method 'Multi Points'.
    name_indiv : str
      Name of the offspring individuals.
      
    Returns
    -------
    DataFrame
      Data frame of offsprings.
    """
    
    # Selecting only the columns with Asset allocations, i.e. the genes
    Npairs = len(parents_values)
    asset_columns = pd.DataFrame(parents_values[0]).filter(regex="Asset").columns.tolist() + ["Born"]
    
    # Creating the offsprings
    offsprings_pop = pd.DataFrame(columns=asset_columns)
    for x in range(Npairs):
        offsprings = mating_pair(parents_values[x], mating_date, method=method, n_points=n_points) # 2 offspring
        offsprings_pop.loc[name_indiv + str(x*2)] = offsprings[0]
        offsprings_pop.loc[name_indiv + str(x*2 + 1)] = offsprings[1]
        
    return offsprings_pop



def mutate_individual(input_individual, upper_limit, lower_limit, sum_target,
                      mutation_rate=2, method='Reset', standard_deviation = 0.001):
    """
    Makes the mutation of a single individual.
    
    Different methods can be used:
    - 'Gauss': normal-distributed modification of affected genes.
    - 'Reset': uniformly distributed modificatio of affected genes.
    
    Parameters
    ----------
    input_individual : DataFrame
      Individual we want to mutate.
    upper_limit : float
      Upper limit of the gene (asset allocation), before renormalization.
      Only for 'Reset' method.
    lower_limit : float
      Lower limit of the gene (asset allocation), before renormalization.
      Only for 'Reset' method.
    sum_target : float
      Tarket sum of genes (asset allocations) used for renormalization.
      Only for 'Reset' method.
    mutation_rate : int
      Number of mutations to apply.
    method : str
      Method used for mutations.
    standard_deviation: float
      Standard deviation of the mutation modification.
      Only for 'Gauss' method.
      
    Returns
    -------
    DataFrame
      Data frame of the mutated individual.
    """
    
    # Checks
    assert(isinstance(mutation_rate, int))
    
    # Initialization
    individual = input_individual.filter(regex="Asset")
    birth_date = input_individual["Born"]
    Ngene = len(individual)
    gene = []
    
    for x in range(mutation_rate):
        gene.append(random.randint(0, Ngene-1))
        while len(set(gene)) < len(gene):
            gene[x] = random.randint(0, Ngene-1)
    mutated_individual = individual.copy()
    
    # Saving the sum of assets
    sum_genes = mutated_individual.sum()
    
    # Generate the mutations:
    if method == 'Gauss':
        for x in gene:
            mutated_individual[x] = individual[x] + random.gauss(0, standard_deviation)
        normalization = sum_genes / mutated_individual.sum()
        for x in range(Ngene):
            mutated_individual[x] *= normalization
    
    if method == 'Reset':
        for x in gene:
            mutated_individual[x] = random.random() * (upper_limit-lower_limit) + lower_limit
        normalization = sum_target / mutated_individual.sum()
        for x in range(Ngene):
            mutated_individual[x] *= normalization
    
    # Adding back the birth date
    mutated_individual["Born"] = birth_date
    
    return mutated_individual


def mutation_set(num_indiv, num_genes, num_mut=0):
    """
    Prepares a list of genes to be mutated for the function `mutate_population`.
    
    Parameters
    ----------
    num_indiv : int
      Number of individuals.
    num_genes : int
      Number of genes.
    num_mut : int
      Number of mutations.
      
    Returns
    -------
    List of lists of int
      List of lists of genes positions to be mutated.
    """
    
    if num_genes <= 1 :
        return 0
    
    mutated_genes = []
    for i in range(num_indiv):
        tmp_set = [random.randint(0, num_genes-1), random.randint(0, num_genes-1)]
        while tmp_set[0] == tmp_set[1]:
            tmp_set[1] = random.randint(0, num_genes-1)
        mutated_genes.append(tmp_set)

    return mutated_genes


def mutate_population(input_individuals, upper_limit, lower_limit, sum_target,
                      mutation_rate=2, method='Reset', standard_deviation = 0.001):
    """
    Makes the mutation of a population of individuals.
        
    Different methods can be used:
    - 'Gauss': normal-distributed modification of affected genes.
    - 'Reset': uniformly distributed modificatio of affected genes.
    
    Parameters
    ----------
    input_individuals : DataFrame
      Population of individuals we want to mutate.
    upper_limit : float
      Upper limit of the gene (asset allocation), before renormalization.
      Only for 'Reset' method.
    lower_limit : float
      Lower limit of the gene (asset allocation), before renormalization.
      Only for 'Reset' method.
    sum_target : float
      Tarket sum of genes (asset allocations) used for renormalization.
      Only for 'Reset' method.
    mutation_rate : int
      Number of mutations to apply.
    method : str
      Method used for mutations.
    standard_deviation: float
      Standard deviation of the mutation modification.
      Only for 'Gauss' method.
      
    Returns
    -------
    DataFrame
      Data frame of the mutated individuals.
    """
    
    # Checks
    assert(isinstance(mutation_rate, int))
    
    # Initialization
    individuals = input_individuals.filter(regex="Asset")
    birth_dates = input_individuals["Born"]
    fitness_cols = input_individuals.filter(regex="Fit")
    M = individuals.shape[0]
    Ngene = individuals.shape[1]
    mutated_genes = mutation_set(num_indiv=M, num_genes=Ngene, num_mut=mutation_rate)
    mutated_individuals = individuals.copy()
    
    # Making mutations happen and renormalizing assets allocation to keep their sum constant
    for i in range(M):
        if method == 'Gauss':
            sum_genes = mutated_individuals.iloc[i].sum()
            for x in mutated_genes[i]:
                mutated_individuals.iloc[i,x] = mutated_individuals.iloc[i,x] + random.gauss(0, standard_deviation)
            normalization = sum_genes / mutated_individuals.iloc[i].sum()
            for x in range(Ngene):
                mutated_individuals.iloc[i,x] *= normalization

        if method == 'Reset':
            for x in mutated_genes[i]:
                mutated_individuals.iloc[i,x] = random.random() * (upper_limit-lower_limit) + lower_limit
            normalization = sum_target / mutated_individuals.iloc[i].sum()
            for x in range(Ngene):
                mutated_individuals.iloc[i,x] *= normalization

    # Adding back the birth date
    mutated_individuals["Born"] = birth_dates
    
    # Adding back the earlier Fitness columns
    for col in fitness_cols.columns:
        mutated_individuals[col] = fitness_cols[col]
    
    return mutated_individuals



def next_generation(elite, gen, market, current_eval_date, next_eval_date,
                    upper_limit, lower_limit, sum_target, mutation_rate, standard_deviation,
                    fitness_lambda=0.5, fitness_method="Last Return and Vol",
                    pairing_method="Fittest", mating_method="Single Point", n_points=None,
                    mutation_method="Gauss", selection_method="Fittest Half",
                    return_propag=False, name_indiv="Offspring", date_format="%Y-%m", Verbose=False):

    """
    Computes the next generation of individuals from selection, forming groups
    with elite, generating offsprings, creating mutations in the offsprings,
    recomputing fitness and sorting the new population.
    
    Parameters
    ----------
    elite : DataFrame
      Generation of elites we start with.
    gen : DataFrame
      Generation of individuals we start with.
    market : DataFrame
      Market which serves as a basis for propagation.
    current_eval_date : DataFrame
      Present date on which we evaluate the individuals.
    next_eval_date : DataFrame
      Target date until which we want to make the individuals evolve.
    upper_limit : float
      Upper limit of the gene (asset allocation), before renormalization.
      Only for 'Reset' method.
    lower_limit : float
      Lower limit of the gene (asset allocation), before renormalization.
      Only for 'Reset' method.
    sum_target : float
      Tarket sum of genes (asset allocations) used for renormalization.
      Only for 'Reset' method.
    mutation_rate : int
      Number of mutations to apply.
    standard_deviation: float
      Standard deviation of the mutation modification.
      Only for 'Gauss' method.
    fitness_lambda : float in [0,1]
      Parameter of the fitness calculation to decide between returns and volatility.
    fitness_method : str
      Method used to evaluate fitness.
        - 'Last Return': uses the value of return at the end of propagation,
                         i.e. the value just before evaluation date.
        - 'Last Return and Vol': combines last return with estimation of volatility,
                                 the dosage between the two being tuned by lambda.
        - 'Avg Return and Vol': uses average return during the propagation and the
                                estimation of volatility, also using lambda.
        - 'Sharpe Ratio': uses the ratio of average return over estimation of volatility.
    pairing_method : str
      Method used for pairing.
        - 'Fittest': the top fitness individuals are paired in order, this does not check
                     if the elite is fit however (we let evolution do that).
                     The selected individuals should be ranked from fittest to less fit
                     if things have been done in the proper order before.
        - 'Random': the pairing is done at random, but no individual
                    can reproduce more than once.
        - 'Weighted Random': the pairing is done with higher probability among high fitness
                             individuals. Some individuals may reproduce several times.
    mating_method : str
      Method used for mating:
        - 'Single Point': using only one pivot value for exchange of genes.
        - 'Two Points': using two pivot values for exchange of genes.
        - 'Multi Points': using `n_points` pivot values for the exchange of genes.
    mutation_method : str
      Method used for the mutations.
        - 'Gauss': normal-distributed modification of affected genes.
        - 'Reset': uniformly distributed modification.
    selection_method : str
      Method used for selection.
        - 'Fittest Half': the first half of the top fitness is kept.
        - 'Random': rows are picked up at random, but can't be the same.
        - 'Roulette Wheel': rows are picked up at random, but with a preference
                            for high cumulative sum (high fitness).
    return_propag : bool
      Option to return the propagations or not.
    name_indiv : str
      String to set the name of the offsprings.
    date_format : str
      Format of dates used in the dataframes.
    Verbose : bool
      Verbose option.
    
    Returns
    -------
    DataFrame
      Data frame with the next generation, including parents and offsprings.
    
    Raises
    ------
    ValueError
      To make sure than elite and current generation of the same columns.
    
    Notes
    -----
      None
    
    Examples
    --------
      None
    """
    
    # Checks
    assert(fitness_lambda>=0 and fitness_lambda<=1)
    assert(fitness_method in ["Last Return", "Last Return and Vol",
                              "Avg Return and Vol", "Sharpe Ratio"])
    assert(pairing_method in ["Fittest", "Random", "Weighted Random"])
    assert(mating_method in ["Single Point", "Two Points", "Multi Points"])
    assert(mutation_method in ["Gauss", "Reset"])
    assert(selection_method in ["Fittest Half", "Random", "Roulette Wheel"])
    
    # Check columns are the same:
    if elite.columns.tolist() != gen.columns.tolist():
        ValueError("Elite and current generation must have the same columns!")
    
    # Next generation empty dataframe
    next_gen = pd.DataFrame(data=None, columns=gen.columns)
    
    # Select the population allowed to reproduce
    if Verbose: print(".... doing selection")
    selected = selection(gen, method=selection_method)
    if Verbose: print("    ", selected.index.values)

    # Combine them with the elite
    if Verbose: print(".... forming pairs")
    testM = elite.shape[0] + selected.shape[0]
    
    if testM % 2 != 0:
        name_of_indiv_to_remove = selected.index[-1]
        selected.drop([name_of_indiv_to_remove], axis=0, inplace=True)
        print(name_of_indiv_to_remove, " has been removed as it was the last \
              element of a selection with odd number of individuals to pair.")
        
    parents_pairs, parents_values = pairing(elite, selected, method=pairing_method)
    if Verbose: print("    ", parents_pairs)
    
    # Generating offsprings
    if Verbose: print(".... generating offsprings")
    offsprings = get_offsprings(parents_values, current_eval_date,
                                method=mating_method, n_points=n_points,
                                name_indiv=name_indiv)
    if Verbose: print("    ", offsprings.index.values)
    
    # Mutating selected individuals and offsprings, but not the elite
    name_col_to_drop = selected.filter(regex="CumSum").columns
    selected.drop(columns=name_col_to_drop, inplace=True)
    
    if Verbose: print(".... appending offsprings to selection")
    unmutated = selected.append(offsprings)
    if Verbose: print("    ", unmutated.index.values)
        
    if Verbose: print(".... mutating offsprings")
    mutated = mutate_population(unmutated, upper_limit, lower_limit, sum_target,
                                mutation_rate, mutation_method, standard_deviation)
    
    if Verbose: print(".... forming mutated generation")
    mutated_gen = get_generation(mutated, market, current_eval_date, next_eval_date,
                                 lamb=fitness_lambda, fitness_method=fitness_method,
                                 return_propag=False, date_format=date_format)
    if Verbose: print("    ", mutated_gen.index.values)
    
    # Combine mutated population with the elite
    if Verbose: print(".... recombining elite with mutated individuals")
    unsorted_individuals = elite.append(mutated_gen)
    if Verbose: print("    ", unsorted_individuals.index.values)
    
    # Compute fitness of that new generation
    # get generation also sorts the individuals by fitness
    if Verbose: print(".... getting the generation of combined elite and mutated individuals")
    if return_propag == True:
        sorted_next_gen, propagation = get_generation(unsorted_individuals, market,
                                                      current_eval_date, next_eval_date,
                                                      lamb=fitness_lambda, return_propag=True,
                                                      date_format=date_format)
        return sorted_next_gen, propagation
    else:
        sorted_next_gen = get_generation(unsorted_individuals, market,
                                         current_eval_date, next_eval_date,
                                         lamb=fitness_lambda, return_propag=False,
                                         date_format=date_format)
        return sorted_next_gen


def get_elite_and_individuals(generation, elite_rate=0.2, renaming=True):
    """
    Creates the elite and non-elite individual populations by ranking
    them according to the last fitness and proceeding to a simple cut,
    the highest fitness individuals going to the new elite.
    Some formerly non-elite can become new elite and formerly elite
    can become non-elite depending on values.
    
    Parameters
    ----------
    generation : DataFrame
      Generation we want to build new elite and non-elite from.
    elite_rate : float in [0,1]
      Rate of individuals going into the elite.
    renaming : bool
      Option to rename the individuals.
    
    Returns
    -------
    DataFrame
      Data Frame of the new elite population.
    DataFrame
      Data Frame of the new non-elite population.
    """
    
    # Checks
    assert(elite_rate>=0 and elite_rate <=1)
    
    # Initialization
    M = generation.shape[0]
    fitness_cols = generation.filter(regex="Fit").columns
    
    # Just to make sure that the generation is sorted
    generation.sort_values(by=fitness_cols[-1], ascending=False, inplace=True)
    
    # Decide for a cut in the top fitness individuals
    cut = int(M * elite_rate)
    
    # Applying the cut - Form the elite
    new_elite = generation.iloc[:cut]
    if renaming:
        new_elite_names = ["New Elite " + str(x) for x in range(cut)]
        new_elite.index = new_elite_names

    # Applying the cut - Form the "non-elite" individuals
    new_individuals = generation.iloc[cut:]
    if renaming:
        new_individuals_names = ["New Individual " + str(x) for x in range(M-cut)]
        new_individuals.index = new_individuals_names
    
    return new_elite, new_individuals


def fitness_similarity_check(generation, number_of_similarity, precision_decimals=1):
    """
    Checks the fitness similarity based on the last fitness calculation.
    This function tests if a certain number 'number_of_similarity' of individuals
    converge to the same fitness value, up to a precision 'precision_decimals'.
    
    Parameters
    ----------
    generation : DataFrame
      Generation that we consider.
    number_of_similarity : int
      Number of individuals to consider for similarity check.
    precision_decimals : int
      Precision decimals of the check.
    
    Returns
    -------
    bool
      True if generation passed similarity check, False otherwise.
    
    Notes
    -----
      This function should be applied after the new generation is created
      and before we do a split into Elite + Non-Elite individuals.
    """
    
    # Checks
    assert(isinstance(number_of_similarity, int))
    assert(isinstance(precision_decimals, int))
    
    # Initialization
    result = False
    similarity = 0
    max_fitness = generation.filter(regex="Fit").iloc[:,-1]
    
    for n in range(len(max_fitness)-1):
        if round(max_fitness[n], precision_decimals) == round(max_fitness[n+1], precision_decimals):
            similarity += 1
        else:
            similarity = 0
    if similarity == number_of_similarity-1:
        result = True
        
    return result


def sum_top_fitness(generation, num_elements=4):
    """
    Computes the sum of the top elements in the population.
    
    Parameters
    ----------
    generation : DataFrame
      Generation that we consider.
    num_elements : int
      Number of elements to consider from the top.
      
    Returns
    -------
    float
      The sum of fitness from the top elements.
    """
    
    gen = generation.filter(regex="Fit").iloc[:,-1]
    
    if num_elements <= gen.shape[0]:
        sum_top_fitness = gen[0:num_elements].sum()
    else:
        ValueError("Generation does not have enough elements.")

    return sum_top_fitness


def plot_compare_genomes(indiv1, indiv2, names=("Indiv1", "Indiv2")):
    """
    Plots the genes of two individual next to each other.
    
    This function can be used for example when we want to
    compare the genes of an offspring and its mutated version.
    
    Parameters
    ----------
    indiv1 : Pandas.Series
      First individual.
    indiv2 : Pandas.Series
      Second individual.
    names : 2-tuple of str
      Names we want for display.
    
    Returns
    -------
    None
      None
    """

    # Checks
    if "Born" in indiv1.index:
        indiv1.drop(index="Born", inplace=True)
    if "Born" in indiv2.index:
        indiv2.drop(index="Born", inplace=True)
    assert(indiv1.index.tolist() == indiv2.index.tolist())
        
    # Building a data frame
    tmp = pd.DataFrame(columns=indiv1.index.tolist())
    tmp.index.names = ["Individuals"]
    tmp.loc[names[0]] = indiv1
    tmp.loc[names[1]] = indiv2

    # Plotting
    tmp.transpose().plot.bar(figsize=(10,5))
    
    return None



#---------#---------#---------#---------#---------#---------#---------#---------#---------#


