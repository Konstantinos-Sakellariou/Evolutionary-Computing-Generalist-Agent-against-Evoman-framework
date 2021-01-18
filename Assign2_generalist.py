# imports framework
import sys

sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
import os
import numpy

experiment_name = 'groupall'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[1,2,3,4,5,6,7,8],
                  multiplemode="yes",
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")

# default environment fitness is assumed for experiment

env.state_to_log()  # checks environment state

####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###

ini = time.time()  # sets time marker

# genetic algorithm params

run_mode = 'train'  # train or test

sol_per_pop = 100
# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5
# Defining the population size.
print(n_vars)
pop_size = (sol_per_pop, n_vars)  # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.


# runs simulation
def simulation(env, x):
    f, p, e, t = env.play(pcont=x)
    return f, p, e


# evaluation
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env, y), x)))



def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = numpy.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents


def crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually, it is at the center.
    crossover_point = numpy.uint8(offspring_size[1] / 2)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k % parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k + 1) % parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring


def mutation(offspring_crossover, num_mutations=3):
    mutations_counter = numpy.uint8(offspring_crossover.shape[1] / num_mutations)
    # Mutation changes a number of genes as defined by the num_mutations argument. The changes are random.
    for idx in range(offspring_crossover.shape[0]):
        gene_idx = mutations_counter - 1
        for mutation_num in range(num_mutations):
            # The random value to be added to the gene.
            random_value = numpy.random.uniform(-1.0, 1.0, 1)
            offspring_crossover[idx, gene_idx] = offspring_crossover[idx, gene_idx] + random_value
            gene_idx = gene_idx + mutations_counter
    return offspring_crossover


# loads file with the best solution for testing
if run_mode =='test':

    bsol = np.loadtxt(experiment_name+'/best.txt')
    print( '\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed','normal')
    evaluate([bsol])

    sys.exit(0)


if not os.path.exists(experiment_name + '/evoman_solstate'):

    print('\nNEW EVOLUTION\n')
    # Creating the initial population

    new_population = numpy.random.uniform(low=-1, high=1, size=pop_size)
    fit_pop = evaluate(new_population)
    best = np.argmax(fit_pop)
    pllife = env.get_playerlife()
    enlife = env.get_enemylife()
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)
    ini_g = 0
    solutions = [new_population, fit_pop]
    env.update_solutions(solutions)

else:

    print('\nCONTINUING EVOLUTION\n')

    env.load_state()
    new_population = env.solutions[0]
    fit_pop = env.solutions[1]

    best = np.argmax(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)

    # finds last generation number
    file_aux = open(experiment_name + '/gen.txt', 'r')
    ini_g = int(file_aux.readline())
    file_aux.close()

# saves results for first pop
file_aux = open(experiment_name + '/results.txt', 'a')
file_aux.write('\n\ngen best mean std playerlife enemylife')
print('\n GENERATION ' + str(ini_g) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(
    round(std, 6)) )
file_aux.write(
    '\n' + str(ini_g) + ' ' + str(round(fit_pop[best], 6)) + ' ' + str(round(mean, 6)) + ' ' + str(round(std, 6)))
file_aux.close()

fileopen = open(experiment_name + '/playerenemy.txt', 'a')
fileopen.write('\n\n playerlife enemylife')
fileopen.write('\n' + str(pllife) + '' + str(enlife))

best_outputs = []
num_generations = 10
num_parents_mating = 2

for generation in range(1, num_generations):
    print("Generation : ", generation)

     # Measuring the fitness of each chromosome in the population.

    fitness = evaluate(new_population)

    print("Fitness")
    print(fitness)
    best = np.argmax(fitness)  # best solution in generation
    standard = numpy.std(fitness)
    average = numpy.mean(fitness)
    pllife = env.get_playerlife()
    enlife = env.get_enemylife()

    #best_outputs.append(numpy.max(numpy.sum(new_population, axis=1)))
    # The best result in the current iteration.
    #print("Best result : ", numpy.max(numpy.sum(new_population, axis=1)))

    # Selecting the best parents in the population for mating.
    parents = select_mating_pool(new_population, fitness,
                                 num_parents_mating)
    print("Parents")
    print(parents)

    # Generating next generation using crossover.
    offspring_crossover = crossover(parents,
                                    offspring_size=(pop_size[0] - parents.shape[0], n_vars))
    print("Crossover")
    print(offspring_crossover)

    # Adding some variations to the offspring using mutation.
    offspring_mutation = mutation(offspring_crossover, num_mutations=3)
    print("Mutation")
    print(offspring_mutation)

    # Creating the new population based on the parents and offspring.
    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation

    final_best_fitness = numpy.argmax(fitness)


    # saves results
    file_aux = open(experiment_name + '/results.txt', 'a')
    print('\n GENERATION ' + str(generation) + ' ' + str(fitness[final_best_fitness]) + ' ' + str(average) + ' ' + str(
        standard))
    file_aux.write(
        '\n' + str(generation) + ' ' + str(fitness[final_best_fitness]) + ' ' + str(average) + ' ' + str(standard))
    file_aux.close()

    fileopen = open(experiment_name + '/playerenemy.txt', 'a')
    fileopen.write('\n\n playerlife enemylife')
    fileopen.write('\n' + str(pllife) + '' + str(enlife))

    # saves generation number
    file_aux = open(experiment_name + '/gen.txt', 'w')
    file_aux.write(str(generation))
    file_aux.close()

    # saves file with the best solution
    np.savetxt(experiment_name + '/best.txt', new_population[final_best_fitness])

    # saves simulation state
    solutions = [new_population, fitness]
    env.update_solutions(solutions)
    env.save_state()

# Getting the best solution after iterating finishing all generations.
# At first, the fitness is calculated for each solution in the final generation.
fitness = evaluate(new_population)
# Then return the index of that solution corresponding to the best fitness.
best_match_idx = numpy.where(fitness == numpy.max(fitness))

print("Best solution : ", new_population[best_match_idx, :])
np.savetxt(experiment_name + '/best2.txt', new_population[best_match_idx])
print("Best solution fitness : ", fitness[best_match_idx])

fim = time.time()  # prints total execution time for experiment
print('\nExecution time: ' + str(round((fim - ini) / 60)) + ' minutes \n')

file = open(experiment_name + '/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
file.close()

env.state_to_log()  # checks environment state