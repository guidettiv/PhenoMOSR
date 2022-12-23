import logging
import os
import random
import time

import numpy as np
import pandas as pd
import pygmo as pg
from joblib.parallel import Parallel, delayed
from symbolic_regression.Program import Program
from symbolic_regression.multiobjective.fitness.Base import BaseFitness

from loky import get_reusable_executor

backend_parallel = 'loky'


class SymbolicRegressor:

    def __init__(
        self,
        checkpoint_file: str = None,
        checkpoint_frequency: int = -1,
        const_range: tuple = None,
        parsimony=0.9,
        parsimony_decay=0.9,
        population_size: int = 100,
        tournament_size: int = 10,
        genetic_operators_frequency: dict = {},
    ) -> None:
        """ This class implements the basic features for training a Symbolic Regression algorithm

        Args:
            - const_range: this is the range of values from which to generate constants in the program
            - fitness_functions: the functions to use for evaluating programs' performance
            - parsimony: the ratio to which a new operation is chosen instead of a terminal node in program generations
            - parsimony_decay: a modulation parameter to decrease the parsimony and limit program generation depth
            - tournament_size: this modulate the tournament selection and set the dimension of the selection
        """

        # Regressor Configuration
        self.checkpoint_file: str = checkpoint_file
        self.checkpoint_frequency: int = checkpoint_frequency
        self.elapsed_time: int = 0
        self.features: list = None
        self.operations: list = None
        self.population_size: int = population_size

        # Population Configuration
        self.average_complexity: float = None
        self.const_range: tuple = const_range
        self.parsimony: float = parsimony
        self.parsimony_decay: float = parsimony_decay
        self.tournament_size: int = tournament_size

        # Training Configuration
        self.best_program = None
        self.best_programs_history: list = []
        self.converged_generation: int = None
        self.fitness_functions: list[BaseFitness] = None
        self.first_pareto_front_history: list = []
        self.fpf_hypervolume: float = None
        self.fpf_hypervolume_history: list = []
        self.generations_to_train: int = None
        self.generation: int = None
        self.genetic_operators_frequency: dict = genetic_operators_frequency
        self.population: list = None
        self.status: str = "Uninitialized"
        self.training_duration: int = None

    def save_model(self, file: str):
        import pickle

        with open(file, "wb") as f:
            pickle.dump(self, f)

    def load_model(self, file: str):
        import pickle

        with open(file, "rb") as f:
            return pickle.load(f)

    def _create_pareto_front(self):

        pareto_front = []

        # Loop over the entire matrix, can be optimised to do only the triangular matrix
        for p1 in self.population:
            p1.rank = float('inf')

            if not p1.is_valid:
                continue
            p1.programs_dominates = []
            p1.programs_dominated_by = []

            for p2 in self.population:
                if p1 == p2 or not p2.is_valid:
                    continue

                if self.dominance(p1, p2):
                    p1.programs_dominates.append(p2)
                elif self.dominance(p2, p1):
                    p1.programs_dominated_by.append(p2)

            if len(p1.programs_dominated_by) == 0:
                p1.rank = 1
                pareto_front.append(p1)

        i = 1

        # Set the belonging pareto front to every element of the population

        while pareto_front:
            next_pareto_front = []

            for p1 in pareto_front:
                if not p1.is_valid:
                    continue
                for p2 in p1.programs_dominates:
                    if not p2.is_valid:
                        continue
                    p2.programs_dominated_by.remove(p1)

                    if len(p2.programs_dominated_by) == 0:
                        p2.rank = i + 1
                        next_pareto_front.append(p2)

            i += 1

            pareto_front = next_pareto_front

    def _crowding_distance(self):

        rank_iter = 1
        pareto_front = self.extract_pareto_front(rank=rank_iter)

        while pareto_front:  # Exits when extract_pareto_front return an empty list
            for ftn in self.fitness_functions:

                fitness_label = ftn.label
                # This exclude the fitness functions which are set not to be minimized
                if not ftn.minimize:
                    continue

                # Highest fitness first for each objective
                pareto_front.sort(
                    key=lambda p: p.fitness[fitness_label], reverse=True)

                norm = pareto_front[0].fitness[fitness_label] - \
                    pareto_front[-1].fitness[fitness_label] + 1e-20

                for index, program in enumerate(pareto_front):
                    if index == 0 or index == len(pareto_front) - 1:
                        program.crowding_distance = float('inf')
                    else:
                        delta = pareto_front[index - 1].fitness[fitness_label] - \
                            pareto_front[index + 1].fitness[fitness_label]

                        program.crowding_distance = delta / norm

            rank_iter += 1
            pareto_front = self.extract_pareto_front(rank=rank_iter)

    @staticmethod
    def dominance(program1: Program, program2: Program) -> bool:
        """
        Return True if program1 dominate over program2.
        It dominates if all the fitnesses are equal or better and at least one fitness is
        better
        """

        # How many element in the p1.fitness are less than p2.fitness
        at_least_one_less_than_zero = False
        all_less_or_eq_than_zero = True

        if program1.program and program2.program:
            for this_fitness in program1.fitness.keys():
                # Ignore the fitness which are not to be optimized
                if program1.is_fitness_to_minimize[this_fitness] == False:
                    continue

                d = abs(program1.fitness[this_fitness]) - \
                    abs(program2.fitness[this_fitness])

                if d < 0:
                    at_least_one_less_than_zero = True
                if d > 0:
                    all_less_or_eq_than_zero = False

            return at_least_one_less_than_zero and all_less_or_eq_than_zero

        return False

    def drop_duplicates(self, inplace: bool = False) -> list:
        """ This method removes duplicated programs

        Programs are considered duplicated if they have the same performance

        Args:
            - inplace: allow to overwrite the current population or duplicate the object
        """

        for index, p in enumerate(self.population):
            if p.is_valid and not p._is_duplicated:
                for p_confront in self.population[index + 1:]:
                    if p.is_duplicate(p_confront):
                        p_confront._is_duplicated = True  # Makes p.is_valid = False

        if inplace:
            self.population = list(
                filter(lambda p: p._is_duplicated == False, self.population))
            return self.population

        return list(
            filter(lambda p: p._is_duplicated == False, self.population))

    def drop_invalids(self, inplace: bool = False) -> list:
        """ This program removes invalid programs from the population

        A program can be invalid when mathematical operation are not possible
        or if the siplification generated operation which are not supported.

        Args:
            - inplace: allow to overwrite the current population or duplicate the object
        """
        if inplace:
            self.population = list(
                filter(lambda p: p.is_valid == True, self.population))
            return self.population

        return list(filter(lambda p: p.is_valid == True, self.population))

    def extract_pareto_front(self, rank: int):
        pareto_front = []
        for p in self.population:
            if p and p.rank == rank:
                pareto_front.append(p)

        return pareto_front

    def fit(self, data: pd.DataFrame, features: list, operations: list, fitness_functions: list, generations_to_train: int, n_jobs: int = -1, stop_at_convergence: bool = False, verbose: int = 0) -> None:
        self.data = data
        self.features = features
        self.operations = operations
        self.fitness_functions = fitness_functions
        self.generations_to_train = generations_to_train
        self.n_jobs = n_jobs
        self.stop_at_convergence = stop_at_convergence
        self.verbose = verbose

        if not self.generation:
            self.generation = 0

        start = time.perf_counter()
        try:
            self._run_training()
        except KeyboardInterrupt:
            self.generation -= 1  # The increment is applied even if the generation is interrupted
            stop = time.perf_counter()
            self.training_duration = stop - start
            self.status = "Interrupted by KeyboardInterrupt"
            logging.warning(f"Training terminated by a KeyboardInterrupt")
            return
        stop = time.perf_counter()
        self.training_duration = stop - start

    def generate_individual(self, data: pd.DataFrame, features: list, operations: list, const_range: tuple, fitness_functions: list, parsimony: float = 0.90, parsimony_decay: float = 0.90):
        """
        We need data in order to evaluate the fitness of the program because
        this method is executed in parallel.
        """
        p = Program(
            features=features,
            operations=operations,
            const_range=const_range,
            parsimony=parsimony,
            parsimony_decay=parsimony_decay
        )

        p.init_program()

        p.evaluate_fitness(fitness_functions=fitness_functions, data=data)

        return p

    def _get_offspring(self):
        """ This function generate an offspring of a program from the current population

        The offspring is a program to which a genetic alteration has been applied.
        The possible operations are as follow:
            - crossover: a random subtree from another program replaces
                a random subtree of the current program
            - mutation: a random subtree of the current program is replaced by a newly
                generated subtree
            - randomization: it is a crossover with a portion of the same program instead of a portion
                of another program
            - deletion: a random subtree is deleted from the current program
            - insertion: a newly generated subtree is inserted in a random spot of the current program
            - operator mutation: a random operation is replaced by another with the same arity
            - leaf mutation: a terminal node (feature or constant) is replaced by a different one
            - simplify: uses a sympy backend to simplify the program ad reduce its complexity
            - do nothing: in this case no mutation is applied

        The frequency of which those operation are applied is determined by the dictionary
        genetic_operations_frequency in which the relative frequency of each of the desired operations
        is expressed with integers. The higher the number the likelier the operation will be chosen.

        The program to which apply the operation is chosen using the tournament_selection, a method
        that identify the best program among a random selection of k programs from the population.

        Args:
            population: The population of programs from which to extract the program for the mutation
            fitness: The list of fitness functions for this task
            tournament_size: The size of the pool of random programs from which to choose in for the mutations
            genetic_operations_frequency: The relative frequency with which to coose the genetic operation to apply
            generations: The number of training generations (used to appropriately behave in the first one)
        """

        # This allow to randomly chose a genetic operation to
        ops = list()
        for op, freq in self.genetic_operators_frequency.items():
            ops += [op] * freq
        gen_op = random.choice(ops)

        program1 = self._tournament_selection(population=self.population,
                                              tournament_size=self.tournament_size,
                                              generation=self.generation)

        if program1 is None or not program1.is_valid:
            return program1

        if gen_op == 'crossover':
            program2 = self._tournament_selection(population=self.population,
                                                  tournament_size=self.tournament_size,
                                                  generation=self.generation)
            if program2 is None or not program2.is_valid:
                return program1
            p_ret = program1.cross_over(other=program2, inplace=False)

        elif gen_op == 'randomize':
            # Will generate a new tree as other
            p_ret = program1.cross_over(other=None, inplace=False)

        elif gen_op == 'mutation':
            p_ret = program1.mutate(inplace=False)

        elif gen_op == 'delete_node':
            p_ret = program1.delete_node(inplace=False)

        elif gen_op == 'insert_node':
            p_ret = program1.insert_node(inplace=False)

        elif gen_op == 'mutate_operator':
            p_ret = program1.mutate_operator(inplace=False)

        elif gen_op == 'mutate_leaf':
            p_ret = program1.mutate_leaf(inplace=False)

        elif gen_op == 'simplification':
            p_ret = program1.simplify(inplace=False)

        elif gen_op == 'recalibrate':
            p_ret = program1.recalibrate(inplace=False)

        elif gen_op == 'do_nothing':
            p_ret = program1
        else:
            logging.warning(
                f'Supported genetic operations: crossover, delete_node, do_nothing, insert_node, mutate_leaf, mutate_operator, simplification, mutation and randomize'
            )
            return program1

        # Add the fitness to the object after the cross_over or mutation
        p_ret.evaluate_fitness(
            fitness_functions=self.fitness_functions, data=self.data)

        # Reset the hash to force the re-computation
        p_ret._hash = None

        return p_ret

    @property
    def hypervolume(self):

        fitness_to_hypervolume = []
        for fitness in self.fitness_functions:
            if fitness.hypervolume_reference and fitness.minimize:
                fitness_to_hypervolume.append(fitness)

        references = [
            fitness.hypervolume_reference for fitness in fitness_to_hypervolume]
        points = [[p.fitness[ftn.label] for ftn in fitness_to_hypervolume]
                  for p in self.first_pareto_front]

        try:
            for index, p_list in enumerate(points):
                for p_i, r_i in zip(p_list, references):
                    if p_i > r_i and not p_i == float('inf'):
                        logging.warning(
                            f"Point {p_i} is outside of the reference {r_i}. Reference point will be set to {p_i + 1e-1}")
                        references[index] = p_i + 1e-1
            self.fpf_hypervolume = pg.hypervolume(points).compute(references)
            self.fpf_hypervolume_history.append(self.fpf_hypervolume)

        except ValueError:
            self.fpf_hypervolume = 0
            self.fpf_hypervolume_history.append(0)

        return self.fpf_hypervolume

    def _run_training(self):
        if not self.population:
            logging.info(f"Initializing population")
            self.status = "Generating population"
            self.population = Parallel(
                n_jobs=self.n_jobs,
                backend=backend_parallel)(delayed(self.generate_individual)(
                    data=self.data,
                    features=self.features,
                    operations=self.operations,
                    const_range=self.const_range,
                    fitness_functions=self.fitness_functions,
                    parsimony=self.parsimony,
                    parsimony_decay=self.parsimony_decay,
                ) for _ in range(self.population_size))

        else:
            logging.info("Fitting with existing population")

        while True:
            if self.generation > 0 and self.generations_to_train <= self.generation:
                logging.info(
                    f"The model already had trained for {self.generation} generations")
                self.status = "Terminated: generations completed"
                return

            self.generation += 1

            start_time_generation = time.perf_counter()

            seconds_iter = round(self.elapsed_time /
                                 self.generation, 1) if self.generation else 0
            timing_str = f"{self.elapsed_time} sec, {seconds_iter} sec/generation"

            print("############################################################")
            print(
                f"Generation {self.generation}/{self.generations_to_train} - {timing_str}")

            self.status = "Generating offspring"

            offsprings = []
            m_workers = self.n_jobs if self.n_jobs > 0 else os.cpu_count()

            executor = get_reusable_executor(max_workers=m_workers)
            offsprings = list(set(executor.map(
                self._get_offspring, timeout=120,
                initargs=(
                    self.population,
                    self.fitness_functions,
                    self.generation,
                    self.tournament_size,
                    self.genetic_operators_frequency
                ))))

            self.population += offsprings

            # Removes all non valid programs in the population
            logging.debug(f"Removing duplicates")
            before_cleaning = len(self.population)

            self.drop_duplicates(inplace=True)

            after_drop_duplicates = len(self.population)
            logging.debug(
                f"{before_cleaning-after_drop_duplicates}/{before_cleaning} duplicates programs removed")

            self.drop_invalids(inplace=True)

            after_cleaning = len(self.population)
            if before_cleaning != after_cleaning:
                logging.debug(
                    f"{after_drop_duplicates-after_cleaning}/{after_drop_duplicates} invalid programs removed"
                )

            # Integrate population in case of too many invalid programs
            if len(self.population) < self.population_size * 2:
                self.status = "Refilling population"
                missing_elements = 2*self.population_size - \
                    len(self.population)

                logging.info(
                    f"Population of {len(self.population)} elements is less than 2*population_size:{self.population_size*2}. Integrating with {missing_elements} new elements")

            batch_size = self.n_jobs if self.n_jobs > 0 else os.cpu_count()
            refill = Parallel(
                n_jobs=self.n_jobs, batch_size=batch_size,
                backend=backend_parallel)(delayed(self.generate_individual)(
                    data=self.data,
                    features=self.features,
                    operations=self.operations,
                    const_range=self.const_range,
                    fitness_functions=self.fitness_functions,
                    parsimony=self.parsimony,
                    parsimony_decay=self.parsimony_decay,
                ) for _ in range(missing_elements))

            self.population += refill

            self.status = "Creating pareto front"
            self._create_pareto_front()

            self.status = "Creating crowding distance"
            self._crowding_distance()

            self.population.sort(
                key=lambda p: p.crowding_distance, reverse=True)
            self.population.sort(key=lambda p: p.rank, reverse=False)
            self.population = self.population[:self.population_size]

            self.best_program = self.population[0]
            self.best_programs_history.append(self.best_program)
            self.first_pareto_front_history.append(
                list(self.first_pareto_front))

            self.average_complexity = np.mean(
                [p.complexity for p in self.population])
            if self.verbose > 1:
                print()
                print(
                    f"Population of {len(self.population)} elements and average complexity of {self.average_complexity} and 1PF hypervolume of {self.hypervolume}\n")
                print(f"\tBest individual(s) in the first Pareto Front")
                first_p_printed = 0
                for p in self.population:
                    if p.rank > 1:
                        continue
                    print(f'{first_p_printed})\t{p.program}')
                    print()
                    print(f'\t{p.fitness}')
                    print()
                    first_p_printed += 1

            end_time_generation = time.perf_counter()

            if self.best_program.converged:
                converged_time = time.perf_counter()
                if not self.converged_generation:
                    self.converged_generation = self.generation
                logging.info(
                    f"Training converged after {self.converged_generation} generations.")
                if self.stop_at_convergence:
                    self.drop_duplicates(inplace=True)
                    self.status = "Terminated: converged"
                    return
            if self.checkpoint_file and self.checkpoint_frequency > 0 and self.generation % self.checkpoint_frequency == 0:
                try:
                    self.save_model(file=self.checkpoint_file)
                except FileNotFoundError:
                    logging.warning(
                        f'FileNotFoundError raised in checkpoint saving')

            # Use generations = -1 to rely only on convergence (risk of infinite loop)
            if self.generations_to_train > 0 and self.generation == self.generations_to_train:
                logging.info(
                    f"Training terminated after {self.generation} generations")
                self.drop_duplicates(inplace=True)
                self.status = "Terminated: generations completed"
                return

            self.elapsed_time += end_time_generation - start_time_generation

    def _tournament_selection(self):
        """ The tournament selection is used to choose the programs to which apply genetic operations

        Firstly a random selection of k elements from the population is selected and 
        the best program among them is chosen.
        """

        tournament_members = random.choices(
            self.population, k=self.tournament_size)

        best_member = tournament_members[0]

        for member in tournament_members:
            if member is None or not member.is_valid:
                continue

            if self.generation == 0:
                try:
                    if best_member > member:
                        best_member = member
                except IndexError:
                    pass  # TODO fix

            else:

                # In the other generations use the pareto front rank and the crowding distance
                if best_member == None or \
                        member.rank < best_member.rank or \
                        (member.rank == best_member.rank and
                            member.crowding_distance > best_member.crowding_distance):
                    best_member = member

        return best_member

    @property
    def first_pareto_front(self):
        return [p for p in self.population if p.rank == 1]

    @property
    def summary(self):
        istances = []

        for index, p in enumerate(self.population):
            row = {}
            row['index'] = index + 1
            row['program'] = p.program
            row['complexity'] = p.complexity
            row['rank'] = p.rank

            for f_k, f_v in p.fitness.items():
                row[f_k] = f_v

            istances.append(row)

        return pd.DataFrame(istances)

    @property
    def best_history(self):
        istances = []

        for index, p in enumerate(self.best_programs_history):
            row = {}
            row['generation'] = index + 1
            row['program'] = p.program
            row['complexity'] = p.complexity
            row['rank'] = p.rank

            for f_k, f_v in p.fitness.items():
                row[f_k] = f_v

            istances.append(row)

        return pd.DataFrame(istances)
