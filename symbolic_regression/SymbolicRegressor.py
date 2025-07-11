import copy
import datetime
import json
import logging
import multiprocessing
import os
import pickle
import random
import re
import time
import traceback
import zlib
from itertools import repeat
from multiprocessing import Process, Queue
from typing import Any, Dict, List, Tuple, Union
from joblib import Parallel, delayed
from symbolic_regression.preprocessing import check_assumptions

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from matplotlib import pyplot as plt
from scipy.stats import spearmanr


from symbolic_regression.callbacks.CallbackBase import MOSRCallbackBase
from symbolic_regression.multiobjective.fitness.Base import BaseFitness
from symbolic_regression.multiobjective.hypervolume import _HyperVolume
from symbolic_regression.Population import Population
from symbolic_regression.Program import Program


class SymbolicRegressor:

    def __init__(self, client_name: str, const_range: tuple = (0, 1), 
                 parsimony=0.8, parsimony_decay=0.85, population_size: int = 300,
                 tournament_size: int = 3, genetic_operators_frequency: dict = {'crossover': 1, 'mutation': 1},
                 callbacks: List[MOSRCallbackBase] = list(), genetic_algorithm: str = 'NSGA-II',
                 offspring_timeout_condition: tuple = None,
                 duplicate_drop_by_similarity: bool = True, duplicates_delta_fitness: float = 1e-2,
                 ) -> None:
        """ This class implements the basic features for training a Symbolic Regression algorithm

        Args:
            - client_name: str
                the name of the client

            - const_range: tuple (default: (0, 1))
                this is the range of values from which to generate constants in the program

            - parsimony: float (default: 0.8)
                the ratio to which a new operation is chosen instead of a terminal node in program generations

            - parsimony_decay: float (default: 0.85)
                a modulation parameter to decrease the parsimony and limit program generation depth

            - population_size: int (default: 300)
                the size of the population

            - tournament_size: int (default: 3)
                this modulate the tournament selection and set the dimension of the selection

            - genetic_operators_frequency: dict (default: {'crossover': 1, 'mutation': 1})
                this is a dictionary that set the relative frequency of each genetic operator

            - statistics_computation_frequency: int (default: 10)
                this is the frequency of computing the statistics of the population
                - if set to -1, the statistics are computed only at the end of the training
                - if set to 1, the statistics are computed at each generation
                - if set to 10, the statistics are computed every 10 generations
                Statistics are:
                    - Hypervolume
                    - Tree Diversity
                    - Spearman Diversity

            - callbacks: List[MOSRCallbackBase] (default: list())
                this is the list of callbacks to be executed during the training

            - genetic_algorithm: str (default: 'NSGA-II')
                the genetic algorithm to be used in the training. Can be 'NSGA-II' or 'SMS-EMOEA'
                If none of the two is provided, the default is 'NSGA-II'

            - offspring_timeout_condition: tuple (default: None)
                this is a tuple that set the condition to stop the training
                The tuple is in the form (int, float) where the first element is the number of seconds
                after which the condition is checked and the second element is the number of offsprings
                per second below which the condition is True. If the condition is True the offspring
                generation is stopped and the training goes to the next generation

            - duplicate_drop_by_similarity: bool (default: False)
                this is a boolean that set if the duplicates are dropped by similarity

            - duplicates_delta_fitness: float (default: 1e-2)
                this is the delta fitness to consider two programs as duplicates

        Returns:
            - None
        """

        # Regressor Configuration
        self.client_name: str = client_name
        self.features: List = None
        self.feature_assumption: dict = None
        self.feature_probability = None
        self.bool_update_feature_probability = None
        self.offspring_timeout_condition: tuple = offspring_timeout_condition
        self.operations: List = None
        self.population_size: int = population_size


        # Population Configuration
        self.population: Population = Population()
        self._average_complexity: float = None
        self.const_range: tuple = const_range
        self.parsimony: float = parsimony
        self.parsimony_decay: float = parsimony_decay
        self.tournament_size: int = tournament_size
        self.duplicates_delta_fitness: float = duplicates_delta_fitness
        self.duplicates_drop_by_similarity: bool = duplicate_drop_by_similarity

        ## degbugging phenotipic operations
        self.nr_success_brgflow: int = 0
        self.nr_success_bootstrap: int = 0

        # Training Configuration
        self.data_shape: tuple = None
        self.converged_generation: int = None
        self.fitness_functions: List[BaseFitness] = None
        self.generations_to_train: int = None
        self.generation: int = 0
        self._genetic_algorithm = None
        self.genetic_algorithm: str = genetic_algorithm if genetic_algorithm in [
            'NSGA-II', 'SMS-EMOEA'] else 'NSGA-II'
        self.genetic_operators_frequency: dict = genetic_operators_frequency
        self.genetic_operators_times: dict = {key:[] for key in  genetic_operators_frequency.keys() }
        self._stop_at_convergence: bool = False
        self._convergence_rolling_window: int = None
        self._convergence_rolling_window_threshold: float = 0.01
        self.training_timeout_mins: int = -1
        self._nadir_points: Dict = dict()
        self._ideal_points: Dict = dict()
        self._fpf_fitness_history: Dict = dict()

        # Statistics
        self.best_history: Dict = dict()
        self.first_pareto_front_history: Dict = dict()
        self.population_history: Dict = dict()
        self.fpf_hypervolume: float = None
        self.fpf_hypervolume_reference: float = None
        self.fpf_tree_diversity: float = None
        self.fpf_spearman_diversity: float = None
        self.fpf_stats: pd.DataFrame = pd.DataFrame()
        self.training_duration: int = 0

        self._callbacks: List[MOSRCallbackBase] = list()
        self.callbacks: List[MOSRCallbackBase] = callbacks
        self._callback_terminate: bool = False

        self.times: pd.DataFrame = pd.DataFrame(
            columns=['count_average_complexity',
                     'count_duplicated_elements',
                     'count_invalid_elements',
                     'ratio_duplicated_elements',
                     'ratio_invalid_elements',
                     'time_crowding_distance_computation',
                     'time_duplicated_drop',
                     'time_generation_total',
                     'time_hypervolume_computation',
                     'time_initialization',
                     'time_invalids_drop',
                     'time_offsprings_generation',
                     'time_offsprings_generation_capped',
                     'time_pareto_front_computation',
                     'time_refill_invalid',
                     'time_spearman_diversity_computation',
                     'time_tree_diversity_computation',]
        )

    def save_model(self, file: str, checkpoint_overwrite: bool = None):
        # Gets the number of the generation padded with zeros to the number
        # of digits of the total number of generations
        generation_str = str(self.generation).zfill(
            len(str(self.generations_to_train)))

        if not checkpoint_overwrite:
            file = file + f".gen{generation_str}.sr"
        else:
            file = file + ".sr"

        # Dump this object in a pickle file
        for p in self.population:
            p.programs_dominated_by: List[Program] = list()
            p.programs_dominates: List[Program] = list()

        directory = os.path.dirname(file)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        with open(file, "wb") as f:
            self_copy = copy.deepcopy(self)
            self_copy.callbacks = list()
            pickle.dump(compress(self_copy), f)

        # Dump the self.metadata in a json file
        with open(file + ".metadata.json", "w") as f:
            json.dump(self.metadata, f)

        with open(file + ".summary.csv", "w") as f:
            self.summary.to_csv(f)

    def load_model(self, file: str):
        with open(file, "rb") as f:
            sr: SymbolicRegressor = pickle.load(f)

        try:
            sr = decompress(sr)
            logging.debug(f"Loaded compressed model from {file}")
        except TypeError:
            pass

        return sr

    @property
    def average_complexity(self):
        return self._average_complexity

    @average_complexity.getter
    def average_complexity(self):
        return np.mean([p.complexity for p in self.population]) if len(self.population) > 0 else 0

    @property
    def callbacks(self):
        if hasattr(self, '_callbacks'):
            return self._callbacks
        else:
            return list()

    @callbacks.setter
    def callbacks(self, callbacks: List[MOSRCallbackBase]):
        if not isinstance(callbacks, List):
            raise TypeError(
                f"Expected a list of MOSRCallbackBase, got {type(callbacks)}")
        self._callbacks = callbacks
        for c in self._callbacks:
            c.sr: 'SymbolicRegressor' = self
            try:
                c.on_callback_set_init()
            except:
                logging.warning(
                    f"Callback {c.__class__.__name__} raised an exception on callback set init")

    @property
    def stop_at_convergence(self):
        if not hasattr(self, '_stop_at_convergence'):
            self._stop_at_convergence = False
        return self._stop_at_convergence

    @stop_at_convergence.setter
    def stop_at_convergence(self, stop_at_convergence: bool):
        self._stop_at_convergence = stop_at_convergence

    @property
    def convergence_rolling_window(self):
        if not hasattr(self, '_convergence_rolling_window'):
            self._convergence_rolling_window = None
        return self._convergence_rolling_window

    @convergence_rolling_window.setter
    def convergence_rolling_window(self, convergence_rolling_window: int):
        self.converged_generation = None
        self._convergence_rolling_window = convergence_rolling_window

    @property
    def convergence_rolling_window_threshold(self):
        if not hasattr(self, '_convergence_rolling_window_threshold'):
            self._convergence_rolling_window_threshold = 0.01
        return self._convergence_rolling_window_threshold

    @convergence_rolling_window_threshold.setter
    def convergence_rolling_window_threshold(self, convergence_rolling_window_threshold: float):
        self._convergence_rolling_window_threshold = convergence_rolling_window_threshold

    @property
    def nadir_points(self):
        if not hasattr(self, '_nadir_points'):
            self._nadir_points = dict()
        return self._nadir_points

    @nadir_points.setter
    def nadir_points(self, nadir_points: Dict):
        self._nadir_points = nadir_points

    @property
    def ideal_points(self):
        if not hasattr(self, '_ideal_points'):
            self._ideal_points = dict()
        return self._ideal_points

    @ideal_points.setter
    def ideal_points(self, ideal_points: Dict):
        self._ideal_points = ideal_points

    @property
    def fpf_fitness_history(self):
        if not hasattr(self, '_fpf_fitness_history'):
            self._fpf_fitness_history = dict()
        return self._fpf_fitness_history

    @fpf_fitness_history.setter
    def fpf_fitness_history(self, fpf_fitness_history: Dict):
        self._fpf_fitness_history = fpf_fitness_history

    @property
    def genetic_algorithm(self):
        return self._genetic_algorithm

    @genetic_algorithm.setter
    def genetic_algorithm(self, genetic_algorithm: str):
        if genetic_algorithm not in ['NSGA-II', 'SMS-EMOEA']:
            raise ValueError(
                f"Expected 'NSGA-II' or 'SMS-EMOEA', got {genetic_algorithm}")

        logging.debug(f"Genetic Algorithm set to {genetic_algorithm}")
        self._genetic_algorithm = genetic_algorithm

    def compute_hypervolume(self):
        """
        This method computes the hypervolume of the current population

        The hypervolume is computed using the hypervolume reference point
        defined in the fitness functions.

        If the reference point is not defined the hypervolume is computed 
        using the maximum value of each fitness function as reference point.

        If the reference point is defined but the fitness function is not
        a minimization problem, the hypervolume is computed using the maximum
        value of each fitness function as reference point.

        If the reference point is defined but is lower than the maximum value
        of the fitness function, the hypervolume is computed using the maximum
        value of each fitness function as reference point and the reference
        point is updated to the new value.

        Args:
            - None

        Returns:
            - hypervolume: float
                The hypervolume of the current population
        """

        fitness_to_hypervolume = [
            fitness for fitness in self.fitness_functions if fitness.hypervolume_reference and fitness.minimize]

        points = np.array([np.array([p.fitness[ftn.label] for ftn in fitness_to_hypervolume])
                           for p in self.first_pareto_front])

        references = np.array(
            [ftn.hypervolume_reference for ftn in fitness_to_hypervolume])

        try:
            points = points[np.sum((points - references)
                                   <= 0, axis=1) == points.shape[1]]
            self.fpf_hypervolume = _HyperVolume(
                references).compute(points.copy().tolist())

        except ValueError:
            self.fpf_hypervolume = np.nan

        self.fpf_stats.loc[self.generation, 'n_individuals'] = len(
            self.first_pareto_front)
        self.fpf_stats.loc[self.generation,
                           'fpf_hypervolume'] = self.fpf_hypervolume
        self.fpf_stats.loc[self.generation,
                           'fpf_hypervolume_reference'] = self.fpf_hypervolume_reference

        self.fpf_hypervolume_reference = np.product(references)

        return self.fpf_hypervolume

    def compute_fitness_population(self, fitness_functions: List[BaseFitness] = None, data: Union[dict, pd.DataFrame, pd.Series] = None, validation: bool = False, validation_federated: bool = False, simplify: bool = True):
        """
        This method computes the performance of each program in the population

        Args:
            - fitness_functions: List[BaseFitness] (default: None)
                The fitness functions to use to compute the performance. Can override by the fitness_functions
                attribute of the class
            - data: Union[dict, pd.DataFrame, pd.Series]
                The data on which the performance is computed
            - validation: bool (default: False)
                If True the performance is computed on the validation data without optimization
            - validation_federated: bool (default: False)
                If True the performance is computed on the training and validation data without optimization (only for federated training)
            - simplify: bool (default: False)
                If True the program is simplified before computing the performance
        """
        if data is None:
            logging.warning(
                f'No {"training" if not validation else "validation"} data provided to compute the fitness of the population')
            return

        for p in self.population:
            p: Program
            p.compute_fitness(
                fitness_functions=self.fitness_functions if fitness_functions is None else fitness_functions,
                data=data,
                validation=validation,
                validation_federated=validation_federated,
                simplify=simplify)

    def _create_pareto_front(self):
        """
        This method creates the pareto front of the population
        The pareto front is a group of programs that are non-dominated by any other program in the population
        The first pareto front is the one with the lowest rank and therefore the best programs
        The following pareto fronts are the ones with the next lowest rank and so on

        We use the first pareto fron to identify the most optimal programs the crowding distance to identify the most
        diverse programs.
        """
        pareto_front = list()

        for index_p, p in enumerate(self.population):
            p.programs_dominates = list()
            p.programs_dominated_by = list()
        
        for index_p1, p1 in enumerate(self.population):
            if self.verbose > 1:
                print(f"\tPareto Fronts: computing dominance {(index_p1+1)/len(self.population):.1%}", end='\r')
            
            p1.rank = np.inf

            if not p1.is_valid:
                continue
            for p2 in self.population[index_p1 + 1:]:
                if p1 == p2 or not p2.is_valid:
                    continue

                if self.dominance(p1, p2):
                    p1.programs_dominates.append(p2)
                    p2.programs_dominated_by.append(p1)
                elif self.dominance(p2, p1):
                    p1.programs_dominated_by.append(p2)
                    p2.programs_dominates.append(p1)

            if len(p1.programs_dominated_by) == 0:
                p1.rank = 1
                pareto_front.append(p1)

        if self.verbose > 1:
            print(f"\tPareto Fronts: computing dominance {(index_p1+1)/len(self.population):.1%}. Completed!", end='\n')

        i = 1

        # Set the belonging pareto front to every element of the population

        while pareto_front:

            if self.verbose > 1:
                print(f"\tPareto Fronts: computing rank {i}", end='\r')
            next_pareto_front = list()

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
        
        if self.verbose > 1:
            print(f"\tPareto Fronts: computing rank {i}. Completed!", end='\n')

    def _crowding_distance(self, population: Population, rank_iter: int = 1):
        """
        This method calculates the crowding distance for each program in the population
        The crowding distance is used to identify the most diverse programs in the population
        It is calculated as the sum of the normalized distances between the programs in the pareto front
        The distance is calculated for each objective function.

        The higher the crowding distance the more diverse the program is with respect to the other programs in the same pareto front
        """

        pareto_front: List[Program] = self.extract_pareto_front(
            population=population, rank=rank_iter)

        while pareto_front:  # Exits when extract_pareto_front return an empty list

            for p in pareto_front:
                p.crowding_distance = 0

            for ftn in self.fitness_functions:

                fitness_label = ftn.label
                # This exclude the fitness functions which are set not to be minimized
                if not ftn.minimize:
                    continue

                # Highest fitness first for each objective
                pareto_front.sort(
                    key=lambda p: p.fitness[fitness_label], reverse=False)

                norm = pareto_front[0].fitness[fitness_label] - \
                    pareto_front[-1].fitness[fitness_label] + 1e-20

                for index, program in enumerate(pareto_front):
                    if index == 0 or index == len(pareto_front) - 1:
                        delta = float('inf')
                    else:
                        delta = pareto_front[index - 1].fitness[fitness_label] - \
                            pareto_front[index + 1].fitness[fitness_label]

                    program.crowding_distance += delta / norm

            rank_iter += 1
            pareto_front = self.extract_pareto_front(
                population=population, rank=rank_iter)

    def _exclusive_hypervolume(self, population: Population, rank: int = 1):
        """
        Calculate the exclusive hypervolume for the given population and rank iteration.

        Args:
            - population (Population):
                The population for which to calculate the exclusive hypervolume.
            - rank (int):
                The rank iteration to consider.

        Returns:
        None

        This method calculates the exclusive hypervolume for a given population and rank iteration.
        It uses the fitness functions with hypervolume reference and minimize set to True.
        The exclusive hypervolume is calculated using the points from the Pareto front of the population
        and the hypervolume references. The calculated hypervolume values are then assigned to the
        corresponding programs in the Pareto front.

        If a ValueError occurs during the calculation, the hypervolume for the corresponding program
        is set to infinity and a warning message is logged.

        The rank iteration is incremented after each calculation, and the process continues until
        there are no more programs in the Pareto front.

        Note: This method assumes that the `extract_pareto_front` method is implemented correctly.

        """
        fitness_to_hypervolume = [
            fitness for fitness in self.fitness_functions if fitness.hypervolume_reference and fitness.minimize]
        references = np.array(
            [ftn.hypervolume_reference for ftn in fitness_to_hypervolume])

        pareto_front: List[Program] = self.extract_pareto_front(
            population=population, rank=rank)
        while pareto_front:
            points = np.array([np.array([p.fitness[ftn.label] for ftn in fitness_to_hypervolume])
                               for p in pareto_front])

            try:
                points = points[np.sum((references - points)
                                       >= 0, axis=1) == points.shape[1]]
                hypervolume = _HyperVolume(
                    references).exclusive(points.copy().tolist())
            except ValueError:
                hypervolume = [np.inf] * len(pareto_front)
            for program, hypervolume in zip(pareto_front, hypervolume):
                program.exclusive_hypervolume = hypervolume

            rank += 1
            pareto_front = self.extract_pareto_front(
                population=population, rank=rank)

    @staticmethod
    def dominance(program1: Program, program2: Program) -> bool:
        """
        This method checks if program1 dominates program2
        A program p1 dominates a program p2 if all the fitness of p1 are 
        less or equal than p2 and at least one is less than p2

        We use this method to create the pareto fronts
        We allow fitness functions to be set not to be minimized, in this 
        case the fitness is calculated but not used to create the pareto fronts

        Args:
            = program1: Program
                The program that is being checked if it dominates program2
            = program2: Program
                The program that is being checked if it is dominated by program1

        Returns:
            = True if program1 dominates program2
            = False otherwise
        """

        # How many element in the p1.fitness are less than p2.fitness
        at_least_one_less_than_zero = False
        all_less_or_eq_than_zero = True

        if program1.program and program2.program:
            for this_fitness in program1.fitness.keys():
                # Ignore the fitness which are not to be optimized
                if program1.is_fitness_to_minimize[this_fitness] == False:
                    continue

                try:
                    p1_fitness = program1.fitness[this_fitness]
                    p2_fitness = program2.fitness[this_fitness]
                except KeyError:
                    break

                d = p1_fitness - p2_fitness

                if d < 0:
                    at_least_one_less_than_zero = True
                if d > 0:
                    all_less_or_eq_than_zero = False

            return at_least_one_less_than_zero and all_less_or_eq_than_zero

        return False

    def drop_duplicates(self, inplace: bool = False) -> list:
        """ 
        This method removes the duplicates from the population
        A program is considered a duplicate if it has the same fitness rounded to 
        the number of decimals specified in the implementation of the program

        Args:
            - inplace: bool (default False)
                If True the population is updated, if False a new list is returned
        """

        complexities_index = np.argsort(
            [p.complexity for p in self.population])

        for index, complexity_index in enumerate(complexities_index):
            if self.verbose > 1:
                print(
                    f"\tDuplicates: finding duplicates {(index+1)/len(complexities_index):.1%}", end='\r')
            p: 'Program' = self.population[complexity_index]

            if p.is_valid and not p._is_duplicated:

                other_indices = [complexities_index[l] for l in range(
                    index + 1, len(complexities_index))]

                for j in other_indices:
                    p_confront = self.population[j]
                    if p_confront.is_valid and not p_confront._is_duplicated:
                        p_confront._is_duplicated = p.is_duplicate(
                            other=p_confront,
                            delta_fitness=self.duplicates_delta_fitness,
                            drop_by_similarity=self.duplicates_drop_by_similarity,
                        )

        if self.verbose > 1:
            print(
                f"\tDuplicates: finding duplicates {(index+1)/len(complexities_index):.1%}. Completed!", end='\n')

        if inplace:
            self.population: Population = Population(
                filter(lambda p: p._is_duplicated == False, self.population))
            return self.population

        return Population(
            filter(lambda p: p._is_duplicated == False, self.population))

    def drop_invalids(self, inplace: bool = False) -> list:
        """
        This method removes the invalid programs from the population
        A program is considered invalid if it has an InvalidNode in its tree or 
        if at least one of the operations is mathematically impossible, like 
        division by zero.

        Args:
            - inplace: bool (default False)
                If True the population is updated, if False a new list is returned
        """
        if self.verbose > 1:
            print(f"\tInvalids: removing invalids. Completed!", end='\n')
        if inplace:
            self.population: Population = Population(
                filter(lambda p: p.is_valid == True, self.population))
            return self.population

        return list(filter(lambda p: p.is_valid == True, self.population))

    def extract_pareto_front(self, population: Population, rank: int):
        """
        This method extracts the programs in the population that are in the pareto front
        of the specified rank

        Args:
            - rank: int
                The rank of the pareto front to be extracted

        Returns:
            - pareto_front: List
                The list of programs in the pareto front of the specified rank
        """
        pareto_front = [p for p in population if p.rank == rank and p.is_valid]

        return pareto_front

    @property
    def first_pareto_front(self):
        """
        First Pareto Front

        This method returns the first Pareto Front of the population.

        Args:
            - None

        Returns:
            - first_pareto_front: list
                The first Pareto Front of the population
        """
        return self.extract_pareto_front(population=self.population, rank=1)

    def fit(self, 
            data: Union[dict, pd.DataFrame, pd.Series], 
            features: List[str], 
            feature_probability: List[float], 
            operations: List[dict], 
            fitness_functions: List[BaseFitness], 
            generations_to_train: int, 
            n_jobs: int = -1, 
            bool_update_feature_probability: bool = False, 
            stop_at_convergence: bool = False, 
            convergence_rolling_window: int = None, 
            convergence_rolling_window_threshold: float = 0.01, 
            verbose: int = 0, 
            val_data: Union[dict, pd.DataFrame, pd.Series] = None, 
            adjust_feature_probability: bool = False,
            training_timeout_mins: int = -1) -> None:
        """
        This method trains the population.

        This method implements store in this object the arguments passed at its execution;
        because these are required in other methods of the class.

        We implement here the call to the method that performs the training allowing to
        catch the exception KeyboardInterrupt that is raised when the user stops the training.
        This allow the user to stop the training at any time and not lose the progress made.
        The actual training is then implemented in the private method _fit.

        Args:
            - data: Union[dict, pd.DataFrame, pd.Series]
                The data on which the training is performed
            - features: List
                The list of features to be used in the training
            - operations: List
                The list of operations to be used in the training
            - fitness_functions: List[BaseFitness]
                The list of fitness functions to be used in the training
            - generations_to_train: int
                The number of generations to train
            - n_jobs: int (default -1)
                The number of jobs to be used in the training
            - stop_at_convergence: bool (default False)
                If True the training stops when the population converges
            - convergence_rolling_window: int (default None)
                The number of generations to consider for the convergence
            - convergence_rolling_window_threshold: float (default 0.01)
                The threshold to consider the population converged
            - verbose: int (default 0)
                The verbosity level of the training
            - val_data: Union[dict, pd.DataFrame, pd.Series] (default None)
                The data on which the validation is performed

        Returns:
            - None
        """
        self.features = features
        self.feature_probability = feature_probability
        self.bool_update_feature_probability = bool_update_feature_probability
        self.adjust_feature_probability = adjust_feature_probability
        self.operations = operations
        self.feature_assumption = check_assumptions(data[self.features])
        
    
        self.data_shape = data.shape

        self.fitness_functions = fitness_functions
        self.generations_to_train = generations_to_train
        self.n_jobs = n_jobs
        self.stop_at_convergence = stop_at_convergence
        self.convergence_rolling_window = convergence_rolling_window
        self.convergence_rolling_window_threshold = convergence_rolling_window_threshold
        self.training_timeout_mins = training_timeout_mins
        self.verbose = verbose

        start = time.perf_counter()
        try:
            self._fit(data=data, val_data=val_data)
        except KeyboardInterrupt:

            self.generation -= 1  # The increment is applied even if the generation is interrupted
            self.status = "Interrupted by KeyboardInterrupt"
            logging.warning(f"Training terminated by a KeyboardInterrupt")

            if not isinstance(self.population, Population):
                self.population = Population(self.population)

        stop = time.perf_counter()
        self.training_duration += stop - start


    def _fit(self, data: Union[dict, pd.DataFrame, pd.Series], val_data: Union[dict, pd.DataFrame, pd.Series] = None) -> None:
        """
        This method is the main loop of the genetic programming algorithm.

        Firstly we initialize the population of N individuals if it is not already initialized.
        Then we loop over the generations and apply the genetic operators
        to the population.

        We then generate other N offspring using the genetic operators and
        we add them to the population. This generates a population of 2N individuals.

        To select the population of N individuals to pass to the next generation
        we use the NSGA-II algorithm (Non-dominant Sorting Genetic Algorithm) which 
        sort the 2N individuals in non-dominated fronts and then select the N individuals
        that have the best performance.

        We need to remove the individuals that are not valid because and also the individuals
        that are duplicated. If removing the invalid and duplicated individuals leaves us
        with less than N individuals, we generate new individuals to fill the population.

        Args:
            - data: Union[dict, pd.DataFrame, pd.Series]
                The data on which the training is performed
            - val_data: Union[dict, pd.DataFrame, pd.Series] (default None)
                The data on which the validation is performed

        Returns:
            - None

        """
        from functools import wraps
        import signal
        def timeout_decorator_generation(seconds):
            def decorator(func):
                @wraps(func)
                def wrapper(*args, **kwargs):
                    def handler(signum, frame):
                        raise TimeoutError(f"Function execution timed out after {seconds} seconds")
                    
                    # Set the timeout handler (only works on Unix-based systems)
                    original_handler = signal.signal(signal.SIGALRM, handler)
                    signal.alarm(seconds)
                    
                    try:
                        result = func(*args, **kwargs)
                        return result
                    except Exception as e:
                        # You can log the exception here if needed
                        return None
                    finally:
                        # Cancel the timeout and restore the original handler
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, original_handler)
                return wrapper
            return decorator
        
        def timeout_decorator_offspring(seconds):
            def decorator(func):
                @wraps(func)
                def wrapper(*args, **kwargs):
                    def handler(signum, frame):
                        raise TimeoutError(f"Function execution timed out after {seconds} seconds")
                    
                    # Set the timeout handler (only works on Unix-based systems)
                    original_handler = signal.signal(signal.SIGALRM, handler)
                    signal.alarm(seconds)
                    
                    try:
                        result = func(*args, **kwargs)
                        return result
                    except Exception as e:
                        # Return a tuple with the same structure but "empty" values
                        return (None, 0, 0, None, 0)
                    finally:
                        # Cancel the timeout and restore the original handler
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, original_handler)
                return wrapper
            return decorator


        total_generation_time = 0
        jobs = self.n_jobs if self.n_jobs > 0 else os.cpu_count()
        fit_start = time.perf_counter()

        if not self.population:

            for c in self.callbacks:
                try:
                    c.on_initialization_start()
                except:
                    logging.warning(
                        f"Callback {c.__class__.__name__} raised an exception on initialization start")
                    logging.warning(
                        traceback.format_exc())

            if self.verbose > 0:
                print(f"Initializing population")
            self.status = "Generating population"

            # Apply the decorator to the generate_individual method
            timeout_generate_individual = timeout_decorator_generation(50)(self.generate_individual)

            results = Parallel(n_jobs=jobs)(
                delayed(timeout_generate_individual)(
                    data=data,
                    features=self.features,
                    feature_assumption=self.feature_assumption,
                    feature_probability=self.feature_probability,
                    bool_update_feature_probability=self.bool_update_feature_probability,
                    operations=self.operations,
                    fitness_functions=self.fitness_functions,
                    const_range=self.const_range, 
                    parsimony=self.parsimony,
                    parsimony_decay=self.parsimony_decay,
                    val_data=val_data
                )
                for _ in range(self.population_size)
            )

            # Filter valid results (those that didn't time out)
            results = [r for r in results if r is not None]
            
            self.population = Population(results)

            self.times.loc[self.generation+1,
                           "time_initialization"] = time.perf_counter() - fit_start

            for c in self.callbacks:
                try:
                    c.on_initialization_end()
                except:
                    logging.warning(
                        f"Callback {c.__class__.__name__} raised an exception on initialization end")
                    logging.warning(
                        traceback.format_exc())

        else:
            # self.population = [
            #     program for program in self.population if program.is_valid]

            if self.verbose > 0:
                print(f"Fitting with existing population")
            logging.info(
                f"Fitting with existing population of {len(self.population)} valid elements")


            if len(self.population) < self.population_size:
                logging.info(
                    f"Population of {len(self.population)} elements is less than population_size:{self.population_size}. Integrating with new elements")
                new_individuals = self.population_size - len(self.population)
                # Apply the decorator to the generate_individual method
                timeout_generate_individual = timeout_decorator_generation(50)(self.generate_individual)

                refill_training_start = Parallel(n_jobs=jobs)(
                        delayed(timeout_generate_individual)(
                            data=data,
                            features=self.features,
                            feature_assumption=self.feature_assumption,
                            feature_probability=self.feature_probability,
                            bool_update_feature_probability = self.bool_update_feature_probability,
                            operations=self.operations,
                            fitness_functions=self.fitness_functions,
                            const_range=self.const_range,
                            parsimony=self.parsimony,
                            parsimony_decay=self.parsimony_decay,
                            val_data=val_data
                        )
                        for _ in range(new_individuals)
                    )
            
                refill_training_start = Population(refill_training_start)

                self.population.extend(refill_training_start)
                self.population = Population(self.population)

                self.times.loc[self.generation+1,
                           "time_initialization"] = time.perf_counter() - fit_start

        while True:
            self.status = "Training"
            start_time_generation = time.perf_counter()

            if self.generation > 0 and self.generations_to_train <= self.generation:
                logging.info(
                    f"The model already had trained for {self.generation} generations")
                self.status = "Terminated: generations completed"
                return

            for c in self.callbacks:
                try:
                    c.on_generation_start()
                except:
                    logging.warning(
                        f"Callback {c.__class__.__name__} raised an exception on generation start")
                    logging.warning(
                        traceback.format_exc())

            for p in self.population:
                p.programs_dominates: List[Program] = list()  # type: ignore
                p.programs_dominated_by: List[Program] = list()  # type: ignore

            if self.generation > 0:
                time_total = datetime.timedelta(seconds=int(
                    self.times['time_generation_total'].sum()))
                seconds_iter_avg = datetime.timedelta(seconds=int(
                    self.times['time_generation_total'].tail(5).median()))
                seconds_iter_std = datetime.timedelta(seconds=int(self.times['time_generation_total'].tail(
                    5).std())) if self.generation > 1 else datetime.timedelta(seconds=0)
                time_per_generation = f"{seconds_iter_avg} ± {seconds_iter_std}"
                expected_time = datetime.timedelta(seconds=int(self.times['time_generation_total'].tail(
                    10).median() * (self.generations_to_train - self.generation)))
            else:
                time_total = f"00:00:00"
                time_per_generation = f"00:00:00 ± 00:00:00"
                expected_time = 'Unknown'

            generation_time = datetime.timedelta(
                seconds=int(total_generation_time))
            timing_str = f"Generation {generation_time} - On average: {time_per_generation} - Total: {time_total} - To completion: {expected_time}"

            self.generation += 1

            if self.verbose > 1:
                print("#" * len(timing_str))
                print(timing_str)
                print("#" * len(timing_str))
            if self.verbose > 0:
                print(
                    f"{self.client_name}: starting generation {self.generation}/{self.generations_to_train} ({self.genetic_algorithm})", end='\r' if self.verbose == 1 else '\n', flush=True)

            start_offspring = time.perf_counter()

            #################################################################################
            #################################################################################
            # Offsprings generation start
            #################################################################################
            #################################################################################

            for c in self.callbacks:
                try:
                    c.on_offspring_generation_start()
                except:
                    logging.warning(
                        f"Callback {c.__class__.__name__} raised an exception on offspring generation start")
                    logging.warning(
                        traceback.format_exc())

            self.status = "Generating Offsprings"

            """
            Generates `population_size` offspring in parallel batches using joblib.
            Preserves:
            1. Running count of offspring generated
            2. Early stopping if generation is too slow or takes too long
            """
            too_long=False

            jobs = self.n_jobs if self.n_jobs > 0 else os.cpu_count()

            offsprings: List[Program] = []
            start_time = time.perf_counter()

            # We'll generate offspring in batches until we reach population_size
            # or until the time-based condition says "stop."
            while len(offsprings) < self.population_size:

                elapsed_s = time.perf_counter() - start_time

                # ------------------------------------------------------------
                # Check your "timeout condition"
                # offspring_timeout_condition might be a tuple like (max_time, min_rate)
                # If (elapsed_s > max_time) AND (offspring_per_second < min_rate),
                # we trigger an early stop.
                # ------------------------------------------------------------
                if (isinstance(self.offspring_timeout_condition, tuple)
                    and elapsed_s > self.offspring_timeout_condition[0]
                    and (len(offsprings) / max(elapsed_s, 1e-9) < self.offspring_timeout_condition[1])
                ):
                    logging.warning(
                        f"Offspring generation got too slow; only {len(offsprings)}/{self.population_size} "
                        f"generated in {int(elapsed_s)}s. Aborting early."
                    )
                    too_long = True
                    break

                # ------------------------------------------------------------
                # Decide how many to generate in this batch
                # e.g.,  you can spawn `jobs` parallel tasks, each of which returns
                # a list of new offspring from _get_offspring_batch.
                # ------------------------------------------------------------
                needed = self.population_size - len(offsprings)
                batch_tasks = needed # min(5*jobs, needed)

                # Make a local copy of the population (if needed),
                # or call .as_binary() if your code requires it
                try:
                    population_to_pass = self.population.as_binary()
                except AttributeError:
                    population_to_pass = self.population

                # ------------------------------------------------------------
                # Run this batch in parallel via joblib
                # Each delayed(...) call spawns a chunk of new offspring
                # ------------------------------------------------------------
                # Apply the decorator for timeout
                timeout_generate_offspring = timeout_decorator_offspring(50)(self._generate_one_offspring)    

                results = Parallel(n_jobs=jobs)(
                    delayed(timeout_generate_offspring)(
                        data=data,
                        genetic_operators_frequency=self.genetic_operators_frequency,
                        fitness_functions=self.fitness_functions,
                        population=population_to_pass,
                        tournament_size=self.tournament_size,
                        generation = self.generation,
                        val_data = val_data
                    )
                    for _ in range(batch_tasks)
                )

                # Filter out timeout results before unpacking
                results = [r for r in results if r[0] is not None]

                # `results` is now a list of lists 
                # unpack them into `offsprings`, nr_brgflow, nr_bootstrap
                # zip(*list_of_lists) effectively "rotates" the rows into columns
                try:
                    children, nr_brgflow, nr_bootstrap, gen_ops, gen_ops_time = zip(*results)
                except ValueError: 
                    print(results)

                # zip returns tuples, so to make children a list:
                offsprings+=list(children)
                self.nr_success_brgflow += sum(nr_brgflow)
                self.nr_success_bootstrap += sum(nr_bootstrap)
                for geop, geop_time in zip(gen_ops, gen_ops_time):
                    self.genetic_operators_times[geop].append(geop_time)
                

                # ------------------------------------------------------------
                # Print progress if verbose
                # ------------------------------------------------------------
                if self.verbose > 1:
                    curr_count = len(offsprings)
                    elapsed_s = max(time.perf_counter() - start_time, 1e-9)
                    ops_s = round(curr_count / elapsed_s, 2)
                    elapsed_td = datetime.timedelta(seconds=int(elapsed_s))
                    print(
                        f"Offsprings generated: {curr_count}/{self.population_size} "
                        f"({elapsed_td}, {ops_s} /s) ...",
                        end="\r",
                        flush=True
                    )

                # Sleep or not – depends on whether you want to slow the loop
                # time.sleep(0.2)  # sometimes helpful to reduce console spam

            # Finished or aborted early
            if self.verbose > 1:
                total = len(offsprings)
                elapsed_s = max(time.perf_counter() - start_time, 1e-9)
                ops_s = round(total / elapsed_s, 2)
                elapsed_td = datetime.timedelta(seconds=int(elapsed_s))
                print(
                    f"\nOffsprings generated: {total}/{self.population_size} "
                    f"({elapsed_td}, {ops_s} /s). Completed!"
                )

            self.times.loc[self.generation,
                           "time_offsprings_generation"] = time.perf_counter() - start_offspring
            self.times.loc[self.generation,
                           "time_offsprings_generation_capped"] = too_long

            self.population = Population(self.population + offsprings)

            for c in self.callbacks:
                try:
                    c.on_offspring_generation_end()
                except:
                    logging.warning(
                        f"Callback {c.__class__.__name__} raised an exception on offspring generation end")
                    logging.warning(
                        traceback.format_exc())

            #################################################################################
            #################################################################################
            # Offsprings generation end
            #################################################################################
            #################################################################################
            # Check duplicates and invalids
            #################################################################################
            #################################################################################

            # Removes all duplicated programs in the population
            start_drop_duplicates = time.perf_counter()

            before_cleaning = len(self.population)
            self.drop_duplicates(inplace=True)
            
            after_drop_duplicates = len(self.population)
            logging.debug(
                f"{before_cleaning-after_drop_duplicates}/{before_cleaning} duplicates programs removed")
            
            self.times.loc[self.generation, "ratio_duplicated_elements"] = (
                before_cleaning-after_drop_duplicates)/before_cleaning
            
            self.times.loc[self.generation,
                           "count_duplicated_elements"] = before_cleaning-after_drop_duplicates
            
            self.times.loc[self.generation,
                           "time_duplicated_drop"] = time.perf_counter() - start_drop_duplicates


            # Removes all non valid programs in the population
            start_drop_invalids = time.perf_counter()
            
            self.drop_invalids(inplace=True)
            
            after_cleaning = len(self.population)
            if before_cleaning != after_cleaning:
                logging.debug(
                    f"{after_drop_duplicates-after_cleaning}/{after_drop_duplicates} invalid programs removed")
            
            self.times.loc[self.generation,
                           "time_invalids_drop"] = time.perf_counter() - start_drop_invalids

            #################################################################################
            #################################################################################
            # Refill population starts
            #################################################################################
            #################################################################################

            # Integrate population in case of too many invalid programs
            if len(self.population) < self.population_size * 2:
                start_refill_population = time.perf_counter()
                self.status = "Refilling Individuals"
                
                missing_elements = 2 * self.population_size - \
                    len(self.population)
                
                refill: List[Program] = list()

                logging.debug(
                    f"Population of {len(self.population)} elements is less than 2*population_size:{self.population_size*2}. Integrating with {missing_elements} new elements")

                for c in self.callbacks:
                    try:
                        c.on_refill_start()
                    except:
                        logging.warning(
                            f"Callback {c.__class__.__name__} raised an exception on refill start")
                        logging.warning(
                            traceback.format_exc())
                        
                while len(refill) < missing_elements:
                    elapsed_s = time.perf_counter() - start_refill_population

                    # ------------------------------------------------------------
                    # Decide how many to generate in this batch
                    # e.g.,  you can spawn `jobs` parallel tasks, each of which returns
                    # a list of new offspring from _get_offspring_batch.
                    # ------------------------------------------------------------
                    needed = missing_elements - len(refill)
                    batch_tasks = needed #min(5*jobs, needed)

                    # Make a local copy of the population (if needed),
                    # or call .as_binary() if your code requires it
                    try:
                        population_to_pass = self.population.as_binary()
                    except AttributeError:
                        population_to_pass = self.population

                    # ------------------------------------------------------------
                    # Run this batch in parallel via joblib
                    # Each delayed(...) call spawns a chunk of new offspring
                    # ------------------------------------------------------------
                    timeout_generate_individual = timeout_decorator_generation(50)(self.generate_individual)

                    results = Parallel(n_jobs=jobs)(
                        delayed(timeout_generate_individual)(
                            data=data,
                            features=self.features,
                            feature_assumption=self.feature_assumption,
                            feature_probability=self.feature_probability,
                            bool_update_feature_probability = self.bool_update_feature_probability,
                            operations=self.operations,
                            fitness_functions=self.fitness_functions,
                            const_range=self.const_range,
                            parsimony=self.parsimony,
                            parsimony_decay=self.parsimony_decay,
                            val_data=val_data
                        )
                        for _ in range(batch_tasks)
                    )
                    # `results` is now a list (each from _get_offspring_batch)
                    # Flatten them into `offsprings`
                        
                    refill += results

                    # ------------------------------------------------------------
                    # Print progress if verbose
                    # ------------------------------------------------------------
                    if self.verbose > 1:
                        curr_count = len(refill)
                        elapsed_s = max(time.perf_counter() - start_refill_population, 1e-9)
                        ops_s = round(curr_count / elapsed_s, 2)
                        elapsed_td = datetime.timedelta(seconds=int(elapsed_s))
                        print(
                            f"Refilling progress: {curr_count}/{self.population_size} "
                            f"({elapsed_td}, {ops_s} /s) ...",
                            end="\r",
                            flush=True
                        )

                    # Sleep or not – depends on whether you want to slow the loop
                    # time.sleep(0.2)  # sometimes helpful to reduce console spam

                # Finished or aborted early
                if self.verbose > 1:
                    total = len(refill)
                    elapsed_s = max(time.perf_counter() - start_refill_population, 1e-9)
                    ops_s = round(total / elapsed_s, 2)
                    elapsed_td = datetime.timedelta(seconds=int(elapsed_s))
                    print(
                        f"\nOffsprings generated: {total}/{self.population_size} "
                        f"({elapsed_td}, {ops_s} /s). Completed!"
                    )


                self.population: Population = Population(
                    self.population + refill)
                

                for c in self.callbacks:
                    try:
                        c.on_refill_end()
                    except:
                        logging.warning(
                            f"Callback {c.__class__.__name__} raised an exception on refill end")
                        logging.warning(
                            traceback.format_exc())

                # exludes every program in refill with an empty fitness
                new_pop=[]
                for p in self.population:
                    try:
                        if not p._has_incomplete_fitness:
                            new_pop.append(p)
                    except:
                        pass
                self.population=new_pop

                self.times.loc[self.generation,
                               "time_refill_invalid"] = time.perf_counter() - start_refill_population
                self.times.loc[self.generation,
                               "count_invalid_elements"] = missing_elements
                self.times.loc[self.generation,
                               "ratio_invalid_elements"] = missing_elements / len(self.population)

            #################################################################################
            #################################################################################
            # Refill population stops / Start drop residual duplicates
            #################################################################################
            #################################################################################
            before_cleaning = len(self.population)
            self.drop_duplicates(inplace=True)
            
            after_drop_duplicates = len(self.population)
            logging.debug(
                f"{before_cleaning-after_drop_duplicates}/{before_cleaning} duplicates programs removed")
            #################################################################################
            #################################################################################
            # End drop residual duplicates
            #################################################################################
            #################################################################################
            
            start_pareto_front_computation = time.perf_counter()

            self.status = "Pareto Front Computation"

            for c in self.callbacks:
                try:
                    c.on_pareto_front_computation_start()
                except:
                    logging.warning(
                        f"Callback {c.__class__.__name__} raised an exception on pareto front computation start")
                    logging.warning(
                        traceback.format_exc())


            # Calculates the Pareto front
            self._create_pareto_front()
            self.times.loc[self.generation,
                           "time_pareto_front_computation"] = time.perf_counter() - start_pareto_front_computation

            # Calculates srecond selection criterion
            start_second_criterion_computation = time.perf_counter()

            if self.genetic_algorithm == 'NSGA-II':
                self.population = self._select_final_population_NSGAII()

            elif self.genetic_algorithm == 'SMS-EMOEA':
                self.population = self._select_final_population_SMS_EMOEA()

            
            self.population = Population(self.population)

            self.times.loc[self.generation,
                           "count_average_complexity"] = self.average_complexity

            self.times.loc[self.generation,
                           "time_second_selection_criterion_computation"] = time.perf_counter() - start_second_criterion_computation



            ###### Convergence check START ######
            self._fpf_fitness_history[self.generation] = [
                {f.label: pr.fitness[f.label] for f in self.fitness_functions if f.minimize == True} for pr in self.first_pareto_front]

            current_nadir_fpf = self._compute_nadir_point(
                population=self.first_pareto_front)
            current_ideal_fpf = self._compute_ideal_point(
                population=self.first_pareto_front)

            self._nadir_points[self.generation] = current_nadir_fpf
            self._ideal_points[self.generation] = current_ideal_fpf

            if self.convergence_rolling_window is None and any(p.converged for p in self.population):
                if not self.converged_generation:
                    self.converged_generation = self.generation
                if self.verbose > 1:
                    print(
                        f"Training converged after {self.converged_generation} generations (performance threshold reached).")

            elif isinstance(self.convergence_rolling_window, int) and self.convergence_rolling_window > 0 and self._rolling_window_termination_criterion():
                if not self.converged_generation:
                    self.converged_generation = self.generation
                if self.verbose > 1:
                    print(
                        f"Training converged after {self.converged_generation} generations (rolling window threshold reached).")
            ###### Convergence check END ######

            for c in self.callbacks:
                try:
                    c.on_pareto_front_computation_end(
                        data=data, val_data=val_data)
                except:
                    logging.warning(
                        f"Callback {c.__class__.__name__} raised an exception on pareto front computation end")
                    logging.warning(
                        traceback.format_exc())

            self._print_first_pareto_front(verbose=self.verbose)
            end_time_generation = time.perf_counter()

            total_generation_time = end_time_generation - start_time_generation
            self.times.loc[self.generation,
                           "time_generation_total"] = total_generation_time

            for c in self.callbacks:
                try:
                    c.on_generation_end()
                except:
                    logging.warning(
                        f"Callback {c.__class__.__name__} raised an exception on generation end")
                    logging.warning(
                        traceback.format_exc())

            # Use generations = -1 to rely only on convergence (risk of infinite loop)
            bool_timeout = (time.perf_counter()- fit_start + self.times['time_generation_total'].tail(5).mean()) > self.training_timeout_mins*60
            bool_generation = (self.generation == self.generations_to_train)
            if (self.generations_to_train > 0 and bool_generation) or (self.training_timeout_mins > 0 and bool_timeout):
                for c in self.callbacks:
                    try:
                        c.on_training_completed()
                    except:
                        logging.warning(
                            f"Callback {c.__class__.__name__} raised an exception on training completed")
                        logging.warning(
                            traceback.format_exc())

                print(
                    f"Training completed {self.generation}/{self.generations_to_train} generations")
                return

            if self.converged_generation and self.stop_at_convergence:
                for c in self.callbacks:
                    try:
                        c.on_convergence()
                    except:
                        logging.warning(
                            f"Callback {c.__class__.__name__} raised an exception on convergence")
                        logging.warning(
                            traceback.format_exc())

                print(
                    f"Training converged after {self.converged_generation} generations and requested to stop.")
                return

            if self._callback_terminate:
                print(f"Training terminated by callback")
                return

    def generate_individual(self,
                            data: Union[dict, pd.DataFrame, pd.Series], 
                            features: List[str],
                            feature_assumption: dict,
                            feature_probability: List[float], 
                            bool_update_feature_probability: bool, 
                            operations: List[dict],
                            fitness_functions: List[BaseFitness], 
                            const_range: tuple = (0, 1),
                            parsimony: float = 0.8, 
                            parsimony_decay: float = 0.85,
                            val_data: Union[dict, pd.DataFrame, pd.Series] = None) -> Program:
        """
        This method generates a new individual for the population

        Args:
            - data: Union[dict, pd.DataFrame, pd.Series]
                The data on which the performance are evaluated. We could use compute_fitness
                later, but we need to evaluate the fitness here to compute it in the
                parallel initialization.
            - features: List[str]
                The list of features to be used in the generation
            - operations: List[dict]
                The list of operations to be used in the generation
            - const_range: tuple
                The range of the constants to be used in the generation. There will then be
                adapted during the optimization process.
            - fitness_functions: List[BaseFitness]
                The list of fitness functions to be used in the generation
            - parsimony: float (default 0.8)
                The parsimony coefficient to be used in the generation. This modulates how
                likely the program is to be complex (deep) or simple (shallow).
            - parsimony_decay: float (default 0.85)
                The parsimony decay to be used in the generation. This modulates how the parsimony
                coefficient is updated at each generation. The parsimony coefficient is multiplied
                by this value at each generation. This allows to decrease the parsimony coefficient
                over time to prevent the program to be too complex (deep).
            - val_data: Union[dict, pd.DataFrame, pd.Series] (default None)
                The data on which the validation is performed

        Returns:
            - p: Program
                The generated program

        """

        new_p = Program(features=features, 
                        feature_assumption=feature_assumption,
                        feature_probability=feature_probability,
                        default_feature_probability=feature_probability,
                        bool_update_feature_probability=bool_update_feature_probability,
                        operations=operations, 
                        const_range=const_range,
                        parsimony=parsimony, 
                        parsimony_decay=parsimony_decay)

        new_p.init_program()
        new_p.compute_fitness(fitness_functions=fitness_functions,
                              data=data, validation=False, simplify=True)
        
        ## COMPUTE FEATURE PROBABILITY
        if new_p.bool_update_feature_probability:
            new_p.feature_probability=new_p.update_feature_probability(data=data)

        if val_data is not None:
            new_p.compute_fitness(
                fitness_functions=fitness_functions, data=val_data, validation=True)
        
        #print(f'New p: {new_p.program}\nFeature prob. {new_p.feature_probability}\nDefault Feature prob. {new_p.default_feature_probability}\n\n')

        return new_p


    def _tournament_selection(self, population: Population, tournament_size: int, generation: int) -> Program:
        """
        Tournament selection

        This method selects the best program from a random sample of the population.
        It is used to apply the genetic operators during the generation of the
        offsprings.

        We firstly chose a random set of K programs from the population. Then we
        select the best program from this set. If the generation is 0, we use the
        program's complexity to select the best program. In the other generations
        we use the pareto front rank and the crowding distance.

        Args:
            - population: Population
                The population of programs
            - tournament_size: int
                The size of the tournament
            - generation: int
                The current generation

        Returns:
            - best_member: Program
                The best program selected by the tournament selection
        """

        tournament_members: List[Program] = random.choices(
            population, k=tournament_size)

        best_member = tournament_members[0]

        for member in tournament_members:
            if member is None or not member.is_valid:
                continue

            if self.genetic_algorithm == 'NSGA-II':
                if best_member == None or \
                        member.rank < best_member.rank or \
                        (member.rank == best_member.rank and
                            member.crowding_distance > best_member.crowding_distance):
                    best_member = member

            elif self.genetic_algorithm == 'SMS-EMOEA':
                if best_member == None or \
                        member.rank < best_member.rank or \
                        (member.rank == best_member.rank and
                            member.exclusive_hypervolume > best_member.exclusive_hypervolume):
                    best_member = member

        return best_member

    def _get_offspring(self,
                       data: Union[dict, pd.DataFrame, pd.Series], 
                       genetic_operators_frequency: Dict[str, float], 
                       fitness_functions: List[BaseFitness], 
                       population: Population, 
                       tournament_size: int, 
                       generation: int, 
                       val_data: Union[Dict, pd.Series, pd.DataFrame] = None) -> Program:
        """
        This method generates an offspring of a program from the current population

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
            - do nothing: in this case no mutation is applied and the program is returned as is

        The frequency of which those operation are applied is determined by the dictionary
        genetic_operations_frequency in which the relative frequency of each of the desired operations
        is expressed with integers. The higher the number the likelier the operation will be chosen.
        Use integers > 1 to increase the frequency of an operation. Use 1 as minimum value.

        The program to which apply the operation is chosen using the tournament_selection, a method
        that identify the best program among a random selection of k programs from the population.

        Args:
            - data: Union[dict, pd.DataFrame, pd.Series]
                The data on which the performance are evaluated. We could use compute_fitness
                later, but we need to evaluate the fitness here to compute it in the
                parallel initialization.
            - genetic_operators_frequency: Dict[str, float]
                The dictionary of the genetic operators and their frequency
            - fitness_functions: List[BaseFitness]
                The list of fitness functions to be used in the generation
            - population: List[Program]
                The current population from which to choose the program to which apply the genetic
                operator
            - tournament_size: int
                The size of the tournament selection
            - generation: int
                The current generation
            - val_data: Union[Dict, pd.Series, pd.DataFrame] (default None)
                The data on which the validation is performed

        Returns:
            - _offspring: Program
                The offspring of the current population

            - program1: Program     (only in case of errors or if the program is not valid)
                The program from which the offspring is generated
        """
        #initialize a timeoutdecorator compatible with joblib
        from functools import wraps
        import threading
        import time
        
        # Timeout decorator that works with joblib
        def timeout(max_seconds):
            def decorator(func):
                @wraps(func)
                def wrapper(*args, **kwargs):
                    result = [TimeoutError(f"Function timed out after {max_seconds} seconds")]
                    
                    def target():
                        try:
                            result[0] = func(*args, **kwargs)
                        except Exception as e:
                            result[0] = e
                    
                    thread = threading.Thread(target=target)
                    thread.daemon = True
                    thread.start()
                    thread.join(max_seconds)
                    
                    if isinstance(result[0], Exception):
                        raise result[0]
                    return result[0]
                return wrapper
            return decorator

        # Function to perform genetic operation with timeout
        @timeout(50)
        def perform_genetic_operation(gen_op, program1, program2=None):
            if gen_op == 'crossover':
                return program1.cross_over(other=program2, inplace=False)
            elif gen_op == 'mutation':
                return program1.mutate(inplace=False)
            elif gen_op == 'delete_node':
                return program1.delete_node(inplace=False)
            elif gen_op == 'insert_node':
                return program1.insert_node(inplace=False)
            elif gen_op == 'mutate_operator':
                return program1.mutate_operator(inplace=False)
            elif gen_op == 'mutate_leaf':
                return program1.mutate_leaf(inplace=False)
            elif gen_op == 'recalibrate':
                return program1.recalibrate(inplace=False)
            elif gen_op == 'regularize_brgflow':
                return program1.regularize_brgflow(data=data, inplace=False)
            elif gen_op == 'regularize_bootstrap':
                return program1.regularize_bootstrap(data=data, inplace=False)
            elif gen_op == 'additive_expansion':
                return program1.additive_expansion(inplace=False)
            elif gen_op == 'regularize_sensitivity':
                return program1.regularize_sensitivity(data=data,inplace=False)
            else:
                logging.warning(
                    f'Supported genetic operations: crossover, delete_node, do_nothing, insert_node, mutate_leaf, mutate_operator, simplification, mutation and randomize'
                )
                print('asking for non-existing mutations!!!!')
                return program1
            

        #initialize successfull brgflow an bootstrap counters
        successfull_brgflow=0
        successfull_bootstrap=0

        ## select program via tournament selection
        program1 = self._tournament_selection(
            population=population, tournament_size=tournament_size, generation=generation)
        
        # Select the genetic operation to apply
        # The frequency of each operation is determined by the dictionary
        # genetic_operators_frequency. The higher the number the likelier the operation will be chosen.
        ops = list()
        for op, freq in genetic_operators_frequency.items():
            if not ( 
                ((op=='regularize_bootstrap') and (len(program1.get_constants())<=2 or program1.already_bootstrapped)) 
                or 
                ((op=='regularize_brgflow') and (len(program1.get_constants())<=2 or program1.already_brgflow)) 
                or 
                ((op=='regularize_sensitivity') and (len(program1.get_constants())<=2 or program1.already_bootstrapped))
                ):
                ops += [op] * freq  
        gen_op = random.choice(ops)

        start = time.perf_counter()

        try:
            if  not program1.is_valid:
                # If the program is not valid, we return a random tree.
                # The invalid program will be removed from the population later.
                #print(f'Program 1 not valid. ')
                new = Program(operations=program1.operations,
                            features=program1.features,
                            feature_assumption=self.feature_assumption,
                            feature_probability=program1.default_feature_probability,
                            default_feature_probability=program1.default_feature_probability,
                            bool_update_feature_probability=program1.bool_update_feature_probability,
                            const_range=program1.const_range,
                            parsimony=program1.parsimony,
                            parsimony_decay=program1.parsimony_decay)

                new.init_program()
                new.compute_fitness(
                    fitness_functions=fitness_functions, data=data, validation=False, simplify=True)
                if val_data is not None:
                    new.compute_fitness(
                        fitness_functions=fitness_functions, data=val_data, validation=True, simplify=False)
                #print(f'New program {new.program}.\n')
                end = time.perf_counter()
                return new, successfull_brgflow, successfull_bootstrap, gen_op, end-start

            _offspring: Program = None
            simplify = True

            if gen_op == 'crossover':
                program2 = self._tournament_selection(
                    population=population, tournament_size=tournament_size, generation=generation)
                if not program2.is_valid:
                    new = Program(operations=program2.operations,
                                features=program2.features,
                                feature_assumption=self.feature_assumption,
                                feature_probability=program2.default_feature_probability,
                                default_feature_probability=program2.default_feature_probability,
                                bool_update_feature_probability=program2.bool_update_feature_probability,
                                const_range=program2.const_range,
                                parsimony=program2.parsimony,
                                parsimony_decay=program2.parsimony_decay)

                    new.init_program()
                    new.compute_fitness(
                        fitness_functions=fitness_functions, data=data, validation=False, simplify=True)
                    if val_data is not None:
                        new.compute_fitness(
                            fitness_functions=fitness_functions, data=val_data, validation=True, simplify=False)
                    end = time.perf_counter()
                    return new, successfull_brgflow, successfull_bootstrap, gen_op, end-start
                
                _offspring = perform_genetic_operation(gen_op, program1, program2)
            
            elif gen_op in ['regularize_brgflow', 'regularize_bootstrap','regularize_sensitivity']:
                result = perform_genetic_operation(gen_op, program1)
                if gen_op == 'regularize_brgflow':
                    _offspring, successfull_brgflow = result
                else:  # regularize_bootstrap
                    _offspring, successfull_bootstrap = result
                simplify = False
                
            else:  # All other genetic operations
                _offspring = perform_genetic_operation(gen_op, program1)
            
            #ensure the new program has a suitable feature assumption dictionary
            _offspring.feature_assumption=self.feature_assumption
            # Add the fitness to the object after the genetic operation
            _offspring.compute_fitness(
                fitness_functions=fitness_functions, data=data, simplify=simplify)
            if val_data is not None:
                _offspring.compute_fitness(
                    fitness_functions=fitness_functions, data=val_data, validation=True, simplify=False)

            ## COMPUTE FEATURE PROBABILITY
            if _offspring.bool_update_feature_probability:
                _offspring.feature_probability=_offspring.update_feature_probability(data=data)

            # Reset the hash to force the re-computation
            _offspring._hash = None
            end = time.perf_counter()
            return _offspring, successfull_brgflow, successfull_bootstrap, gen_op, end-start
           
        except TimeoutError:
            end = time.perf_counter()
            print('Timeout', gen_op, end-start)
            print(program1.program)
            return program1, successfull_brgflow, successfull_bootstrap, gen_op, end-start

    def _get_offspring_old(self,
                       data: Union[dict, pd.DataFrame, pd.Series], 
                       genetic_operators_frequency: Dict[str, float], 
                       fitness_functions: List[BaseFitness], 
                       population: Population, 
                       tournament_size: int, 
                       generation: int, 
                       val_data: Union[Dict, pd.Series, pd.DataFrame] = None) -> Program:
        """
        This method generates an offspring of a program from the current population

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
            - do nothing: in this case no mutation is applied and the program is returned as is

        The frequency of which those operation are applied is determined by the dictionary
        genetic_operations_frequency in which the relative frequency of each of the desired operations
        is expressed with integers. The higher the number the likelier the operation will be chosen.
        Use integers > 1 to increase the frequency of an operation. Use 1 as minimum value.

        The program to which apply the operation is chosen using the tournament_selection, a method
        that identify the best program among a random selection of k programs from the population.

        Args:
            - data: Union[dict, pd.DataFrame, pd.Series]
                The data on which the performance are evaluated. We could use compute_fitness
                later, but we need to evaluate the fitness here to compute it in the
                parallel initialization.
            - genetic_operators_frequency: Dict[str, float]
                The dictionary of the genetic operators and their frequency
            - fitness_functions: List[BaseFitness]
                The list of fitness functions to be used in the generation
            - population: List[Program]
                The current population from which to choose the program to which apply the genetic
                operator
            - tournament_size: int
                The size of the tournament selection
            - generation: int
                The current generation
            - val_data: Union[Dict, pd.Series, pd.DataFrame] (default None)
                The data on which the validation is performed

        Returns:
            - _offspring: Program
                The offspring of the current population

            - program1: Program     (only in case of errors or if the program is not valid)
                The program from which the offspring is generated
        """

        # Select the genetic operation to apply
        # The frequency of each operation is determined by the dictionary
        # genetic_operators_frequency. The higher the number the likelier the operation will be chosen.
        successfull_brgflow=0
        successfull_bootstrap=0

        program1 = self._tournament_selection(
            population=population, tournament_size=tournament_size, generation=generation)
        
 
        ops = list()
        for op, freq in genetic_operators_frequency.items():
            if not ( ((op=='regularize_bootstrap') and (len(program1.get_constants())<=2 or program1.already_bootstrapped)) or ((op=='regularize_brgflow') and (len(program1.get_constants())<=2 or program1.already_brgflow))):
                ops += [op] * freq  
        gen_op = random.choice(ops)

        start = time.perf_counter()


        import signal
        max_time=3
        def timeout_offspring_handler(signum, frame):
            raise TimeoutError(f"Offspring operation timed out after {max_time}s")
        
        # Set the timeout alarm
        signal.signal(signal.SIGALRM, timeout_offspring_handler)
        signal.alarm(3)  
        try:
            if  not program1.is_valid:
                # If the program is not valid, we return a random tree.
                # The invalid program will be removed from the population later.
                #print(f'Program 1 not valid. ')
                new = Program(operations=program1.operations,
                            features=program1.features,
                            feature_assumption=self.feature_assumption,
                            feature_probability=program1.default_feature_probability,
                            default_feature_probability=program1.default_feature_probability,
                            bool_update_feature_probability=program1.bool_update_feature_probability,
                            const_range=program1.const_range,
                            parsimony=program1.parsimony,
                            parsimony_decay=program1.parsimony_decay)

                new.init_program()
                new.compute_fitness(
                    fitness_functions=fitness_functions, data=data, validation=False, simplify=True)
                if val_data is not None:
                    new.compute_fitness(
                        fitness_functions=fitness_functions, data=val_data, validation=True, simplify=False)
                #print(f'New program {new.program}.\n')
                end = time.perf_counter()
                signal.alarm(0)
                print(gen_op, end-start)
                return new, successfull_brgflow, successfull_bootstrap, gen_op, end-start

            _offspring: Program = None
            if gen_op == 'crossover':
                program2 = self._tournament_selection(
                    population=population, tournament_size=tournament_size, generation=generation)
                if not program2.is_valid:
                    new = Program(operations=program2.operations,
                                features=program2.features,
                                feature_assumption=self.feature_assumption,
                                feature_probability=program2.default_feature_probability,
                                default_feature_probability=program2.default_feature_probability,
                                bool_update_feature_probability=program2.bool_update_feature_probability,
                                const_range=program2.const_range,
                                parsimony=program2.parsimony,
                                parsimony_decay=program2.parsimony_decay)

                    new.init_program()
                    new.compute_fitness(
                        fitness_functions=fitness_functions, data=data, validation=False, simplify=True)
                    if val_data is not None:
                        new.compute_fitness(
                            fitness_functions=fitness_functions, data=val_data, validation=True, simplify=False)
                    end = time.perf_counter()
                    signal.alarm(0)
                    print(gen_op, end-start)
                    return new, successfull_brgflow, successfull_bootstrap, gen_op, end-start
                
                _offspring = program1.cross_over(
                    other=program2, inplace=False)
                simplify=True

            elif gen_op == 'mutation':
                _offspring = program1.mutate(inplace=False)
                simplify=True

            elif gen_op == 'delete_node':
                _offspring = program1.delete_node(inplace=False)
                simplify=True

            elif gen_op == 'insert_node':
                _offspring = program1.insert_node(inplace=False)
                simplify=True

            elif gen_op == 'mutate_operator':
                _offspring = program1.mutate_operator(inplace=False)
                simplify=True

            elif gen_op == 'mutate_leaf':
                _offspring = program1.mutate_leaf(inplace=False)
                simplify=True

            elif gen_op == 'recalibrate':
                _offspring = program1.recalibrate(inplace=False)
                simplify=True

            elif gen_op == 'regularize_brgflow':
                _offspring, successfull_brgflow = program1.regularize_brgflow(data=data,
                                                        inplace=False)
                simplify=False
            
            elif gen_op == 'regularize_bootstrap':
                _offspring, successfull_bootstrap = program1.regularize_bootstrap(data=data,
                                                        inplace=False)
                simplify=False
            
            elif gen_op == 'additive_expansion':
                _offspring = program1.additive_expansion(inplace=False)
                simplify=True

            else:
                logging.warning(
                    f'Supported genetic operations: crossover, delete_node, do_nothing, insert_node, mutate_leaf, mutate_operator, simplification, mutation and randomize'
                )

                print('asking for non-existing mutations!!!!')
                return program1
            
            #ensure the new program has a suitable feature assumption dictionary
            _offspring.feature_assumption=self.feature_assumption
            # Add the fitness to the object after the genetic operation
            _offspring.compute_fitness(
                fitness_functions=fitness_functions, data=data, simplify=simplify)
            if val_data is not None:
                _offspring.compute_fitness(
                    fitness_functions=fitness_functions, data=val_data, validation=True, simplify=False)

            ## COMPUTE FEATURE PROBABILITY
            if _offspring.bool_update_feature_probability:
                _offspring.feature_probability=_offspring.update_feature_probability(data=data)

            #print(f'Result: {_offspring.program}\n')#Feature prob. {_offspring.feature_probability}')

            # Reset the hash to force the re-computation
            _offspring._hash = None
            end = time.perf_counter()
            signal.alarm(0)
            print(gen_op, end-start)
            return _offspring, successfull_brgflow, successfull_bootstrap, gen_op, end-start
           
        except TimeoutError:
            signal.alarm(0)
            end = time.perf_counter()
            print(gen_op, max_time, 'Timeout')
            return program1, successfull_brgflow, successfull_bootstrap, gen_op, max_time


    def _generate_one_offspring(self,
                                data: Union[dict, pd.DataFrame, pd.Series],
                                genetic_operators_frequency: Dict[str, float],
                                fitness_functions: List[BaseFitness],
                                population: Population,
                                tournament_size: int,
                                generation: int,
                                #batch_size: int,
                                val_data: Union[Dict, pd.Series, pd.DataFrame] = None
                                ) -> List[Program]:
        """
        Generate `batch_size * n_jobs` offspring in parallel using joblib.
        - Continues generating until we have enough *valid* offspring
        (those that don't have incomplete fitness).
        - If `queue` is given, offspring go there (and we return an empty list).
        Otherwise, we return a list of generated offspring.

        CAUTION: This is "inner" parallelism. If your "outer" code also uses joblib,
        you might want to reduce either the outer or inner n_jobs to avoid nested
        oversubscription. 
        """

        # Optionally convert population to "program"
        try:
            population = population.as_program()
        except AttributeError:
            pass

        # If the status changed (user code might set self.status != "Generating Offsprings"), abort
        if self.status != "Generating Offsprings":
            return None

        # Call your method that actually creates a single offspring
        
        offspring, successfull_brgflow, successfull_bootstrap, gen_op, gen_op_time = self._get_offspring(
            data=data,
            val_data=val_data,
            genetic_operators_frequency=genetic_operators_frequency,
            fitness_functions=fitness_functions,
            population=population,
            tournament_size=tournament_size,
            generation=generation
        )
        # Skip "incomplete" offspring
        if offspring._has_incomplete_fitness:
            return None, 0, 0
        return offspring, successfull_brgflow, successfull_bootstrap, gen_op, gen_op_time
 

    @property
    def metadata(self) -> Dict[str, Any]:
        """
        This method returns the metadata of the current population

        Args:
            - None

        Returns:
            - metadata: Dict[str, Any]
                The metadata of the current population
        """
        metadata = {
            'average_complexity': self.average_complexity,
            'client_name': self.client_name,
            'const_range': self.const_range,
            'converged_generation': self.converged_generation,
            'fitness_functions': [ff.__repr__() for ff in self.fitness_functions],
            'fpf_hypervolume': self.fpf_hypervolume,
            'fpf_tree_diversity': self.fpf_tree_diversity,
            'generation': self.generation,
            'generations_to_train': self.generations_to_train,
            'genetic_algorithm': self.genetic_algorithm,
            'genetic_operators_frequency': self.genetic_operators_frequency,
            'operations': [op['symbol'] for op in self.operations],
            'parsimony_decay': self.parsimony_decay,
            'parsimony': self.parsimony,
            'population_size': self.population_size,
            'total_time': self._total_time,
            'tournament_size': self.tournament_size,
        }

        return metadata

    def _compute_nadir_point(self, population: List) -> Dict[str, float]:
        """
        This method returns the nadir point of the current population.
        The nadir point is the worst point of the current population.

        Args:
            - None

        Returns:
            - nadir: Dict[str, float]
                The nadir point of the current population
        """

        nadir = dict()
        for f in self.fitness_functions:
            if not f.minimize:
                continue
            nadir[f.label] = max([p.fitness[f.label]
                                 for p in population if p.fitness[f.label] < np.inf])

            if f.hypervolume_reference is None or nadir[f.label] > f.hypervolume_reference:
                f.hypervolume_reference = nadir[f.label] + 1e-2

        return nadir

    def _compute_ideal_point(self, population: List) -> Dict[str, float]:
        """
        This method returns the ideal point of the current population.
        The ideal point is the best point of the current population.

        Args:
            - None

        Returns:
            - ideal: Dict[str, float]
                The ideal point of the current population
        """

        ideal = dict()
        for f in self.fitness_functions:
            if not f.minimize:
                continue
            ideal[f.label] = min([p.fitness[f.label] for p in population])

        return ideal

    def _delta_nadir(self, generation: int) -> float:
        """
        This method returns the delta nadir of the current population.
        The delta nadir is the difference between the nadir point of the current
        population and the nadir point of the previous population normalized by the
        nadir point minus the ideal point of the current population.

        Args:
            - generation: int


        Returns:
            - delta_nadir: float
                The delta nadir of the current population
        """

        nadir_prev: Dict = self.nadir_points[generation-1]
        nadir: Dict = self.nadir_points[generation]
        ideal: Dict = self.ideal_points[generation]

        delta_nadir = max([(nadir_prev[f] - nadir[f]) / (nadir[f] - ideal[f] if nadir[f] -
                          ideal[f] != 0 else 1) for f in list(nadir.keys())])

        return delta_nadir

    def _rolling_delta_nadir(self, rolling_window: int) -> float:
        """
        This method returns the rolling delta nadir of the current population.
        The rolling delta nadir is the difference between the nadir point of the current
        population and the nadir point of the previous population normalized by the
        nadir point minus the ideal point of the current population.

        Args:
            - rolling_window: int
                The rolling window to use to compute the rolling delta nadir

        Returns:
            - rolling_delta_nadir: float
                The rolling delta nadir of the current population
        """

        gen_start = max(self.generation - rolling_window + 1, 1)
        rolling_delta_nadir = []
        for gen in range(gen_start, self.generation + 1):
            rolling_delta_nadir.append(self._delta_nadir(gen))

        return rolling_delta_nadir

    def _delta_ideal(self, generation: int) -> float:
        """
        This method returns the delta ideal of the current population.
        The delta ideal is the difference between the ideal point of the current
        population and the ideal point of the previous population normalized by the
        nadir point minus the ideal point of the current population.

        Args:
            - generation: int
                The generation for which to compute the delta ideal

        Returns:
            - delta_ideal: float
                The delta ideal of the current population
        """

        nadir: Dict = self.nadir_points[generation]
        ideal_prev: Dict = self.ideal_points[generation-1]
        ideal: Dict = self.ideal_points[generation]

        delta_ideal = max([(ideal_prev[f] - ideal[f]) / (nadir[f] - ideal[f] if nadir[f] -
                          ideal[f] != 0 else 1) for f in list(nadir.keys())])

        return delta_ideal

    def _rolling_delta_ideal(self, rolling_window: int) -> float:
        """
        This method returns the rolling delta ideal of the current population.
        The rolling delta ideal is the difference between the ideal point of the current
        population and the ideal point of the previous population normalized by the
        nadir point minus the ideal point of the current population.

        Args:
            - rolling_window: int
                The rolling window to use to compute the rolling delta nadir

        Returns:
            - rolling_delta_ideal: float
                The rolling delta ideal of the current population
        """

        gen_start = max(self.generation - rolling_window + 1, 1)
        rolling_delta_ideal = []
        for gen in range(gen_start, self.generation + 1):
            rolling_delta_ideal.append(self._delta_ideal(gen))

        return rolling_delta_ideal

    def _rolling_inverted_generational_distance_plus(self, rolling_window: int) -> float:
        """
        This method returns the inverted generational distance plus of the current population.
        The inverted generational distance plus is the sum of the distances between the
        current population and the previous population.

        Args:
            - rolling_window: int
                The rolling window to use to compute the rolling delta nadir

        Returns:
            - inverted_generational_distance_plus: float
                The inverted generational distance plus of the current population
        """
        from pymoo.indicators.igd_plus import IGDPlus

        max_nadir = {ftn.label: np.max([self.nadir_points[gen][ftn.label] for gen in self.nadir_points.keys(
        )]) for ftn in self.fitness_functions if ftn.minimize}
        min_ideal = {ftn.label: np.min([self.ideal_points[gen][ftn.label] for gen in self.ideal_points.keys(
        )]) for ftn in self.fitness_functions if ftn.minimize}

        gen_start = max(self.generation - rolling_window, 1)

        # For each element of self._fpf_fitness_history[gen_start] and for each fitness function
        # we rescale the value using min_ideal and max_nadir

        rescale_ref = dict()

        for gen in range(gen_start, self.generation + 1):
            rescale_ref[gen] = list()
            for point in self._fpf_fitness_history[gen]:
                rescale_ref[gen].append([(f_value - min_ideal[f_label]) / (
                    max_nadir[f_label] - min_ideal[f_label]) for f_label, f_value in point.items()])

        igd_window = []
        for gen in range(gen_start + 1, self.generation + 1):
            metric = IGDPlus(np.array(rescale_ref[gen]), zero_to_one=True)
            igd = metric.do(np.array(rescale_ref[gen - 1]))
            igd_window.append(igd)

        return igd_window

    def _rolling_window_termination_criterion(self) -> bool:
        """
        Check if the termination criterion for the rolling window has been met.

        Returns:
            bool: True if the termination criterion is met, False otherwise.
        """
        if self.convergence_rolling_window is None or self.generation < (self.convergence_rolling_window + 1):
            return False

        max_roll_delta_nadir = max(self._rolling_delta_nadir(
            rolling_window=self.convergence_rolling_window))
        max_roll_delta_ideal = max(self._rolling_delta_ideal(
            rolling_window=self.convergence_rolling_window))
        max_roll_igd = max(self._rolling_inverted_generational_distance_plus(
            rolling_window=self.convergence_rolling_window))

        if self.verbose > 2:
            print(
                f"Convergence rolling window: {self.generation}/{self.generations_to_train} - Delta Nadir: {max_roll_delta_nadir:3f} - Delta Ideal: {max_roll_delta_ideal:3f} - IGD: {max_roll_igd:3f}")
        if max_roll_delta_nadir > self.convergence_rolling_window_threshold:
            return False

        if max_roll_delta_ideal > self.convergence_rolling_window_threshold:
            return False

        if max_roll_igd > self.convergence_rolling_window_threshold:
            return False

        return True

    def plot_generations_time(self):
        """ This method plots the time spent in each generation

        Args:
            - None

        Returns:
            - fig: go.Figure
                The plotly figure
        """

        import plotly.express as px

        def format_time(seconds):
            return datetime.datetime.utcfromtimestamp(seconds)

        self.times['time_generation_total_formatted'] = self.times['time_generation_total'].apply(
            format_time)

        fig = px.line(
            self.times, y='time_generation_total_formatted', title='Generation Time')
        fig.update_yaxes(title='Time', tickformat='%M:%S')
        return fig

    def plot_best_individual(self, fitness: str, on_val: bool = False):
        """ This method plots the best individual in a given fitness for each generation

        Args:
            - fitness: str
                The name of the fitness to plot
            - on_val: bool (default False)
                Whether to plot the performance on the validation data or on the training data

        Returns:
            - fig: go.Figure
                The plotly figure
        """

        if not fitness in [f.label for f in self.fitness_functions]:
            raise ValueError(
                f'Fitness {fitness} not found in the fitness functions. Please use one of the following: {", ".join([f.label for f in self.fitness_functions])}')

        import plotly.express as px

        branch = 'validation' if on_val else 'training'

        if on_val and not self.best_history.get('validation'):
            logging.warning(
                f'No validation data available. Plotting training data instead')
            branch = 'training'

        data = [
            {'Generation': generation,
             fitness: decompress(self.best_history[branch][generation][fitness]).fitness_validation[fitness] if on_val else decompress(
                 self.best_history[branch][generation][fitness]).fitness[fitness]
             } for generation in self.best_history[branch]
        ]

        # Create a scatter plot with the generation on the x-axis and the fitness on the y-axis
        fig = px.scatter(data, x='Generation', y=fitness,
                         title=f'Best individual on {fitness} for each generation on {branch} set')

        return fig

    def plot_compare_fitness(self, perf_x: str, perf_y: str, pf_rank: Union[int, str] = 1, generation: int = None, on_val: bool = False,
                             marker_dict: Dict[str, Any] = dict(size=5, color='grey'), highlight_best: bool = True,
                             xlim: Tuple[float, float] = None, ylim: Tuple[float, float] = None,
                             xlog: bool = False, ylog: bool = False,
                             figsize: Union[Dict[str,  Any], None] = None, title: str = None):
        """
        This method plots the distribution of two performance metrics for the programs in the first pareto front

        Args:
            - perf_x: str
                The name of the first performance metric to plot
            - perf_y: str
                The name of the second performance metric to plot
            - pf_rank: Union[int, str] (default 1)
                The rank of the pareto front to plot
            - generation: int (default None)
                The generation to plot. If -1, the last generation is plotted
            - on_val: bool (default False)
                Whether to plot the performance on the validation data or on the training data
            - marker_dict: Dict[str, Any] (default dict(size=5,color='grey'))
                The dictionary of the marker to use in the plot
            - highlight_best: bool (default True)
                Whether to highlight the best program in the pareto front
            - xlim: Tuple[float, float] (default None)
                The limits of the x axis
            - ylim: Tuple[float, float] (default None)
                The limits of the y axis
            - xlog: bool (default False)
                Whether to use a log scale for the x axis
            - ylog: bool (default False)
                Whether to use a log scale for the y axis
            - figsize: Union[Dict[str,  Any], None] (default None)
                The size of the figure

        Returns:
            - fig: go.Figure    
                The plotly figure
        """

        if generation and generation != self.generation and pf_rank > 1:
            raise ValueError(
                "Cannot plot pareto fronts beyond the first for historic generations")

        generation = max(self.first_pareto_front_history.keys()
                         ) if not generation else generation

        if isinstance(pf_rank, int):
            iterate_over = self.extract_pareto_front(
                population=self.population, rank=pf_rank)
        elif pf_rank == 'all':
            iterate_over = self.population
        elif re.match(r'^\d+-\d+$', pf_rank):
            rank_start, rank_end = map(int, pf_rank.split('-'))
            iterate_over = []
            for r in range(rank_start, rank_end+1):
                iterate_over.extend(self.extract_pareto_front(
                    population=self.population, rank=r))

        if not self.first_pareto_front_history.get(generation):
            raise ValueError(
                f"Generation {generation} not found in the history. Please use one of the following: {', '.join([str(g) for g in self.first_pareto_front_history.keys()])}")

        # iterate_over = self.first_pareto_front_history[generation]

        try:
            iterate_over = decompress(iterate_over)
        except TypeError:
            logging.warning(
                f"Legacy version. PF was not compressed. Consider using compress() to save memory and disk space")

        perf_df = pd.DataFrame()
        for index, p in enumerate(iterate_over):
            p: 'Program'
            if on_val:
                for perf in p.fitness_validation:
                    perf_df.loc[index, perf] = p.fitness_validation[perf]
            else:
                for perf in p.fitness:
                    perf_df.loc[index, perf] = p.fitness[perf]

        fig = go.Figure()

        perf_df['color'] = marker_dict.get('color', 'grey')

        if highlight_best:
            perf_x_max = perf_df[perf_x].max()
            perf_x_min = perf_df[perf_x].min()
            perf_y_max = perf_df[perf_y].max()
            perf_y_min = perf_df[perf_y].min()
            perf_x_max_index = perf_df[perf_df[perf_x] == perf_x_max].index[0]
            perf_x_min_index = perf_df[perf_df[perf_x] == perf_x_min].index[0]
            perf_y_max_index = perf_df[perf_df[perf_y] == perf_y_max].index[0]
            perf_y_min_index = perf_df[perf_df[perf_y] == perf_y_min].index[0]
            perf_df.loc[perf_x_max_index, 'color'] = 'red'
            perf_df.loc[perf_y_max_index, 'color'] = 'red'
            perf_df.loc[perf_x_min_index, 'color'] = 'blue'
            perf_df.loc[perf_y_min_index, 'color'] = 'blue'

        # Add the points to the plot
        fig.add_trace(go.Scatter(
            x=perf_df[perf_x],
            y=perf_df[perf_y],
            mode='markers',
            marker=marker_dict,
            marker_color=perf_df['color'],
            # customdata=list(perf_df.columns)
        ))

        fig.update_traces(
            hovertemplate="<br>".join([
                f"{perf_x}: %{{x}}",
                f"{perf_y}: %{{y}}",
                # f"Program: %{{customdata[0]}}",
                # f"Complexity: %{{customdata[1]}}"
            ])
        )

        title = title if title is not None else f"Distribution of {perf_x} and {perf_y} for the {pf_rank} Pareto Front of generation {generation}/{self.generations_to_train} on {'validation' if on_val else 'training'} data"

        fig.update_layout(
            title=title,
            xaxis_title=perf_x,
            yaxis_title=perf_y
        )

        if xlim is not None:
            fig.update_xaxes(range=xlim)
        if ylim is not None:
            fig.update_yaxes(range=ylim)
        if xlog:
            fig.update_xaxes(type="log")
        if ylog:
            fig.update_yaxes(type="log")

        if isinstance(figsize, Dict):
            fig.update_layout(
                autosize=False,
                width=figsize['width'],
                height=figsize['height'],
            )

        # Set background color to white but with the grid lines
        fig.update_layout(
            plot_bgcolor="white",
            xaxis=dict(
                gridcolor="lightgrey",
                gridwidth=2,
            ),
            yaxis=dict(
                gridcolor="lightgrey",
                gridwidth=2,
            ),
        )

        return fig

    def plot_convergence(self, logy: bool = True):

        df = pd.DataFrame(index=range(1, self.generation))

        df['IGD+'] = self._rolling_inverted_generational_distance_plus(
            rolling_window=self.generation-1)
        df['Ideal'] = self._rolling_delta_ideal(
            rolling_window=self.generation-1)
        df['Nadir'] = self._rolling_delta_nadir(
            rolling_window=self.generation-1)

        df.rolling(window=self.convergence_rolling_window).max().plot()

        plt.title('Convergence')
        plt.xlabel('Generation')
        plt.ylabel('Value')
        if logy:
            plt.yscale('log')
        plt.grid()
        return plt

    def _print_first_pareto_front(self, verbose: int = 3):
        """
        Print best programs

        This method prints the programs of the first pareto front of the current population.

        Args:
            - None
        Returns:
            - None
        """
        if verbose > 1:
            fpf_hypervolume_str = f'and Hypervolume {round(self.fpf_hypervolume, 2)}/{int(self.fpf_hypervolume_reference)}' if self.fpf_hypervolume is not None else ''

            print()
            print(
                f"Average complexity of {round(self.average_complexity,1)} and 1PF of length {len(self.first_pareto_front)} {fpf_hypervolume_str}\n")
        if verbose > 2:
            print(f"\tBest individual(s) in the first Pareto Front")
            for index, p in enumerate(self.first_pareto_front):
                print(f'{index})\t{p.program}')
                print()
                print(f'\tTrain fitness')
                print(f'\t{p.fitness}')
                if hasattr(p, 'fitness_validation') and len(p.fitness_validation) > 0:
                    print(f'\tValidation fitness')
                    print(f'\t{p.fitness_validation}')
                print()

    def load_population_from_summary(self, summary: pd.DataFrame, inplace=True):
        """
        This method loads the population from a summary DataFrame

        Args:
            - summary: pd.DataFrame
                The summary DataFrame from which to load the population
            - inplace: bool (default True)
                Whether to load the population in place or to return a new instance
        Returns:
            - None
        """

        new_population: List = list()

        for index, row in summary.iterrows():
            p = Program(operations=self.operations, features=self.features).from_string(
                row['Info Program'])
            p.fitness = {k.replace("Train ", ""): v for k,
                         v in row.items() if k.startswith('Train ')}
            p.fitness_validation = {
                k.replace("Validation ", ""): v for k, v in row.items() if k.startswith('Validation ')}

            new_population.append(p)

        if inplace:
            self.population = new_population
            return self.population
        
        return new_population

    def _select_final_population_NSGAII(self):
        """
        This method selects the final population that should propagate to the next generation.  
        It selects based on rank until it risks saturating population size. Afterward, it recursively remove from the list
        the worst elements of the reamining front based on secondary selection criteria. 

        Args:
            - None

        Returns:
            - survivors: List[Program]
                The list of programs that will propagate to the next generation

        """
        survivors = []

        rank_iter = 1
        pareto_front: List[Program] = self.extract_pareto_front(
            population=self.population, rank=rank_iter)

        
        while (len(survivors)+len(pareto_front) <= self.population_size) and pareto_front:
            survivors += pareto_front
            rank_iter += 1
            pareto_front = self.extract_pareto_front(
                population=self.population, rank=rank_iter)
            
            

        if not pareto_front or (len(survivors) == self.population_size):
            
            return survivors

        else:
            self._crowding_distance(
                population=self.population, rank_iter=rank_iter)

            pareto_front.sort(
                key=lambda p: p.crowding_distance, reverse=True)

            survivors += pareto_front[:self.population_size-len(survivors)]
            
           

        

        for rank in range(1, rank_iter + 1):
            if self.verbose > 1:
                print(f"\tSelecting Final Population: computing crowding distance {rank/(rank_iter):.1%}", end="\r")
            
            self._crowding_distance(
                population=survivors, rank_iter=rank)

        if self.verbose > 1:
            print(f"\tSelecting Final Population: computing crowding distance {rank/(rank_iter):.1%}. Completed!", end="\n")
        
        return survivors

    def _select_final_population_SMS_EMOEA(self):
        """

        Args:
            - None

        Returns:
            - survivors: List[Program]
                The list of programs that will propagate to the next generation
        """

        _ = self._compute_nadir_point(population=self.population)

        survivors = []

        rank_iter = 1
        pareto_front: List[Program] = self.extract_pareto_front(
            population=self.population, rank=rank_iter)

        
        while (len(survivors)+len(pareto_front) <= self.population_size) and pareto_front:
            survivors += pareto_front
            rank_iter += 1
            pareto_front = self.extract_pareto_front(
                population=self.population, rank=rank_iter)
            
            
            

        if not pareto_front or (len(survivors) == self.population_size):         

            return survivors

        else:
            
            while len(survivors) + len(pareto_front) > self.population_size:
                self._exclusive_hypervolume(
                    population=pareto_front, rank=rank_iter)

                pareto_front.sort(
                    key=lambda p: p.exclusive_hypervolume, reverse=False)

                pareto_front.pop(0)
                

            survivors += pareto_front

            
            

        

        for rank in range(1, rank_iter + 1):
            if self.verbose > 1:
                print(f"\tSelecting Final Population: computing hypervolume {rank/(rank_iter):.1%}", end="\r")
            self._exclusive_hypervolume(population=survivors, rank=rank)

        if self.verbose > 1:
            print(f"\tSelecting Final Population: computing hypervolume {rank/(rank_iter):.1%}. Completed!", end="\n")
            
        return survivors

    def spearman_diversity(self, data: Union[dict, pd.Series, pd.DataFrame]) -> float:
        """
        This method computes the diversity of the first pareto front using the spearman correlation

        Args:
            - None

        Returns:
            - diversity: float
                The spearman diversity of the first pareto front

        """

        if len(self.first_pareto_front) == 1:
            self.fpf_spearman_diversity = 0
        else:
            diversities = list()
            for index, program in enumerate(self.first_pareto_front):
                for other_program in self.first_pareto_front[index + 1:]:
                    stats, _ = spearmanr(program.evaluate(
                        data), other_program.evaluate(data))
                    diversities.append(stats)

            self.fpf_spearman_diversity = np.mean(1 - np.array(diversities))

        self.fpf_stats.loc[self.generation, 'n_individuals'] = len(
            self.first_pareto_front)
        self.fpf_stats.loc[self.generation,
                           'fpf_spearman_diversity'] = self.fpf_spearman_diversity

        return self.fpf_spearman_diversity

    @property
    def summary(self):
        """ Summary of the evolutionary process

        This method returns a summary of the evolutionary process.

        Args:
            - None
        Returns:
            - summary: pd.DataFrame
        """
        istances = list()

        for index, p in enumerate(self.population):
            p: 'Program'
            row = dict()
            row["Info Index"] = index + 1
            row["Info Program"] = str(p.program)
            row["Info Complexity"] = p.complexity
            row["Info Rank"] = p.rank

            columns = []
            for f_k, f_v in p.fitness.items():
                row[f"Train {f_k}"] = f_v

            if hasattr(p, 'fitness_validation'):
                for f_k, f_v in p.fitness_validation.items():
                    row[f"Validation {f_k}"] = f_v

            istances.append(row)

        df = pd.DataFrame(istances)
        return df

    @property
    def _total_time(self) -> float:
        if not 'time_generation_total' in self.times.columns:
            return 0
        tot = self.times['time_generation_total'].sum()
        if pd.isna(tot):
            return 0
        return tot

    def tree_diversity(self) -> float:
        """
        This method computes the tree diversity of the current population

        The tree diversity is computed as the average diversity of
        programs in the population. The diversity is computed as the
        1 - the similarity between two programs. The similarity is
        computed as the number of common sub-trees between two programs
        divided by the total number of sub-trees in the two programs.

        Args:
            - None

        Returns:
            - tree_diversity: float
                The tree diversity of the current population
        """

        if len(self.first_pareto_front) == 1:
            self.fpf_tree_diversity = 0
        else:
            diversities = list()
            for index, program in enumerate(self.first_pareto_front):
                for other_program in self.first_pareto_front[index + 1:]:
                    diversities.append(program.similarity(other_program))

            self.fpf_tree_diversity = np.mean(1 - np.array(diversities))

        self.fpf_stats.loc[self.generation, 'n_individuals'] = len(
            self.first_pareto_front)
        self.fpf_stats.loc[self.generation,
                           'fpf_tree_diversity'] = self.fpf_tree_diversity

        return self.fpf_tree_diversity

def compress(object):
    serialized_data = pickle.dumps(object)
    compressed_data = zlib.compress(serialized_data)
    return compressed_data

def decompress(compressed_data):
    serialized_data = zlib.decompress(compressed_data)
    object = pickle.loads(serialized_data)
    return object
