import copy
import logging
import random
from typing import Callable, Dict, List, Tuple, Union


import numpy as np
import pandas as pd
import sympy
from joblib import Parallel, delayed
from pytexit import py2tex
import traceback
import sympy

from symbolic_regression.multiobjective.fitness.Base import BaseFitness
from symbolic_regression.multiobjective.hypervolume import _HyperVolume
from symbolic_regression.multiobjective.optimization import (ADAM, ADAM2FOLD,
                                                             SCIPY, SGD,GaussProcess)
from symbolic_regression.Node import (FeatureNode, InvalidNode, Node,
                                      OperationNode)
from symbolic_regression.operators import (OPERATOR_ADD, OPERATOR_MUL,
                                           OPERATOR_POW)

def timeout_handler(signum, frame):
    raise TimeoutError("Simplify operation timed out")


class Program:
    """ A program is a tree that represent an arithmetic formula

    The nodes are the operations, each of which has an arbitrary number of operands
    (usually 1 or 2). An operand can be another operation, or a terminal node.
    A terminal node in a feature taken from the training dataset or a numerical
    constant.

    The generation of a tree follow a genetic approach in which, starting from a root
    node, the choice of which type of node will be the operand is modulated by a random
    distribution. Based on how likely the next node will be an operation or a terminal
    node, the tree will become deeper and so the formula more complex.
    The choice of the terminal node is determined by a random distribution as well to
    modulate how likely a random feature from the dataset will be chosen instead of a
    random numerical constant.

    The program can be evaluated and also printed.
    The evaluation execute recursively the defined operation using the values
    provided by the dataset. In the same way, printing the tree will recursively print
    each operation and terminal node based on its formatting pattern or numerical value.

    According to the genetic approach, the tree can perform a mutation or a cross-over.
    A mutation is a modification of a random inner node of the tree with a newly generated
    subtree. The cross-over is the replacement of a random subtree of this program with
    a random subtree of another given program.

    The program need to be provided with a set of possible operations, and a set of
    possible terminal node features from which to choose. Also, if numerical constant are
    desired, also the range from which to choose the constant must be provided.

    """

    def __init__(self, 
                 operations: List[Dict], 
                 features: List[str], 
                 feature_assumption: dict = None,
                 feature_probability: List = [], 
                 default_feature_probability: List = [], 
                 bool_update_feature_probability: bool = False,
                 const_range: Tuple = (0, 1), 
                 program: Node = None, 
                 parsimony: float = .8, 
                 parsimony_decay: float = .85) -> None:
        """


        Args:
            - operations: List[Dict]
                List of possible operations. Each operation is a dictionary with the following keys:
                    - func: callable
                        The function that will be executed by the operation
                    - arity: int
                        The number of operands of the operation
                    - format_str: str
                        The format pattern of the operation when printed in a string formula
                    - format_tf: str
                        The format pattern of the operation when printed in a tensorflow formula
                    - symbol: str
                        The symbol of the operation

            - features: List[str]
                List of possible features from which to choose the terminal nodes

            - const_range: Tuple  (default: (0, 1))
                Range from which to choose the numerical constants.

            - program: Node  (default: None)
                The root node of the tree. If None, a new tree will be generated.

            - parsimony: float  (default: .8)
                The parsimony coefficient. It is used to modulate the depth of the program.
                Use values between 0 and 1. The higher the value, the deeper the program.

            - parsimony_decay: float  (default: .85)
                The decay of the parsimony coefficient. It is used to modulate the depth of the program.
                Use values between 0 and 1. The higher the value, the deeper the program.

        """

        self.operations: List[Dict] = operations
        self.features: List[str] = features
        self.feature_assumption = feature_assumption
        self.feature_probability: List = feature_probability
        self.default_feature_probability: List = default_feature_probability
        self.bool_update_feature_probability: bool = bool_update_feature_probability
        self.const_range: Tuple = const_range
        self._constants: List = list()
        self.converged: bool = False

        # Operational attributes
        self._override_is_valid: bool = True
        self._is_duplicated: bool = False
        self._program_depth: int = 0
        self._complexity: int = 0
        self._exclusive_hypervolume: float = np.nan

        # Pareto Front Attributes
        self.rank: int = np.inf
        self.programs_dominates: List[Program] = list()
        self.programs_dominated_by: List[Program] = list()
        self.crowding_distance: float = 0
        self.program_hypervolume: float = np.nan
        self._hash: List[int] = None

        self.parsimony: float = parsimony
        self._parsimony_bkp: float = parsimony
        self.parsimony_decay: float = parsimony_decay

        self.is_logistic: bool = False
        self.is_affine: bool = False
        self.already_brgflow: bool = False
        self.already_bootstrapped: bool = False

        if program:
            self.program: Node = program
        else:
            self.program: Node = InvalidNode()
            self.fitness: Dict[str, float] = dict()
            self.fitness_validation: Dict[str, float] = dict()
            self.fitness_functions: List[BaseFitness] = list()
            self.is_fitness_to_minimize: Dict = dict()

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))

        result.programs_dominated_by = list()
        result.programs_dominates = list()

        result.program_hypervolume = np.nan
        result._override_is_valid = True
        return result

    def __lt__(self, other: 'Program') -> bool:
        """
        Overload of the less than operator. It is used to compare two programs.

        A program is less than another if it has a lower or equal rank and a higher or equal crowding distance.

        Args:
            - other: Program
                The other program to compare with

        Returns:
            - bool
                True if the program is less than the other program, False otherwise
        """

        rank_dominance = self.rank <= other.rank
        crowding_distance_dominance = self.crowding_distance >= other.crowding_distance

        return rank_dominance and crowding_distance_dominance

    def __len__(self) -> int:
        return self.complexity

    @property
    def complexity(self) -> int:
        """ The complexity of a program is the number of nodes (OperationNodes or FeatureNodes)
        """
        return self._complexity

    @complexity.getter
    def complexity(self, base_complexity=0):
        return self.program._get_complexity(base_complexity)

    @property
    def program_depth(self):
        """ The depth of a program is the length of the deepest branch
        """
        return self._program_depth

    @program_depth.getter
    def program_depth(self, base_depth=0):
        return self.program._get_depth(base_depth)

    @property
    def operations_used(self):
        """ This allow to get a list of all unique operations used in a program
        """
        return self._operations_used

    @operations_used.getter
    def operations_used(self):
        return self.program._get_operations(base_operations_used={})

    @property
    def all_operations(self):
        """ This allow to get a list of all the operations used in a program
        """
        return self._all_operations

    @all_operations.getter
    def all_operations(self):
        return self.program._get_all_operations(all_operations=[])

    @property
    def exclusive_hypervolume(self):
        """ This allow to get the exclusive hypervolume of the program
        """
        return self._exclusive_hypervolume

    @exclusive_hypervolume.setter
    def exclusive_hypervolume(self, value):
        self._exclusive_hypervolume = value

    @property
    def features_used(self):
        """ This allow to get all the unique features used in the tree
        """
        return self._features_used

    @features_used.getter
    def features_used(self):
        return self.program._get_features(features_list=[])

    def _internal_bootstrap(self, data: Union[dict, pd.Series, pd.DataFrame], target: str, weights: str, constants_optimization: str, constants_optimization_conf: dict, frac: float = .6) -> 'Program':
        bs_data = data.sample(frac=frac, replace=True)
        # !!! Weights should now be adapted to the bs_data target distribution

        # Increase the epochs to be used for the bootstrap
        constants_optimization_conf = copy.deepcopy(
            constants_optimization_conf)
        constants_optimization_conf['epochs'] = max(
            200, constants_optimization_conf.get('epochs', 100))

        recalibrated = copy.deepcopy(self)
        recalibrated.set_constants(
            new=list(np.random.uniform(
                low=self.const_range[0],
                high=self.const_range[1],
                size=len(self.get_constants())
            ))
        )
        return recalibrated.optimize(
            data=bs_data,
            target=target,
            weights=weights,
            constants_optimization=constants_optimization,
            constants_optimization_conf=constants_optimization_conf,
            inplace=False
        ).get_constants(return_objects=False)

    def bootstrap(self, data: Union[dict, pd.Series, pd.DataFrame], target: str, weights: str, constants_optimization: str, constants_optimization_conf: dict, inplace: bool = False, k: int = 1000, frac: float = .6) -> 'Program':
        """ This method allow to bootstrap a program.

        The bootstrapping is a statistical method to estimate the uncertainty of a model.
        We optimize the constants k times over a random fraction frac of the dataset.
        The result is a list of k constants for each constant of the program. These are
        then used to create a new program with the same structure of the original one
        but with the new constants. It leverages the confidence intervals capability of
        FeatureNode to receive a list of values for each constant.

        Args:
            - data: Union[dict, pd.Series, pd.DataFrame]
                The data on which the program will be evaluated
            - target: str
                The target variable of the data
            - weights: str
                The weights of the data
            - constants_optimization: str
                The constants optimization method to use
            - constants_optimization_conf: dict
                The configuration of the constants optimization method
            - inplace: bool  (default=False)
                If True, the method will modify the program in place, otherwise it will return a new program
            - k: int  (default=1000)
                The number of bootstrap iterations
            - frac: float  (default=.6)
                The fraction of the data to use for each bootstrap iteration
        """

        n_constants = len(self.get_constants(return_objects=False))
        n_features_used = len(self.features_used)
        if not isinstance(self.program, FeatureNode) and n_constants > 0 and n_features_used > 0:
            if inplace:
                return self
            else:
                return copy.deepcopy(self)

        bootstrapped_constants: List[float] = Parallel(n_jobs=-1)(delayed(self._internal_bootstrap)(
            data=data,
            target=target,
            weights=weights,
            constants_optimization=constants_optimization,
            constants_optimization_conf=constants_optimization_conf,
            frac=frac
        ) for _ in range(k))

        if inplace:
            bootstrapped_program = self
        else:
            bootstrapped_program = copy.deepcopy(self)

        bootstrapped_program.set_constants(
            [list(new_constants) for new_constants in zip(*bootstrapped_constants)])

        return bootstrapped_program

    def constants_confidence_intervals_overlap(self, other: 'Program') -> bool:

        if not len(self.get_constants(return_objects=False)) == len(other.get_constants(return_objects=False)):
            raise ValueError(
                "The two programs have different number of constants")

        for self_constant, other_constant in zip(self.get_constants(return_objects=True), other.get_constants(return_objects=True)):
            if self_constant.feature_confidence_intervals == [np.nan, np.nan] or other_constant.feature_confidence_intervals == [np.nan, np.nan]:
                return False
            overlap = max(
                0,
                min(self_constant.feature_confidence_intervals[1], other_constant.feature_confidence_intervals[1]) -
                max(self_constant.feature_confidence_intervals[0],
                    other_constant.feature_confidence_intervals[0])
            )
            if overlap <= 1e-7:
                return False

        return True

    def compute_fitness(self, fitness_functions: List[BaseFitness], data: Union[dict, pd.Series, pd.DataFrame], validation: bool = False, validation_federated: bool = False, simplify: bool = True) -> None:
        """ This function evaluate the fitness of the program on the given data.
        The data can be a dictionary, a pandas Series or a pandas DataFrame.

        Args:
            - fitness_functions: List[BaseFitness]
                The fitness functions to evaluate
            - data: Union[dict, pd.Series, pd.DataFrame]
                The data on which the program will be evaluated
            - validation: bool  (default: False)
                If True, the fitness will be computed on the validation data without optimization
            - validation_federated: bool  (default: False)
                If True, the fitness will be computed on the training and validation data without optimization (only for federated training)
            - simplify: bool  (default: True)
                If True, the program will be simplified before computing the fitness

        Returns:
            - None
        """

        # We store the fitness functions in the program object
        # as we need them in other parts of the code (e.g. in the hypervolume computation)
        self.fitness_functions: List[BaseFitness] = fitness_functions
        for ftn in self.fitness_functions:
            for ftn2 in self.fitness_functions:
                if ftn.label == ftn2.label and ftn != ftn2:
                    raise ValueError(
                        f"Fitness function with label {ftn.label} already used")

        if validation or validation_federated:
            simplify = False
            self.fitness_validation: Dict[str, float] = dict()
        else:
            self.fitness: Dict[str, float] = dict()

        self.is_fitness_to_minimize: Dict[str, bool] = dict()

        for ftn in self.fitness_functions:
            self.is_fitness_to_minimize[ftn.label] = ftn.minimize

        if simplify:
            try:
                self.simplify(inplace=True)
            except ValueError:
                self._override_is_valid = False
                self.fitness = {
                    ftn.label: np.inf for ftn in self.fitness_functions}
                return

        _converged: List[bool] = list()
        _export = {}

        for ftn in self.fitness_functions:
            try:
                """ We don't want optimization of the constants in the validation stage, both local (validation)
                and federated (validation_federated)
                """
                if ftn.export is True:
                    fitness_value, _this_export = ftn.evaluate(
                        program=self, data=data, validation=validation or validation_federated, inject=_export)

                    _export = {**_export, **_this_export}

                else:
                    fitness_value = ftn.evaluate(
                        program=self, data=data, pred=_export.get('pred'), validation=validation or validation_federated, inject=_export)

                fitness_value = round(fitness_value, 5) if not pd.isna(
                    fitness_value) else fitness_value

            except Exception:
                import traceback
                print(traceback.format_exc())
                fitness_value = np.inf
                self._override_is_valid = False

            if pd.isna(fitness_value) or (fitness_value==np.inf): 
                fitness_value = np.inf
                self._override_is_valid = False

            if validation:
                self.fitness_validation[ftn.label] = fitness_value
            else:
                self.fitness[ftn.label] = fitness_value

                if ftn.minimize and isinstance(ftn.convergence_threshold, (int, float)):
                    if fitness_value <= ftn.convergence_threshold:
                        _converged.append(True)
                    else:
                        _converged.append(False)

                    # Only if all the fitness functions have converged, then the program has converged
                    self.converged = all(_converged) if len(
                        _converged) > 0 else False

        if not validation:
            self._compute_hypervolume()

    def _compute_hypervolume(self) -> float:
        """
        This method return the hypervolume of the program

        The hypervolume is the volume occupied by the fitness space by the program.
        It can be of any dimension. We allow to compute the hypervolume only if the
        fitness functions are set to be minimized, otherwise we assume that the fitness
        are computed only for comparison purposes and not for optimization.

        Args:
            - None

        Returns:
            - float
                The hypervolume of the program.
        """

        if not self.program.is_valid:
            return np.nan

        fitness_to_hypervolume: List[BaseFitness] = list()
        for fitness in self.fitness_functions:
            if fitness.hypervolume_reference and fitness.minimize:
                fitness_to_hypervolume.append(fitness)

        if not fitness_to_hypervolume:
            return np.nan

        points = [np.array([self.fitness[ftn.label]
                            for ftn in fitness_to_hypervolume])]
        references = np.array(
            [ftn.hypervolume_reference for ftn in fitness_to_hypervolume])

        self.program_hypervolume = _HyperVolume(references).compute(points)

    def from_string(self, string: str) -> None:
        """ This function allow to create a program from a string.

        The string must be a valid mathematical formula that can be parsed by the sympy library.

        Args:
            - string: str
                The string representing the mathematical formula

        Returns:
            - Program
        """

        self.program: 'Node' = self.simplify(
            inject=string, inplace=False).program

        return self

    def predict(self, data: Union[dict, pd.Series, pd.DataFrame], logistic: bool = False, threshold: float = None) -> Union[int, float]:
        """ This function predict the value of the program on the given data.
        The data can be a dictionary, a pandas Series or a pandas DataFrame.

        Args:
            - data: dict, pd.Series, pd.DataFrame
                The data on which the program will be evaluated
            - logistic: bool  (default: False)
                If True, the program will be evaluated using the logistic function
            - threshold: float  (default: None)
                The threshold to use for the logistic function


        Returns:
            - int, float
                The result of the prediction
        """
        return self.evaluate(data=data, logistic=logistic, threshold=threshold)

    def predict_proba(self, data: Union[dict, pd.Series, pd.DataFrame]) -> Union[int, float]:
        """ This function predict the value of the program on the given data.
        The data can be a dictionary, a pandas Series or a pandas DataFrame.

        Args:
            - data: dict, pd.Series, pd.DataFrame
                The data on which the program will be evaluated

        Returns:
            - int, float
                The result of the prediction
        """
        evaluated = pd.DataFrame()
        evaluated['proba_1'] = self.to_logistic().evaluate(data=data, logistic=True)
        evaluated['proba_0'] = 1 - evaluated['proba_1']

        return evaluated[['proba_0', 'proba_1']]

    def evaluate(self, data: Union[dict, pd.Series, pd.DataFrame], logistic: bool = False, threshold: float = None) -> Union[int, float]:
        """ This function evaluate the program on the given data.
        The data can be a dictionary, a pandas Series or a pandas DataFrame.

        Args:
            - data: dict, pd.Series, pd.DataFrame
                The data on which the program will be evaluated
            - logistic: bool  (default: False)
                If True, the program will be evaluated using the logistic function

        Returns:
            - int, float
                The result of the evaluation
        """
        if not self.is_valid:
            return np.nan

        if logistic:
            if isinstance(threshold, float) and 0 <= threshold <= 1:
                return np.where(self.to_logistic(inplace=False).evaluate(data=data) > threshold, 1, 0)

            return self.to_logistic(inplace=False).evaluate(data=data)

        return self.program.evaluate(data=data)

    def _generate_tree(self, depth=-1, 
                       parsimony: float = .8, parsimony_decay: float = .85, father: Union[Node, None] = None, force_constant: bool = False, count:int=0) -> Node:
        """ This function generate a tree of a given depth.

        Args:
            - depth: int  (default=-1)
                The depth of the tree to generate. If -1, the depth will be randomly generated

            - parsimony: float  (default=.8)
                The parsimony coefficient. This modulates the depth of the generated tree.
                Use values between 0 and 1; the closer to 1, the deeper the tree will be.

            - parsimony_decay: float  (default=.85)
                The parsimony decay coefficient. This value is multiplied to the parsimony coefficient
                at each depth level. Use values between 0 and 1; the closer to 1, the quicker the
                parsimony coefficient will decay and therefore the shallower the tree will be.
                Use a lower value to prevent the tree from exploding and reaching a RecursionError.

            - father: Node  (default=None)
                The father of the node to generate. If None, the node will be the root of the tree.

            - force_constant: bool  (default=False)
                If True, the next node will be a constant node.

        Returns:
            - Node
                The generated tree. It's a recursive process and the returned node is the root of the tree.
        """
        def gen_operation(operation_conf: Dict, father: Union[Node, None] = None):
            return OperationNode(operation=operation_conf['func'],
                                 arity=operation_conf['arity'],
                                 format_str=operation_conf.get('format_str'),
                                 format_tf=operation_conf.get('format_tf'),
                                 format_result=operation_conf.get('format_result'),
                                 symbol=operation_conf.get('symbol'),
                                 format_diff=operation_conf.get(
                                     'format_diff', operation_conf.get('format_str')),
                                 father=father)

        def gen_feature(feature: str, father: Union[Node, None], is_constant: bool = False):
            return FeatureNode(feature=feature, father=father, is_constant=is_constant)

        self._override_is_valid = True

        # We can either pass dedicated parsimony and parsimony_decay values or use the ones
        # defined in the class
        if not parsimony:
            parsimony = self.parsimony
        if not parsimony_decay:
            parsimony_decay = self.parsimony_decay

        selector_operation = random.random()
        if ((selector_operation < parsimony and depth == -1) or (depth > 0)) and not force_constant:
            # We generate a random operation
            ##TO BE REMOVED
            #count+=1
            #print(count)
            operation = random.choice(self.operations)
            node = gen_operation(operation_conf=operation, father=father)

            new_depth = -1 if depth == -1 else depth - 1

            for i in range(node.arity):
                if operation == OPERATOR_POW and i == 1:
                    # In case of the power operator, the second operand must be a constant
                    # We do not want to generate a tree for the second operand otherwise it may
                    # generate an unrealistic mathematical model
                    force_constant = True

                node.add_operand(
                    self._generate_tree(depth=new_depth,
                                        father=node,
                                        parsimony=parsimony * parsimony_decay,
                                        parsimony_decay=parsimony_decay,
                                        force_constant=force_constant,count=count))
            force_constant = False

        else:
            # We generate a random feature

            # The probability to get a feature from the training data is
            # (n-1) / n where n is the number of features.
            # Otherwise a constant value will be generated.

            selector_feature = random.random()
            # To prevent constant features to be underrepresented, we set a threshold
            threshold = max(.1, (1 / len(self.features)))
            threshold = min(.3, threshold)

            if selector_feature > threshold and not force_constant:
                # A Feature from the dataset
                feature = np.random.choice(self.features,p=np.array(self.feature_probability))

                node = gen_feature(
                    feature=feature, father=father, is_constant=False)

            else:
                # Generate a constant

                feature = random.uniform(
                    self.const_range[0], self.const_range[1])

                # Arbitrary rounding of the generated constant

                node = gen_feature(
                    feature=feature, father=father, is_constant=True)

        return node

    def get_constants(self, return_objects: bool = False):
        """
        This method allow to get all constants used in a tree.

        The constants are used for the neuronal-based constants optimizer; it requires
        all constants to be in a fixed order explored by a DFS descent of the tree.

        Args:
            - return_objects: bool  (default=False)
                If True, the method will return a list of FeatureNode objects instead of a list of
                feature names.

        Returns:
            - List[FeatureNode]
                A list of FeatureNode objects representing the constants used in the tree.
        """
        to_return = None
        if isinstance(self.program, OperationNode):
            to_return = self.program._get_constants(const_list=[])

        # Only one constant FeatureNode
        elif self.program.is_constant:
            to_return = [self.program]

        else:
            # Only one non-constant FeatureNode
            to_return = list()

        for index, constant in enumerate(to_return):
            self._set_constants_index(constant=constant, index=index)

        if not return_objects:
            to_return = [f.feature for f in to_return]

        return to_return

    def get_features(self, return_objects: bool = False):
        """
        This method recursively explore the tree and return a list of unique features used.

        Args:
            - return_objects: bool  (default=False)
                If True, the method will return a list of FeatureNode objects instead of a list of
                feature names.

        Returns:
            - List[str] or List[FeatureNode]
                A list of unique features used in the tree.
        """
        if isinstance(self.program, OperationNode):
            return self.program._get_features(features_list=[],
                                              return_objects=return_objects)

        # Only one non-constant FeatureNode
        elif not self.program.is_constant:
            return [self.program]

        # Case for programs of only one constant FeatureNode.
        # Use get_constants() to have a list of all constant FeatureNode objects
        return []

    @property
    def _has_incomplete_fitness(self):
        """
        This method return True if the program has an incomplete fitness.
        An incomplete fitness cannot be interpreted as invalid program because
        initially all programs have an incomplete fitness. We need a dedicated
        method to check if the fitness is incomplete.

        Args:
            - None

        Returns:
            - bool
                True if the program has an incomplete fitness.
        """
        return len(self.fitness) != len(self.fitness_functions)

    @property
    def hash(self):
        """ This method return the hash of the program

        The hash is a list of unique ideantifiers of the nodes of the tree.
        It is used to compare two programs.

        Args:
            - None

        Returns:
            - List[int]
                A list of unique identifiers of the nodes of the tree.
        """
        if not self._hash:
            if isinstance(self.program, OperationNode):
                self._hash = sorted(self.program.hash(hash_list=[]))
            else:
                self._hash = [self.program.hash(hash_list=[])]

        return self._hash

    @property
    def has_valid_fitness(self) -> bool:
        """
        This method return True if the program has a valid fitness.

        Args:
            - None

        Returns:
            - bool
                True if the program has a valid fitness.
        """
        for label, value in self.fitness.items():
            if np.isnan(value) or np.isinf(value):
                return False
        return True

    def hypervolume(self) -> float:
        if not self.program_hypervolume:
            self._compute_hypervolume()

        return self.program_hypervolume

    def init_program(self) -> None:
        """
        This method initialize a new program calling the recursive generation function.

        The generation of a program follows a genetic algorithm in which the choice on how to
        progress in the generation randomly choose whether to put anothe operation (deepening
        the program) or to put a terminal node (a feature from the dataset or a constant)

        Args:
            - None

        Returns:
            - None
        """

        logging.debug(
            f'Generating a tree with parsimony={self.parsimony} and parsimony_decay={self.parsimony_decay}')

        # Father=None is used to identify the root node of the program
        self.program = self._generate_tree(
            father=None,
            parsimony=self.parsimony,
            parsimony_decay=self.parsimony_decay)
        
        #print(f'Tree generated: {self.program}')
        self.to_affine_default(inplace=True)

        self.parsimony = self._parsimony_bkp  # Restore parsimony for future operations
        
        logging.debug(f'Generated a program of depth {self.program_depth}')
        logging.debug(self.program)

        # Reset the hash to force the re-computation
        self._hash = None

    def is_duplicate(self, other: 'Program', delta_fitness: float = 0.01, drop_by_similarity: bool = False) -> bool:
        """ Determines whether two programs are equivalent based on equal fitnesses

        If the fitness of two programs are identical, we assume they are equivalent to each other.
        We round to the 5th decimal to state whether the fitness are equal.

        Args:
            - other: Program
                The other program to compare to

        Returns:
            - bool
                True if the two programs are equivalent, False otherwise.
        """
        
        if drop_by_similarity and self.similarity(other) > 0.99:  
            return True
        a_fit_min = {f.label: self.fitness[f.label] for f in self.fitness_functions if f.minimize == True}
        b_fit_min = {f.label: other.fitness[f.label] for f in other.fitness_functions if f.minimize == True}
        for (_, a_fit), (_, b_fit) in zip(a_fit_min.items(),
                                                      b_fit_min.items()):

            # One difference is enough for them not to be identical
            if abs(a_fit-b_fit) / abs(a_fit + 1e-8) > delta_fitness:
                return False
        return True

    def is_constant(self):
        """ This method return True if the program is a constant, False otherwise.
        """
        return isinstance(self.program, FeatureNode) and self.program.is_constant

    @property
    def is_valid(self) -> bool:
        """ This method return True if the program is valid, False otherwise.

        A program is valid if:
            - It is a valid tree
            - It has a valid fitness

        Returns:
            - bool (default=True)
                True if the program is valid, False otherwise.    
        """

        return self.program.is_valid and self._override_is_valid

    def lambdify(self) -> Callable:
        # Initialize symbols for variables and constants
        n_constants = len(self.get_constants(return_objects=True))

        x_sym = ''
        for f in self.features:
            x_sym += f'{f},'
        x_sym = sympy.symbols(x_sym, real=True)
        c_sym = sympy.symbols('c0:{}'.format(n_constants),real=True)

        p_sym = self.program.render(format_diff=True)
        try:
            pyf_prog = sympy.lambdify([x_sym, c_sym], p_sym)

        except KeyError:
            #print(p_sym,x_sym,c_sym)
            print('key error in lambdify')
            #traceback.print_exc() 
            return None
        except AttributeError:
            #print(p_sym,x_sym,c_sym)
            print('Attribtute error in lambdify')
            #traceback.print_exc() 
            return None
        except ImportError:
            print('ImportError error in lambdify')
            #traceback.print_exc() 
            return None
    
        return pyf_prog
    
    def optimize(self,
                 data: Union[dict, pd.Series, pd.DataFrame],
                 target: str,
                 weights: str,
                 constants_optimization: str,
                 constants_optimization_conf: dict,
                 bootstrap: bool = False,
                 inplace: bool = False) -> 'Program':
        """ This method allow to optimize the constants of a program.

        The optimization of constants consists of executing a gradient descent strategy on the constants
        based on the task of the training (classification or regression). The optimization is done using
        implementations of Stochastic Gradient Descent (SGD) and ADAM (both in 1D and 2D version).

        In the input dictionary constants_optimization_conf, the following parameters can be set:
            - learning_rate: float
                The learning rate
            - batch_size: int
                The batch size
            - epochs: int
                The number of epochs
            - gradient_clip: bool
                Whether to clip the gradients
            - beta_1: float
                The beta 1 parameter for ADAM
            - beta_2: float
                The beta 2 parameter for ADAM
            - epsilon: floatbins
                The l1 regularization parameter
            - l2_param: float
                The l2 regularization parameter

        Args:
            - data: Union[dict, pd.Series, pd.DataFrame]
                The data on which the program will be evaluated
            - target: str
                The target variable of the data
            - weights: str
                The weights of the data
            - constants_optimization: str
                The constants optimization method to use
            - constants_optimization_conf: dict
                The configuration of the constants optimization method
            - bootstrap: bool  (default=False)
                If True, the constants will be optimized using bootstrapping
            - inplace: bool  (default=False)
                If True, the method will modify the program in place, otherwise it will return a new program

        Returns:
            - Program
                The optimized program
        """
        to_optimize = self if inplace else copy.deepcopy(self)

        if not constants_optimization or not self.is_valid:
            return to_optimize

        task = constants_optimization_conf['task']
        n_constants = len(self.get_constants(return_objects=False))
        n_features_used = len(self.features_used)

        if task not in ['regression:wmse', 'regression:cox',  'regression:cox_efron', 'regression:finegray', 'binary:logistic', 'non-derivative:GP']:
            raise AttributeError(
                f'Task supported are regression:wmse/cox/cox-efron/finegray, binary:logistic or non-derivative:GP')

        if not isinstance(self.program, FeatureNode) and n_constants > 0 and n_features_used > 0:
            ''' Rationale for the conditions:

            not isinstance(program.program, FeatureNode)
                programs with only a FeatureNode are not acceptable anyway

            n_constants > 0
                as the optimization algorithm optimize only constants

            n_features_used > 0
                as it is a constant program anyway and the optimized won't work with this configuration
            '''

            if constants_optimization == 'SGD':
                f_opt = SGD
                #self.to_affine(data=data, target=target, inplace=True)

            elif constants_optimization == 'ADAM':
                f_opt = ADAM
                #self.to_affine(data=data, target=target, inplace=True)

            elif constants_optimization == 'ADAM2FOLD':
                # Here there can be more than one target so need the index
                f_opt = ADAM2FOLD
                #self.to_affine(data=data, target=target[0], inplace=True)

            elif constants_optimization == 'scipy':
                f_opt = SCIPY
                #self.to_affine(data=data, target=target, inplace=True)

            elif constants_optimization == 'GaussianProcess':
                assert task=='non-derivative:GP', 'Only IC strategy is implemented with GaussianProcess optimization'
                f_opt = GaussProcess
                # need to simplify here as this usually happens inside derivative objectives 
                # so here it wouldn't happen
                to_optimize.simplify(inplace=True)

            else:
                raise AttributeError(
                    f'Constants optimization method {constants_optimization} not supported')

            
            try:
                final_parameters, _, _ = f_opt(
                    program=to_optimize,
                    data=data,
                    target=target,
                    weights=weights,
                    constants_optimization_conf=constants_optimization_conf,
                    task=task,
                    bootstrap=bootstrap,
                )
            except NameError:
                return to_optimize
            except TimeoutError:
                return to_optimize
            
            if len(final_parameters) > 0:
                to_optimize.set_constants(new=final_parameters)

            return to_optimize

        return to_optimize

    def _select_random_node(self, root_node: Union[OperationNode, FeatureNode, InvalidNode], depth: float = .8, only_operations: bool = False) -> Union[OperationNode, FeatureNode]:
        """ This method return a random node of a sub-tree starting from root_node.

        Args:
            - root_node: OperationNode, FeatureNode, InvalidNode
                The root node of the sub-tree.
            - depth: float (default=.8)
                The depth to which select the node. The value must be between 0 and 1.
            - only_operations: bool (default=False)
                If True, the method will return only an OperationNode, otherwise it can return a FeatureNode as well.

        Returns:
            - OperationNode, FeatureNode
                The selected node.
        """

        if isinstance(root_node, InvalidNode):
            return None

        if isinstance(root_node, FeatureNode):
            if only_operations:
                return root_node.father
            return root_node

        if isinstance(root_node, OperationNode):

            if random.random() < depth:
                # Select a random operand
                operand = random.choice(root_node.operands)

                return self._select_random_node(root_node=operand, depth=depth * .9, only_operations=only_operations)
            else:
                return root_node

    def set_constants(self, new: List[float]) -> None:
        """ This method allow to overwrite the value of constants after the neuron-based optimization

        Args:
            - new: list
                The new values of the constants

        Returns:
            - None
        """
        for constant, new_value in zip(self.get_constants(return_objects=True), new):
            constant.feature = new_value

    def update_feature_probability(self, data: Union[dict, pd.Series, pd.DataFrame])-> List[float]:   
        find_config=False
        for f in self.fitness_functions:
            if (f.minimize==True) and (f.constants_optimization) and (f.constants_optimization_conf):
                target=f.target
                find_config=True

        if not find_config:
            return self.feature_probability
        else:
            try:
                y_pred=self.evaluate(data=data)
                y_true=data[target]
                diff=y_true-y_pred

                from sklearn.ensemble import RandomForestRegressor
                regr = RandomForestRegressor(max_depth=4, random_state=42)
                regr.fit(data[self.features], diff)
                scores=regr.feature_importances_
                scores=scores/np.sum(scores)
                scores = np.round(scores, decimals=4).astype(np.float64)
                id_max=np.argmax(scores)
                max_val=scores[id_max]
                scores[id_max]=1-(scores.sum()-max_val)
                return scores.tolist()
            except ValueError:
                return self.feature_probability
    
    @staticmethod
    def _set_constants_index(constant, index) -> None:
        """ This method allow to overwrite the index of a constant

        Args:
            - constant: ConstantNode
                The constant to modify

            - index: int
                The new index of the constant

        Returns:
            - None
        """
        constant.index = index

    def similarity(self, other: 'Program') -> float:
        """ This method return the similarity between two programs

        The similarity is computed as the number of common elements between the two programs
        divided by the total number of elements in the two programs.

        Args:
            - other: Program
                The other program to compare to

        Returns:
            - float
                The similarity between the two programs
        """
        def common_elements(list1, list2):
            result = []
            for element in list1:
                if element in list2:
                    result.append(element)
            return result

        c_elements = common_elements(self.hash, other.hash)

        return 2 * len(c_elements) / (len(self.hash) + len(other.hash))

    def simplify(self, inplace: bool = False, inject: Union[str, None] = None) -> 'Program':
        """ This method allow to simplify the structure of a program using a SymPy backend

        Args:
            - inplace: bool (default=False)
                If True, the program is simplified in place. If False, a new Program object is returned.
            - inject: Union[str, None] (default=None)
                If not None, the program is simplified using the inject string as a root node.

        Returns:
            - Program
                The simplified program
        """
        from symbolic_regression.simplification import parse_and_extract_operation
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

        # Function to perform simplification with timeout
        @timeout(10)
        def simplify_program(program: Union[Program, str]) -> Union[FeatureNode, OperationNode, InvalidNode]:
            """ This function simplify a program using a SymPy backend

            try: the root node of the program, not the Program object

            """
            try:
                if isinstance(program, Program) and isinstance(
                        program.program, FeatureNode):
                    return program.program

                logging.debug(f'Simplifying program {program}')

                #print('begin parse expr')
                
                try:
                                    
                    string = program.program.render() if isinstance(program, Program) else program
                    new_program = parse_and_extract_operation(string,simplify=True)
                    #print(f'COMPARING:\nprogram {string}\nsympy obj {sympy_obj}\nnumeric_eval {sympy.pretty(expr_numeric_evaluated)}\nsimplified {simplified}\nsimplified {new_program}')
                    #print(f'COMPARING:\nprogram {string}\nsimplified {new_program}')
                    


                except Exception as e:
                    print("other error in sympy.simplify inside simplify:", e)
                    print(f'program : {program.program}')
                    print(f'program render : {program.program.render()}')
                    #traceback.print_exc()  # if you want the full traceback
                    #program._override_is_valid = False
                    return program.program
                logging.debug(
                    f'Extracting the program tree from the simplified')
                #print('end parse expr. Begin clean div')

                logging.debug(f'Simplified program {new_program}')

                return new_program

            except UnboundLocalError:
                return program.program

        if self._hash:
            return self
        
        if inplace:
            to_return = self
        else:
            to_return = copy.deepcopy(self)

        try:
            if inject:
                simp = simplify_program(inject)
            else:
                simp = simplify_program(to_return)
        except TimeoutError:
            print('Simplification timedout after 10 seconds')
            return to_return
        
        to_return.program = simp
        if not simp:
            to_return._override_is_valid = False

        # Reset the hash to force the re-computation
        to_return._hash = None
    
        return to_return



    def to_affine(self, data: Union[dict, pd.Series, pd.DataFrame], target: str, inplace: bool = False) -> 'Program':
        """ This function create an affine version of the program between the target maximum and minimum

        The affine of a program is defined as:

        .. math::

            \\hat{y} = \\frac{y_{max} - y_{min}}{\\hat{y}_{max} - \\hat{y}_{min}} \\hat{y} + \\frac{y_{min} \\hat{y}_{max} - y_{max} \\hat{y}_{min}}{\\hat{y}_{max} - \\hat{y}_{min}}

        Where:

        - :math:`\\hat{y}` is the output of the program
        - :math:`y_{min}` and :math:`y_{max}` are the minimum and maximum of the target
        - :math:`\\hat{y}_{min}` and :math:`\\hat{y}_{max}` are the minimum and maximum of the output of the program

        Args:
            - data: Union[dict, pd.Series, pd.DataFrame]
                The data used to evaluate the program

            - target: str
                The target of the program

            - inplace: bool (default=False)
                If True, the program is simplified in place. If False, a new Program object is returned.

        Returns:
            - Program
                The affine program
        """

        if not self.is_valid:
            return self

        if inplace:
            prog = self
        else:
            prog = copy.deepcopy(self)

        y_pred = prog.evaluate(data=data)

        y_pred_min = min(y_pred)
        y_pred_max = max(y_pred)
        target_min = data[target].min()
        target_max = data[target].max()

        if y_pred_min == y_pred_max:
            return prog

        alpha = target_max + (target_max - target_min) * \
            y_pred_max / (y_pred_min - y_pred_max)
        beta = (target_max - target_min) / (y_pred_max - y_pred_min)

        if pd.isna(alpha) or pd.isna(beta):
            return prog

        add_node = OperationNode(
            operation=OPERATOR_ADD['func'],
            arity=OPERATOR_ADD['arity'],
            format_str=OPERATOR_ADD['format_str'],
            format_tf=OPERATOR_ADD.get('format_tf'),
            format_result=OPERATOR_ADD.get('format_result'),
            symbol=OPERATOR_ADD.get('symbol'),
            format_diff=OPERATOR_ADD.get(
                'format_diff', OPERATOR_ADD['format_str']),
            father=None
        )
        add_node.add_operand(FeatureNode(
            feature=alpha, father=add_node, is_constant=True))

        mul_node = OperationNode(
            operation=OPERATOR_MUL['func'],
            arity=OPERATOR_MUL['arity'],
            format_str=OPERATOR_MUL['format_str'],
            format_tf=OPERATOR_MUL.get('format_tf'),
            format_result=OPERATOR_MUL.get('format_result'),
            symbol=OPERATOR_MUL.get('symbol'),
            format_diff=OPERATOR_MUL.get(
                'format_diff', OPERATOR_MUL['format_str']),
            father=add_node
        )

        mul_node.add_operand(FeatureNode(
            feature=beta, father=mul_node, is_constant=True))

        prog.program.father = mul_node
        mul_node.add_operand(prog.program)
        add_node.add_operand(mul_node)

        prog.program = add_node
        prog.is_affine = True

        # Reset the hash to force the re-computation
        self._hash = None
        return prog
    
    def to_affine_default(self, inplace: bool = False) -> 'Program':
        """ 
        Args:

            - inplace: bool (default=False)
                If True, the program is simplified in place. If False, a new Program object is returned.

        Returns:
            - Program
                The affine program
        """

        if not self.is_valid:
            return self

        if inplace:
            prog = self
        else:
            prog = copy.deepcopy(self)

        

        alpha = 1e-3
        beta = 1+1e-4

        if pd.isna(alpha) or pd.isna(beta):
            return prog

        add_node = OperationNode(
            operation=OPERATOR_ADD['func'],
            arity=OPERATOR_ADD['arity'],
            format_str=OPERATOR_ADD['format_str'],
            format_tf=OPERATOR_ADD.get('format_tf'),
            format_result=OPERATOR_ADD.get('format_result'),
            symbol=OPERATOR_ADD.get('symbol'),
            format_diff=OPERATOR_ADD.get(
                'format_diff', OPERATOR_ADD['format_str']),
            father=None
        )
        add_node.add_operand(FeatureNode(
            feature=alpha, father=add_node, is_constant=True))

        mul_node = OperationNode(
            operation=OPERATOR_MUL['func'],
            arity=OPERATOR_MUL['arity'],
            format_str=OPERATOR_MUL['format_str'],
            format_tf=OPERATOR_MUL.get('format_tf'),
            format_result=OPERATOR_MUL.get('format_result'),
            symbol=OPERATOR_MUL.get('symbol'),
            format_diff=OPERATOR_MUL.get(
                'format_diff', OPERATOR_MUL['format_str']),
            father=add_node
        )

        mul_node.add_operand(FeatureNode(
            feature=beta, father=mul_node, is_constant=True))

        prog.program.father = mul_node
        mul_node.add_operand(prog.program)
        add_node.add_operand(mul_node)

        prog.program = add_node
        prog.is_affine = True

        # Reset the hash to force the re-computation
        self._hash = None
        return prog

    def to_logistic(self, inplace: bool = False) -> 'Program':
        """ This function create a logistic version of the program

        The logistic of a program defined as the program between the sigmoid function.

        Args:
            - inplace: bool (default=False)
                If True, the program is simplified in place. If False, a new Program object is returned.

        Returns:
            - Program
                The logistic program
        """
        from symbolic_regression.operators import OPERATOR_SIGMOID
        logistic_node = OperationNode(
            operation=OPERATOR_SIGMOID['func'],
            arity=OPERATOR_SIGMOID['arity'],
            format_str=OPERATOR_SIGMOID['format_str'],
            format_tf=OPERATOR_SIGMOID.get('format_tf'),
            format_result=OPERATOR_SIGMOID.get('format_result'),
            symbol=OPERATOR_SIGMOID.get('symbol'),
            format_diff=OPERATOR_SIGMOID.get(
                'format_diff', OPERATOR_SIGMOID['format_str']),
            father=None
        )
        # So the upward pointer of the father is not permanent
        if inplace:
            program_to_logistic = self
        else:
            program_to_logistic = copy.deepcopy(self)

        logistic_node.operands.append(program_to_logistic.program)
        program_to_logistic.program.father = logistic_node
        program_to_logistic.program = logistic_node
        program_to_logistic.is_logistic = True

        # Reset the hash to force the re-computation
        self._hash = None

        return program_to_logistic

    def to_latex(self) -> str:
        """ This allow to print the program in LaTeX format

        Returns:
            A string representing the program in LaTeX format
        """
        return py2tex(str(self.program))

    def to_mathematica(self) -> str:
        """ This allow to print the program in Mathematica format

        Returns:
            A string representing the program in Mathematica format
        """
        return sympy.printing.mathematica.mathematica_code(self.program)

    # GENETIC OPERATIONS

    def cross_over(self, other: 'Program' = None, inplace: bool = False) -> None:
        """ This module perform a cross-over between this program and another from the population

        A cross-over is the switch between sub-trees from two different programs.
        The cut point are chosen randomly from both programs and the sub-tree from the second
        program (other) will replace the sub-tree from the current program.

        This is a modification only on the current program, so the other one will not be
        affected by this switch.

        It can be performed inplace, overwriting the current program, or returning a new program
        equivalent to the current one after the cross-over is applied.

        Args:
            - other: Program (default=None)
                The other program to cross-over with. If None, a new program is created and used.
            - inplace: bool (default=False)
                If True, the cross-over is performed in place. If False, a new Program object is returned.

        Returns:
            - Program
                The new program after the cross-over is applied
        """

        if not other:
            other = Program(operations=self.operations,
                            features=self.features,
                            feature_probability=self.default_feature_probability,
                            default_feature_probability=self.default_feature_probability,
                            bool_update_feature_probability=self.bool_update_feature_probability,
                            const_range=self.const_range,
                            program=self.program,
                            parsimony=self.parsimony,
                            parsimony_decay=self.parsimony_decay)
            other.init_program()


        if not isinstance(other, Program):
            raise TypeError(
                f'Can cross-over only using another Program object: {type(other)} provided'
            )

        if self.features != other.features:
            raise AttributeError(
                f'The two programs must have the same features set')

        if self.operations != other.operations:
            raise AttributeError(
                f'The two programs must have the same operations')

        offspring = copy.deepcopy(self.program)

        cross_over_point1 = self._select_random_node(root_node=offspring)

        cross_over_point2 = copy.deepcopy(
            self._select_random_node(root_node=other.program))

        if not cross_over_point1 or not cross_over_point2:
            new = Program(operations=self.operations,
                                features=self.features,
                                feature_probability=self.default_feature_probability,
                                default_feature_probability=self.default_feature_probability,
                                bool_update_feature_probability=self.bool_update_feature_probability,
                                const_range=self.const_range,
                                parsimony=self.parsimony,
                                parsimony_decay=self.parsimony_decay)
            new.init_program()
            if inplace:
                self.program = new
                self._hash = None
                self.already_bootstrapped = False
                self.already_brgflow = False
                return self
            return new

        cross_over_point2.father = cross_over_point1.father

        if cross_over_point1.father:
            cross_over_point1.father.operands[
                cross_over_point1.father.operands.index(
                    cross_over_point1)] = cross_over_point2
        else:
            offspring = cross_over_point2

        if inplace:
            self.program = offspring
            self._hash = None
            self.already_bootstrapped = False
            self.already_brgflow = False
            return self

        new = Program(program=offspring,
                      operations=self.operations,
                      features=self.features,
                      feature_probability=self.feature_probability,
                      default_feature_probability=self.default_feature_probability,
                      bool_update_feature_probability=self.bool_update_feature_probability,
                      const_range=self.const_range,
                      parsimony=self.parsimony,
                      parsimony_decay=self.parsimony_decay)

        return new

    def mutate(self, inplace: bool = False) -> 'Program':
        """ This method perform a mutation on a random node of the current program

        A mutation is a random generation of a new sub-tree replacing another random
        sub-tree from the current program.

        Args:
            - inplace: bool (default=False)
                If True, the mutation is performed in place. If False, a new Program object is returned.

        Returns:
            - Program
                The new program after the mutation is applied
        """

        if self.program_depth == 0:
            # Case in which a one FeatureNode only program is passed.
            # A new tree is generated.
            new = Program(operations=self.operations,
                          features=self.features,
                          feature_probability=self.default_feature_probability,
                          default_feature_probability=self.default_feature_probability,
                          bool_update_feature_probability=self.bool_update_feature_probability,
                          const_range=self.const_range,
                          parsimony=self.parsimony,
                          parsimony_decay=self.parsimony_decay)

            new.init_program()

            if inplace:
                self.program = new
                self._hash = None
                self.already_bootstrapped = False
                self.already_brgflow = False
                return self

            return new

        offspring = copy.deepcopy(self.program)

        mutate_point = self._select_random_node(root_node=offspring)

        if (not mutate_point) or not (mutate_point.father):
            new = Program(operations=self.operations,
                          features=self.features,
                          feature_probability=self.default_feature_probability,
                          default_feature_probability=self.default_feature_probability,
                          bool_update_feature_probability=self.bool_update_feature_probability,
                          const_range=self.const_range,
                          parsimony=self.parsimony,
                          parsimony_decay=self.parsimony_decay)

            new.init_program()

            if inplace:
                self.program = new
                self._hash = None
                self.already_bootstrapped = False
                self.already_brgflow = False
                return self

            return new

        mutated = self._generate_tree(father=mutate_point.father, 
                                      parsimony=self.parsimony, 
                                      parsimony_decay=self.parsimony_decay)

        mutate_point.father.operands[
            mutate_point.father.operands.index(mutate_point)] = mutated

        if inplace:
            self.program = offspring
            self._hash = None
            self.already_bootstrapped = False
            self.already_brgflow = False
            return self

        new = Program(program=offspring,
                      operations=self.operations,
                      features=self.features,
                      feature_probability=self.feature_probability,
                      default_feature_probability=self.default_feature_probability,
                      bool_update_feature_probability=self.bool_update_feature_probability,
                      const_range=self.const_range,
                      parsimony=self.parsimony,
                      parsimony_decay=self.parsimony_decay)

        return new

    def insert_node(self, inplace: bool = False) -> 'Program':
        """ This method allow to insert a OperationNode in a random spot in the program

        The insertion of a OperationNode must comply with the arity of the existing
        one and must link to the existing operands.

        Args:
            - inplace: bool (default=False)
                If True, the insertion is performed in place. If False, a new Program object is returned.

        Returns:
            - Program
                The new program after the insertion is applied
        """
        offspring = copy.deepcopy(self.program)
        mutate_point = self._select_random_node(root_node=offspring)

        if not mutate_point:
            new = Program(operations=self.operations,
                          features=self.features,
                          feature_probability=self.default_feature_probability,
                          default_feature_probability=self.default_feature_probability,
                          bool_update_feature_probability=self.bool_update_feature_probability,
                          const_range=self.const_range,
                          parsimony=self.parsimony,
                          parsimony_decay=self.parsimony_decay)

            new.init_program()

            if inplace:
                self.program = new
                self._hash = None
                self.already_bootstrapped = False
                self.already_brgflow = False
                return self

            return new

        new_node = self._generate_tree(father=mutate_point.father, depth=1)

        if mutate_point.father:  # Can be None if it is the root
            # Is a new tree of only one OperationNode
            mutate_point.father.operands[mutate_point.father.operands.index(
                mutate_point)] = new_node

        # Choose a random children to attach the previous mutate_point
        # The new_node is already a tree with depth = 1 so, in case of arity=2
        # operations the other operator is already set.
        new_node.operands[random.randint(0, new_node.arity - 1)] = mutate_point

        # If the new_node is also the new root, offspring need to be updated.
        if not mutate_point.father:
            offspring = new_node

        mutate_point.father = new_node

        if inplace:
            self.program = offspring
            self._hash = None
            self.already_bootstrapped = False
            self.already_brgflow = False
            return self

        new = Program(program=offspring,
                      operations=self.operations,
                      features=self.features,
                      feature_probability=self.feature_probability,
                      default_feature_probability=self.default_feature_probability,
                      bool_update_feature_probability=self.bool_update_feature_probability,
                      const_range=self.const_range,
                      parsimony=self.parsimony,
                      parsimony_decay=self.parsimony_decay)

        return new

    def delete_node(self, inplace: bool = False) -> 'Program':
        """ This method delete a random OperationNode from the program.

        It selects a random children of the deleted node to replace itself
        as child of its father.

        Args:
            - inplace: bool (default=False)
                If True, the deletion is performed in place. If False, a new Program object is returned.

        Returns:
            - Program
                The new program after the deletion is applied
        """
        offspring = copy.deepcopy(self.program)
        mutate_point = self._select_random_node(
            root_node=offspring, only_operations=True)

        if mutate_point:
            mutate_father = mutate_point.father
        else:  # When the mutate point is None, can happen when program is only a FeatureNode
            new = Program(operations=self.operations,
                        features=self.features,
                        feature_probability=self.default_feature_probability,
                        default_feature_probability=self.default_feature_probability,
                        bool_update_feature_probability=self.bool_update_feature_probability,
                        const_range=self.const_range,
                        parsimony=self.parsimony,
                        parsimony_decay=self.parsimony_decay)
            new.init_program()
            if inplace:
                self.program = new
                self._hash = None
                self.already_bootstrapped = False
                self.already_brgflow = False
                return self
            return new

        mutate_child = random.choice(mutate_point.operands)
        mutate_child.father = mutate_father

        if mutate_father:  # Can be None if it is the root
            mutate_father.operands[mutate_father.operands.index(
                mutate_point)] = mutate_child
        else:
            offspring = mutate_child

        if inplace:
            self.program = offspring
            self._hash = None
            self.already_bootstrapped = False
            self.already_brgflow = False
            return self

        new = Program(program=offspring,
                      operations=self.operations,
                      features=self.features,
                      feature_probability=self.feature_probability,
                      default_feature_probability=self.default_feature_probability,
                      bool_update_feature_probability=self.bool_update_feature_probability,
                      const_range=self.const_range,
                      parsimony=self.parsimony,
                      parsimony_decay=self.parsimony_decay)

        return new

    def mutate_leaf(self, inplace: bool = False) -> 'Program':
        """ This method selects a random FeatureNode and changes the associated feature

        The new FeatureNode will replace one random leaf among features and constants.

        Args:
            - inplace: bool (default=False)
                If True, the mutation is performed in place. If False, a new Program object is returned.

        Returns:
            - Program
                The new program after the mutation is applied
        """
        offspring = copy.deepcopy(self)

        leaves = offspring.get_features(
            return_objects=True) + offspring.get_constants(return_objects=True)

        mutate_point = random.choice(leaves)
        mutate_father = mutate_point.father

        # depth=0 generate a tree of only one FeatureNode
        new_feature = offspring._generate_tree(depth=0, father=mutate_father)

        if mutate_father:
            mutate_father.operands[mutate_father.operands.index(
                mutate_point)] = new_feature
        else:
            offspring.program = new_feature

        if inplace:
            self.program = offspring.program
            self._hash = None
            self.already_bootstrapped = False
            self.already_brgflow = False
            return self

        offspring._hash = None
        offspring.already_bootstrapped = False
        offspring.already_brgflow = False

        return offspring

    def mutate_operator(self, inplace: bool = False) -> 'Program':
        """ This method selects a random OperationNode and changes the associated operation

        The new OperationNode will replace one random leaf among features and constants.

        Args:
            - inplace: bool (default=False)
                If True, the mutation is performed in place. If False, a new Program object is returned.

        Returns:
            - Program
                The new program after the mutation is applied
        """
        offspring = copy.deepcopy(self.program)

        mutate_point = self._select_random_node(
            root_node=offspring, only_operations=True)

        if not mutate_point:  # Only a FeatureNode without any OperationNode
            new = Program(operations=self.operations,
                          features=self.features,
                          feature_probability=self.default_feature_probability,
                          default_feature_probability=self.default_feature_probability,
                          bool_update_feature_probability=self.bool_update_feature_probability,
                          const_range=self.const_range,
                          parsimony=self.parsimony,
                          parsimony_decay=self.parsimony_decay)

            new.init_program()

            if inplace:
                self.program = new
                self._hash = None
                self.already_bootstrapped = False
                self.already_brgflow = False
                return self
            return new

        new_operation = random.choice(self.operations)
        while new_operation['arity'] != mutate_point.arity or (mutate_point.symbol==new_operation.get('symbol')):
            new_operation = random.choice(self.operations)

        mutate_point.operation = new_operation.get('func')
        mutate_point.format_str = new_operation.get('format_str')
        mutate_point.format_tf = new_operation.get('format_tf')
        mutate_point.format_result = new_operation.get('format_result'),
        mutate_point.symbol = new_operation.get('symbol')
        mutate_point.format_diff = new_operation.get(
            'format_diff', new_operation.get('format_str'))

        if inplace:
            self.program = offspring
            self._hash = None
            self.already_bootstrapped = False
            self.already_brgflow = False
            return self

        new = Program(program=offspring,
                      operations=self.operations,
                      features=self.features,
                      feature_probability=self.feature_probability,
                      default_feature_probability=self.default_feature_probability,
                      bool_update_feature_probability=self.bool_update_feature_probability,
                      const_range=self.const_range,
                      parsimony=self.parsimony,
                      parsimony_decay=self.parsimony_decay)

        return new

    def recalibrate(self, inplace: bool = False) -> 'Program':
        """ This method recalibrate the constants of the program

        The new constants will be sampled from a uniform distribution.

        Args:
            - inplace: bool (default=False)
                If True, the recalibration is performed in place. If False, a new Program object is returned.

        Returns:
            - Program
                The new program after the recalibration is applied
        """
        offspring: Program = copy.deepcopy(self)

        offspring.set_constants(
            new=list(np.random.uniform(
                low=self.const_range[0],
                high=self.const_range[1],
                size=len(self.get_constants())
            ))
        )

        if inplace:
            self.program = offspring.program
            self.already_bootstrapped = False
            self.already_brgflow = False
            return self

        new = Program(program=offspring.program,
                      operations=self.operations,
                      features=self.features,
                      feature_probability=self.feature_probability,
                      default_feature_probability=self.default_feature_probability,
                      bool_update_feature_probability=self.bool_update_feature_probability,
                      const_range=self.const_range,
                      parsimony=self.parsimony,
                      parsimony_decay=self.parsimony_decay)

        return new

    def additive_expansion(self, inplace: bool = False) -> 'Program':
        """ This method perform a mutation on a random node of the current program

        An expansion presumes to include a new additive term in a formula containing
        a new random subtree 

        Args:
            - inplace: bool (default=False)
                If True, expand is performed in place. If False, a new Program object is returned.

        Returns:
            - Program
                The new program after the mutation is applied
        """

        def find_operation_nodes(node,symbol='+'):
            """
            Recursively traverses the tree starting at 'node' and collects 
            all OperationNode instances whose symbol is '/'.
            
            Returns:
                A list of matching nodes.
            """
            from symbolic_regression.Node import OperationNode
            operation_nodes = []
            
            if isinstance(node, OperationNode):
                if node.symbol == symbol:
                    operation_nodes.append(node)
                # Traverse each operand (child) recursively
                for child in node.operands:
                    operation_nodes.extend(find_operation_nodes(child))
            # If it's a FeatureNode, we don't do anything (it has no children to traverse)
            return operation_nodes


        offspring = copy.deepcopy(self)
        plus_nodes=find_operation_nodes(offspring.program,symbol='+')

        if (not plus_nodes):
            new = Program(operations=self.operations,
                        features=self.features,
                        feature_probability=self.default_feature_probability,
                        default_feature_probability=self.default_feature_probability,
                        bool_update_feature_probability=self.bool_update_feature_probability,
                        const_range=self.const_range,
                        parsimony=self.parsimony,
                        parsimony_decay=self.parsimony_decay)

            new.init_program()
            if inplace:
                self.program = new
                self._hash = None
                self.already_bootstrapped = False
                self.already_brgflow = False
                return self

            return new

        plus_node = np.random.choice(plus_nodes)

        add_operation = OperationNode(
                        operation=OPERATOR_ADD['func'],
                        arity=OPERATOR_ADD['arity'],
                        format_str=OPERATOR_ADD['format_str'],
                        format_tf=OPERATOR_ADD['format_tf'],
                        format_result=OPERATOR_ADD.get('format_result'),
                        symbol=OPERATOR_ADD['symbol'],
                        format_diff=OPERATOR_ADD.get(
                            'format_diff', OPERATOR_ADD['format_str']),
                        father=plus_node.father
                    )
        
        
        if plus_node.father is not None: 
            new_program=offspring.program
            index=plus_node.father.operands.index(plus_node)
            plus_node.father.operands[index]=add_operation
        else:
            new_program=add_operation

        plus_node.father = add_operation
        add_operation.operands.append(plus_node)

        ### QUA RICALCOLA FEATURE PROBABILITY
        new_addendum = self._generate_tree(father=add_operation, 
                                            parsimony=self.parsimony, 
                                            parsimony_decay=self.parsimony_decay)
        add_operation.operands.append(new_addendum)

        if inplace:
            self.program = new_program
            self._hash = None
            self.already_bootstrapped = False
            self.already_brgflow = False
            return self

        new = Program(program=new_program,
                        operations=self.operations,
                        features=self.features,
                        feature_probability=self.feature_probability,
                        default_feature_probability=self.default_feature_probability,
                        bool_update_feature_probability=self.bool_update_feature_probability,
                        const_range=self.const_range,
                        parsimony=self.parsimony,
                        parsimony_decay=self.parsimony_decay)

        return new
    
    def compute_FIM_diag(self, data):
        from sympy import lambdify
        def DiracDeltaV(x):
            return np.where(np.abs(x) < 1e-7, 1e7, 0)
        
        constants = np.array(self.get_constants())
        n_constants=len(constants)
        n_features = len(self.features)

        split_c = np.split(constants *np.ones((data.shape[0],1)), n_constants, 1)
        split_X = np.split(data[self.features].to_numpy(), n_features, 1)

        x_sym = ''
        for f in self.features:
            x_sym += f'{f},'
        x_sym = sympy.symbols(x_sym, real=True)
        c_sym = sympy.symbols('c0:{}'.format(n_constants), real=True)

        local_dict={}
        for x in x_sym:
            local_dict[str(x)]=x
        for c in c_sym:
            local_dict[str(c)]=c
        
        ##compute FIM
        string = self.program.render(format_diff=True)
        expr = sympy.parse_expr(string).subs(local_dict)
        FIM_avg=[]
        for ci in c_sym:
            row=[]
            for cj in c_sym:
                entry=sympy.diff(expr, ci).doit() * sympy.diff(expr, cj).doit()
                f_entry=lambdify([x_sym, c_sym], 
                                entry, 
                                modules=['numpy', {'DiracDelta': DiracDeltaV}])
                avg_entry=np.mean(f_entry(tuple(split_X), tuple(split_c)))
                #print(f'\n\n\n diff1 = {sympy.diff(expr, ci).doit()}\navg entry= {avg_entry}')
                row.append(float(avg_entry))
            FIM_avg.append(row)
        FIM_avg=np.array(FIM_avg)
        diag_FIM_avg=np.diag(FIM_avg)
        #print(f'\n\n\ndiagFIM matrix {diag_FIM_avg}\n\n\n')
        return diag_FIM_avg



    def regularize_brgflow(self, 
                            data,
                            knee=False, 
                            threshold=0.001,
                            inplace: bool = False) -> 'Program':
        """ This method perform a regularization of the tree aimed to streamline the tree structure and remove redundancies

        This mutation relies on the BRG Flow approach or on the bootstrap approach. 

        Args:
            - BRGFlow=True use brgflow approach
            - bootstrap=True use bootstrap approach
            - k number of bootstrap iterations
            - frac = dataset fraction used in bootstrapping coefficients
            - knee = use the knee method in brgflow to identify optimal cutting point
            - threshold = a priori threshold for brgflow regularization
            - inplace: bool (default=False)
                If True, the mutation is performed in place. If False, a new Program object is returned.

        Returns:
            - Program
                The new program after the mutation is applied
        """
        import copy
        from symbolic_regression.Program import Program
        from symbolic_regression.simplification import extract_operation, clean_division
        from sympy import lambdify

        
            
        def find_knee_value_orthogonal(array):
            # 1) Sort the array
            sorted_array = np.sort(array)
            n = len(sorted_array)
            
            # If there's only one point or two points, the concept of a "knee" might not be meaningful;
            # we'll just handle the trivial cases directly:
            if n < 3:
                return sorted_array[0]
            
            # 2) Define the x-coordinates as the indices (0 to n-1)
            x = np.arange(n)
            
            # 3) The line goes from (x1, y1) = (0, sorted_array[0]) to (x2, y2) = (n-1, sorted_array[-1])
            O = np.array([0, sorted_array[0]])
            Op = np.array([n - 1, sorted_array[-1]])
            OOp=Op-O

            # 4) Compute distances for each i
            distances = []
            for i in range(n):
                OA = np.array([x[i], sorted_array[i]]) - O
                OR= np.matmul(OOp,OA)/np.linalg.norm(OOp)
                distance = np.sqrt(np.abs(np.matmul(OA,OA)-OR**2))
                distances.append(distance)
            
            distances = np.array(distances)
            
            # 5) The knee index is where the orthogonal distance is maximum
            knee_index = np.argmax(distances)
            
            # Return the corresponding (sorted) value
            #print(f'\nTHRESHOLD: {sorted_array[knee_index]}\n')
            return sorted_array[knee_index]
        
            
        succesfull=0
        constants = np.array([item.feature for item in self.get_constants(return_objects=True)])
        n_constants=len(constants)

        if (n_constants<=2) or (self.program_depth  == 0) or (len(self.features_used)<1) or (self.already_brgflow):
            #if (self.already_brgflow):
            #    print('Already BRGFLOW')
            #elif (self.program_depth  == 0):
            #    print('no depth')
            #elif (len(self.features_used)<1):
            #    print('no features')
            #elif (n_constants<=2):
            #    print('less than 2 constants')
            #Case in which a 
            # 1) FeatureNode program is passed,
            # 2) the program does not have more than 2 constants,
            # A new tree is generated.
            new = Program(operations=self.operations,
                          features=self.features,
                          feature_assumption=self.feature_assumption,
                          feature_probability=self.default_feature_probability,
                          default_feature_probability=self.default_feature_probability,
                          bool_update_feature_probability=self.bool_update_feature_probability,
                          const_range=self.const_range,
                          parsimony=self.parsimony,
                          parsimony_decay=self.parsimony_decay)
            new.init_program()
            try:
                new.simplify(inplace=True)
            except ValueError:
                #print('error in simplfy new prog')
                new._override_is_valid = False

            if inplace:
                self.program = new
                self.override_is_valid = new._override_is_valid
                self._hash = None
                self.already_bootstrapped = False
                self.already_brgflow = False
                return self
            return new, succesfull
        


        offspring = copy.deepcopy(self)

        try:
            diag_FIM_avg=offspring.compute_FIM_diag(data)
            diag_FIM_avg=diag_FIM_avg/np.max(diag_FIM_avg)

        except (NameError, KeyError) as e:
            if isinstance(e, KeyError) and str(e) == "'ComplexInfinity'":
                # Ignore only this specific KeyError
                print('complex infinity')
            else:
                # Re-raise any other KeyError or any NameError               
                print('name error brutto in brgflow: new prog')
                #traceback.print_exc()  # if you want the full traceback
            ## generate a new program
            new = Program(operations=self.operations,
                            features=self.features,
                            feature_assumption=self.feature_assumption,
                            feature_probability=self.default_feature_probability,
                            default_feature_probability=self.default_feature_probability,
                            bool_update_feature_probability=self.bool_update_feature_probability,
                            const_range=self.const_range,
                            parsimony=self.parsimony,
                            parsimony_decay=self.parsimony_decay)
            new.init_program()
            try:
                new.simplify(inplace=True)
            except ValueError:
                #print('error in simplfy new prog')
                #new._override_is_valid = False
                pass
            if inplace:
                self.program = new
                self.override_is_valid = new._override_is_valid
                self._hash = None
                self.already_bootstrapped = False
                self.already_brgflow = False
                return self
            return new, succesfull
        
        if knee:
            threshold_knee = find_knee_value_orthogonal(diag_FIM_avg)
            threshold=np.min([threshold_knee,threshold])

        mask=diag_FIM_avg<=threshold 
        sum_masked=np.sum(mask)
        constants[mask]=0.

        if sum_masked>0:
            try:
                x_sym = ''
                for f in self.features:
                    x_sym += f'{f},'
                x_sym = sympy.symbols(x_sym, real=True)

                local_dict={}
                for x in x_sym:
                    local_dict[str(x)]=x

                offspring.set_constants(new=constants)
                string = offspring.program.render()
                expr = sympy.parse_expr(string).subs(local_dict)
                new_program = extract_operation(expr)
                new_program = clean_division(new_program)
                offspring.program=new_program

    
                #print(f'!!!!!BRGFLOW SUCCESSFULLY APPLIED!!!!\n/Before:{self.program}\nAfter:{offspring.program}\nFIM:{diag_FIM_avg}')
                succesfull=1
                if inplace:
                    self.program = new_program
                    self._hash = None
                    self.already_brgflow = True
                    return self
                else:
                    offspring.already_brgflow = True
                    offspring._hash = None
                    return offspring, succesfull

            except ValueError:
                print('error in brgflow simplification')
                #traceback.print_exc()
                ## generate a new program
                new = Program(operations=self.operations,
                                features=self.features,
                                feature_assumption=self.feature_assumption,
                                feature_probability=self.default_feature_probability,
                                default_feature_probability=self.default_feature_probability,
                                bool_update_feature_probability=self.bool_update_feature_probability,
                                const_range=self.const_range,
                                parsimony=self.parsimony,
                                parsimony_decay=self.parsimony_decay)
                new.init_program()
                try:
                    new.simplify(inplace=True)
                except ValueError:
                    pass
                    #print('error in simplfy new prog')
                    #new._override_is_valid = False
                if inplace:
                    self.program = new
                    self.override_is_valid = new._override_is_valid
                    self._hash = None
                    self.already_bootstrapped = False
                    self.already_brgflow = False
                    return self
                return new, succesfull
        else:
            #print(f'no constants to mask: {np.diag(FIM_avg)}')
            ## generate a new program
            new = Program(operations=self.operations,
                            features=self.features,
                            feature_assumption=self.feature_assumption,
                            feature_probability=self.default_feature_probability,
                            default_feature_probability=self.default_feature_probability,
                            bool_update_feature_probability=self.bool_update_feature_probability,
                            const_range=self.const_range,
                            parsimony=self.parsimony,
                            parsimony_decay=self.parsimony_decay)
            new.init_program()
            try:
                new.simplify(inplace=True)
            except ValueError:
                #print('error in simplfy new prog')
                #new._override_is_valid = False
                pass
            if inplace:
                self.program = new
                self.override_is_valid = new._override_is_valid
                self._hash = None
                self.already_bootstrapped = False
                self.already_brgflow = False
                return self
            return new, succesfull
            
    def regularize_bootstrap(self, 
                            data,
                            k=150,
                            frac=0.70,
                            n_jobs=3,
                            inplace: bool = False) -> 'Program':
        """ This method perform a regularization of the tree aimed to streamline the tree structure and remove redundancies

        This mutation relies on the BRG Flow approach or on the bootstrap approach. 

        Args:
            - BRGFlow=True use brgflow approach
            - bootstrap=True use bootstrap approach
            - k number of bootstrap iterations
            - frac = dataset fraction used in bootstrapping coefficients
            - knee = use the knee method in brgflow to identify optimal cutting point
            - threshold = a priori threshold for brgflow regularization
            - inplace: bool (default=False)
                If True, the mutation is performed in place. If False, a new Program object is returned.

        Returns:
            - Program
                The new program after the mutation is applied
        """
        import copy
        from symbolic_regression.Program import Program
        
        
        def single_bootstrap(program,
                    data,
                    target,
                    weights,
                    constants_optimization,
                    constants_optimization_conf,
                    frac=0.6,
                    low=-1,
                    high=1):
    
            # Sample data
            bs_data = data.sample(frac=frac, replace=False)
            bs_data[weights] = 1.0
            
            # Create a copy of the program
            recalibrated = copy.deepcopy(program)
            
            # Set random constants
            new_constants = list(np.random.uniform(low=low, 
                                                high=high, 
                                                size=len(program.get_constants())))
            recalibrated.set_constants(new=new_constants)
            
            # Optimize
            optimized = recalibrated.optimize(
                        data=bs_data,
                        target=target,
                        weights=weights,
                        constants_optimization=constants_optimization,
                        constants_optimization_conf=constants_optimization_conf,
                        inplace=False
                    )
            
            return optimized.get_constants(return_objects=False)

        def Bootstrapping_constants(program,
                                    data,
                                    target,
                                    weights,
                                    constants_optimization,
                                    constants_optimization_conf,
                                    k=1000,
                                    frac=0.6,
                                    n_jobs=3):
            from joblib import Parallel, delayed
            import copy
            from typing import List

            n_constants = len(program.get_constants())
            n_features_used = len(program.features_used)
            if not (n_constants > 0 and n_features_used > 0):
                print('This should never happen')
                return None
            
    
            bootstrapped_constants: List[float] = Parallel(n_jobs=n_jobs)(delayed(single_bootstrap)(program=program,
                                                                                                data=data,
                                                                                                target=target,
                                                                                                weights=weights,
                                                                                                constants_optimization=constants_optimization,
                                                                                                constants_optimization_conf=constants_optimization_conf,
                                                                                                frac=frac) for _ in range(k))
            
            
            return np.array(bootstrapped_constants)
        
            
        succesfull=0
        constants = np.array([item.feature for item in self.get_constants(return_objects=True)])
        n_constants=len(constants)

        find_config=False
        for f in self.fitness_functions:
            if (f.minimize==True) and (f.constants_optimization) and (f.constants_optimization_conf):
                constants_optimization=f.constants_optimization
                constants_optimization_conf=f.constants_optimization_conf
                target=f.target
                weights=f.weights
                find_config=True

        if (not find_config) or (n_constants<=2) or (self.program_depth  == 0) or (len(self.features_used)<1) or (self.already_bootstrapped):
            # Case in which a 
            # 1) FeatureNode program is passed,
            # 2) the program does not have more than 2 constants,
            # 3) was unabl to find optimization configuration
            # A new tree is generated
            #if (not find_config):
            #    print('no config')
            #elif (n_constants<=2):
            #    print('less than 2 const')
            #elif (self.program_depth  == 0):
            #    print('no depth')
            #elif (len(self.features_used)<1):
            #    print('zero features used')
            #elif (self.already_bootstrapped):
            #    print('already bootstrapped')
            new = Program(operations=self.operations,
                            features=self.features,
                            feature_assumption=self.feature_assumption,
                            feature_probability=self.default_feature_probability,
                            default_feature_probability=self.default_feature_probability,
                            bool_update_feature_probability=self.bool_update_feature_probability,
                            const_range=self.const_range,
                            parsimony=self.parsimony,
                            parsimony_decay=self.parsimony_decay)
            new.init_program()
            try:
                new.simplify(inplace=True)
            except ValueError:
                #print('error in simplfy new prog')
                #new._override_is_valid = False
                pass
            if inplace:
                self.program = new
                self._override_is_valid=new._override_is_valid
                self._hash = None
                self.already_bootstrapped = False
                self.already_brgflow = False
                return self, succesfull
            return new, succesfull

        offspring = copy.deepcopy(self)
        
        constants_boots=Bootstrapping_constants(program=offspring,
                                                data=data,
                                                target=target,
                                                weights=weights,
                                                constants_optimization=constants_optimization,
                                                constants_optimization_conf=constants_optimization_conf,
                                                k=k,
                                                frac=frac,
                                                n_jobs=n_jobs
                                                )
        #mask constants which are not significantly differrent from zero
        
        quantiles = np.quantile(constants_boots,q=[0.25,0.75],axis=0)
        mask=quantiles.prod(0)<0
        sum_masked=np.sum(mask)
        or_constants=constants.copy()
        constants[mask]=0.

        if sum_masked>0:
            try:
                from symbolic_regression.simplification import extract_operation, clean_division
                x_sym = ''
                for f in self.features:
                    x_sym += f'{f},'
                x_sym = sympy.symbols(x_sym, real=True)

                local_dict={}
                for x in x_sym:
                    local_dict[str(x)]=x

                offspring.set_constants(new=constants)
                string = offspring.program.render()
                expr = sympy.parse_expr(string).subs(local_dict)
                new_program = extract_operation(expr)
                new_program = clean_division(new_program)
                offspring.program=new_program

                succesfull=1
                #print(f'!!!SUCCESSFULL BOOTSTRAP!!!!!\nBefore:{self.program}\nold const:{or_constants}\nnew const {constants}\nAfter:{offspring.program}')
                
                if inplace:
                    self.program = new_program
                    self.already_bootstrapped = True
                    self._hash=None
                    return self, succesfull
                else:
                    offspring.already_bootstrapped = True
                    offspring._hash=None
                    return offspring, succesfull
            except ValueError:
                #print('error in simplfying sfter mask computation')
                ## generate a new program
                new = Program(operations=self.operations,
                                features=self.features,
                                feature_assumption=self.feature_assumption,
                                feature_probability=self.default_feature_probability,
                                default_feature_probability=self.default_feature_probability,
                                bool_update_feature_probability=self.bool_update_feature_probability,
                                const_range=self.const_range,
                                parsimony=self.parsimony,
                                parsimony_decay=self.parsimony_decay)
                new.init_program()
                try:
                    new.simplify(inplace=True)
                except ValueError:
                    pass
                if inplace:
                    self.program = new
                    self._override_is_valid = new._override_is_valid
                    self._hash = None
                    self.already_bootstrapped = False
                    self.already_brgflow = False
                    return self, succesfull
                return new, succesfull
        else:
            #print('no constants to mask')
            ## generate a new program
            new = Program(operations=self.operations,
                            features=self.features,
                            feature_assumption=self.feature_assumption,
                            feature_probability=self.feature_probability,
                            default_feature_probability=self.default_feature_probability,
                            bool_update_feature_probability=self.bool_update_feature_probability,
                            const_range=self.const_range,
                            parsimony=self.parsimony,
                            parsimony_decay=self.parsimony_decay)
            new.init_program()
            try:
                new.simplify(inplace=True)
            except ValueError:
                #print('error in simplfy new prog')
                #new._override_is_valid = False
                pass
            if inplace:
                self.program = new
                self.override_is_valid = new._override_is_valid
                self._hash = None
                self.already_bootstrapped = False
                self.already_brgflow = False
                return self, succesfull
            return new, succesfull
        
   