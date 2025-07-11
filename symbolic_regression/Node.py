import copy
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd


class Node(ABC):
    """ A node can represent an operation or a feature in a binary tree
    """

    def __init__(self, father=None) -> None:
        self.father = father

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))

        return result

    @abstractmethod
    def evaluate(self, data: Union[dict, pd.Series, pd.DataFrame]) -> Union[int, float]:
        raise NotImplementedError

    @abstractmethod
    def _get_all_operations(self, all_operations: List['OperationNode'] = list) -> List['OperationNode']:
        raise NotImplementedError

    @abstractmethod
    def _get_complexity(self, complexity: int = 0) -> int:
        raise NotImplementedError

    @abstractmethod
    def _get_constants(self, constants: List['FeatureNode'] = list) -> List['FeatureNode']:
        raise NotImplementedError

    @abstractmethod
    def _get_depth(self, depth: int = 0) -> int:
        raise NotImplementedError

    @abstractmethod
    def _get_features(self, features: List['FeatureNode'] = list) -> List['FeatureNode']:
        raise NotImplementedError

    @abstractmethod
    def _get_operations_used(self, operations_used: List['OperationNode'] = list) -> List['OperationNode']:
        raise NotImplementedError

    @abstractmethod
    def hash(self, hash: int = 0) -> int:
        raise NotImplementedError

    @abstractmethod
    def render(self, render: str = '') -> str:
        raise NotImplementedError


class OperationNode(Node):
    """ An OperationNode represents an operation in a binary tree

    Attributes:
        = operation: callable
            The operation to be performed on the operands
        = format_tf: str
            The format string to be used to render the operation in the TensorFlow graph
        = format_diff: str  (optional)
            The format string to be used to render the operation in the TensorFlow graph when differentiating
        = arity: int
            The number of operands the operation requires
        = symbol: str
            The symbol to be used to represent the operation in the program
        = format_str: str
            The format string to be used to render the operation in the program
        = father: Node
            The father of the OperationNode
        = operands: List[Node]
            The operands of the OperationNode

    Methods:
        - __repr__: str (override)
            This method is used to represent the OperationNode as a string
        - add_operand: None
            This method adds an operand to the OperationNode
        - evaluate: Union[int, float]
            This method recursively calls the operations callable to evaluate the result of the program on data
        - _get_all_operations: List[OperationNode]  (private)
            This method recursively gets all the operations in the program
        - _get_complexity: int  (private)
            This method recursively increment the complexity count of this program
        - _get_constants: int  (private)
            This method recursively gets all the constants in the program
        - _get_depth: int  (private)
            This method recursively gets the depth of the program
        - _get_features: int  (private)
            This method recursively gets all the features in the program
        - _get_operations_used: int  (private)
            This method recursively gets all the operations used in the program
        - hash: List
            This method recursively gets the hash of the program
        - render: str
            This method recursively renders the program as a string
    """

    def __init__(self, operation: callable, format_tf: str, format_diff: str, format_result: str, arity: int, symbol: str, format_str: str, father: Node) -> None:
        """ This method initializes the OperationNode

        Args:
            = operation: callable
                The operation to be performed on the operands
            = format_tf: str
                The format string to be used to render the operation in the TensorFlow graph
            = format_diff: str  (optional)
                The format string to be used to render the operation in the TensorFlow graph when differentiating
            = arity: int
                The number of operands the operation requires
            = symbol: str
                The symbol to be used to represent the operation in the program
            = format_str: str
                The format string to be used to render the operation in the program
            = father: Node
                The father of the OperationNode
        """
        super().__init__(father)

        self.operation = operation
        self.symbol = symbol
        self.arity = arity
        self.format_str = format_str
        self.format_tf = format_tf
        self.format_result = format_result
        self.format_diff = format_diff if format_diff else format_str

        self.operands = []

    def __repr__(self) -> str:
        """ This method is used to represent the OperationNode as a string
        """
        return self.render()

    def add_operand(self, operand: Node) -> None:
        """ This method adds an operand to the OperationNode

        In case the OperationNode already has the maximum number of operands, an AttributeError is raised.

        Args:
            = operand: Node
                The operand to add to the OperationNode

        Returns:
            = None
        """
        if len(self.operands) >= self.arity:
            raise AttributeError(
                f'This operation support at most {self.arity} operands.')

        self.operands.append(operand)

    def evaluate(self, data: Union[dict, pd.Series, pd.DataFrame]) -> Union[int, float]:
        """ This method recursively calls the operations callable to evaluate the result of the program on data

        Args:
            = data: dict, pd.Series, or pd.DataFrame
                The data to evaluate the program on

        Returns:
            = result: Union[int, float]
                The result of the evaluation
        """
        result = None
        if not (isinstance(data, dict) or isinstance(data, pd.DataFrame) or isinstance(data, pd.Series)):
            raise TypeError(
                f'Evaluation supports only data as dict, pd.Series or pd.DataFrame objects')

        result = self.operation(*[node.evaluate(data=data)
                                for node in self.operands])

        return result

    def _get_all_operations(self, all_operations: List['OperationNode'] = list) -> List['OperationNode']:
        """ This method recursively gets all the operations in the program

        Args:
            = all_operations: List[OperationNode]
                The list of all the operations in the program. To be recursively updated.

        Returns:
            = all_operations: List[OperationNode]
                The list of all the operations in the program
        """
        all_operations.append(self)

        for child in self.operands:
            if isinstance(child, OperationNode):
                all_operations = child._get_all_operations(
                    all_operations=all_operations)

        return all_operations

    def _get_complexity(self, base_complexity: int = 0) -> int:
        """ This method recursively increment the complexity count of this program

        Args:
            = base_complexity: int
                The complexity count to be recursively updated

        Returns:
            = base_complexity: int
                The complexity count of this program
        """

        base_complexity += 1  # Count for this operation contribution

        for child in self.operands:
            base_complexity = child._get_complexity(
                base_complexity=base_complexity)

        return base_complexity

    def _get_constants(self, const_list: List = list) -> List:
        """ This method recursively gets all the constants in the program

        Args:
            = const_list: List
                The list of all the constants in the program. To be recursively updated.

        Returns:
            = const_list: List
                The list of all the constants in the program
        """
        for child in self.operands:
            if isinstance(child, OperationNode):
                const_list = child._get_constants(const_list=const_list)
            elif child.is_constant:
                const_list += [child]

        return const_list

    def _get_depth(self, base_depth: int = 0) -> int:
        """ This method recursively gets the depth of the program

        Args:
            = base_depth: int
                The depth of the program. To be recursively updated.

        Returns:
            = new_depth: int
                The depth of the program
        """

        base_depth += 1

        for child in self.operands:
            new_depth = max(
                child._get_depth(base_depth=base_depth)
                for child in self.operands)

        return new_depth

    def _get_features(self, features_list: List = list, return_objects: bool = False) -> List:
        """ This method recursively gets all the features in the program

        Args:
            = features_list: List
                The list of all the features in the program. To be recursively updated.
            = return_objects: bool
                If True, the features are returned as FeatureNode objects. If False, the features are returned as strings.

        Returns:
            = features_list: List
                The list of all the features in the program
        """
        for child in self.operands:
            if isinstance(child, OperationNode):
                features_list = child._get_features(
                    features_list=features_list, return_objects=return_objects)
            elif not child.is_constant:
                if return_objects:
                    features_list.append(child)
                else:
                    features_list.append(child.feature)

                    # If returning only the string with the names, we don't want duplicates
                    features_list = list(set(features_list))

        return features_list

    def _get_operations_used(self, base_operations_used: Dict = dict) -> Dict:
        """ This method recursively gets all the operations used in the program

        Args:
            = base_operations_used: Dict
                The dictionary of all the operations used in the program. To be recursively updated.

        Returns:
            = base_operations_used: Dict
                The dictionary of all the operations used in the program
        """
        if not base_operations_used.get(self.operation):
            base_operations_used[self.operation] = 0

        base_operations_used[self.operation] += 1

        for op in self.operands:
            if isinstance(op, OperationNode):
                base_operations_used = op._get_operations_used(
                    base_operations_used=base_operations_used)

        return base_operations_used

    def hash(self, hash_list: List = list) -> List['Hash']:
        """ This method recursively hashes the program

        Args:
            = hash_list: List
                The list of hashes of the program. To be recursively updated.

        Returns:
            = hash_list: List[Hash]
                The list of hashes of the program

        """

        # Evaluate child hashes to then evaluate the operation one
        child_hash = []
        for child in self.operands:
            ch = child.hash(hash_list=[])
            if isinstance(ch, str):
                child_hash.append(ch)
            else:
                child_hash += ch

        operation_hash = Hash(''.join([self.symbol, *child_hash[:self.arity]]))
        operation_hash.ref = self

        # Add the operation hash to the list and then the children ones

        for child in child_hash:
            if isinstance(child, str):
                # Keep the insert(0) to keep the order for the operation_hash computation
                hash_list.insert(0, child)

        # Keep the insert(0) to keep the order for the operation_hash computation
        hash_list.insert(0, operation_hash)

        return hash_list

    @property
    def is_valid(self):
        """ This method checks if the program is valid

        The validity of a program is defined as the validity of all its operations.
        If any of the operations is invalid, the program is invalid.
        A program can be invalid by structure or by the values of its constants.
        For example, if the denominator of a division operation is a constant with value 0, the program is invalid on that data.
        Also, if the program optimizer produces np.inf or np.nan, the program is invalid.

        Returns:
            = v: bool
                True if the program is valid, False otherwise

        """

        v = True

        for child in self.operands:
            v = v and child.is_valid

        return v

    def render(self, data: Union[dict, pd.Series, pd.DataFrame, None] = None, format_tf: bool = False, format_diff: bool = False, format_result: bool = False) -> str:
        """ This method render the string of the program according to the formatting rules of its operations

        Args:
            = data: Union[dict, pd.Series, pd.DataFrame, None]
                The data to be used to render the program.
                If it's None, the program is rendered with the names of the features.
                If it's a dictionary, pd.Series or pd.DataFrame, the program is rendered with the values of the features.
            = format_tf: bool
                If True, the program is rendered in the format of a TensorFlow function
            = format_diff: bool
                If True, the program is rendered in the format of a TensorFlow function with the derivatives of the operations

        Returns:
            = s: str
                The string of the program
        """
        if format_tf:
            return self.format_tf.format(*[
                node.render(data=data, format_tf=True)
                for node in self.operands
            ])

        if format_diff:
            return self.format_diff.format(*[
                node.render(data=data, format_diff=format_diff)
                for node in self.operands
            ])
        
        if format_result:
            return self.format_result.format(*[
                node.render(data=data, format_result=format_result)
                for node in self.operands
            ])

        return self.format_str.format(*[
            node.render(data=data, format_diff=format_diff)
            for node in self.operands
        ])


class FeatureNode(Node):
    """ A FeatureNode represent a terminal node of the binary tree of the program and is always a feature or a constant

    Attributes:
        = feature: Union[str, float]    (default: None)
            The name of the feature or the value of the constant
        = father: Union[OperationNode, None]    (default: None)
            The father of this node in the binary tree of the program
        = is_constant: bool  (default: False)
            If True, the feature is a constant, otherwise it is a feature
        = arity: int    (default: 0)
            The arity of the node. It is always 0 because it is a constant and not an operator
        = index: Union[int, None]   (default: None)
            The index of the feature in the data

    Methods: Uses the methods of the Node class        
    """

    def __init__(self, feature: Union[str, float], father: Union[OperationNode, None] = None, is_constant: bool = False) -> None:
        """ To initalize this FeatureNode

        Args:
            = feature: Union[str, float]
                The name of the feature or the value of the constant
            = father: Union[OperationNode, None]    (default: None)
                The father of this node in the binary tree of the program
            = is_constant: bool   (default: False)
                If True, the feature is a constant, otherwise it is a feature
        """
        super().__init__(father)

        self._feature_confidence_intervals: List[float, float] = [np.nan, np.nan]
        self._feature: Union[str, float] = feature
        self.feature: Union[str, float] = feature

        self.arity = 0  # because it is a constant and not an operator
        self.is_constant = is_constant
        self.index = None

    def __repr__(self) -> str:
        """ This method returns the string representation of the FeatureNode
        """
        return self.render()

    @property
    def feature_confidence_intervals(self) -> List[float]:
        """ This property returns the confidence intervals of the feature

        Returns:
            = self._feature_confidence_intervals: List[float, float]
                The confidence intervals of the feature
        """
        return self._feature_confidence_intervals

    @property
    def feature(self) -> Union[str, float]:
        """ This property returns the feature of the FeatureNode

        Returns:
            = self._feature: Union[str, float]
                The feature of the FeatureNode
        """
        return self._feature
    
    @feature.setter
    def feature(self, feature: Union[float, List[float]]) -> None:
        """ This property sets the feature of the FeatureNode

        Args:
            - feature: Union[str, float]
                The feature of the FeatureNode

        Returns:
            - None
        """
        if isinstance(feature, str):
            self._feature = feature
            self._feature_confidence_intervals = [np.nan, np.nan]
            self.is_constant = False
        if isinstance(feature, float) or isinstance(feature, int):
            self._feature = float(feature)
            self._feature_confidence_intervals = [float(feature), float(feature)]
            self.is_constant = True
        elif isinstance(feature, list):
            feature = self._reject_outliers(values=feature, n_sigma=5.)

            self._feature = float(np.median(feature))
            self._feature_confidence_intervals = [float(np.quantile(feature, 0.05)), float(np.quantile(feature, 0.95))]
            self.is_constant = True

    @feature.getter
    def feature(self) -> float:
        """ This property returns the feature of the FeatureNode

        Returns:
            - self._feature: Union[str, float]
                The feature of the FeatureNode
        """
        return self._feature
    
    @staticmethod
    def _reject_outliers(values: List[float], n_sigma: float = 2.):
        """ This method returns the values of a list of values that are not outliers

        Args:
            - values: List[float]
                The list of values
            - n_sigma: float    (default: 2.)
                The number of standard deviations to consider a value an outlier

        Returns:
            - values[s < n_sigma]: List[float]
                The values of the list that are not outliers
        """
        values = np.array(values)

        d = np.abs(values - np.median(values))
        mdev = np.median(d)
        s = d / mdev if mdev else np.zeros(len(d))
        return list(values[s < n_sigma])

    def evaluate(self, data: Union[dict, pd.Series, pd.DataFrame, None] = None) -> Union[int, float]:
        """ This function evaluate the value of a FeatureNode, which is the datapoint passed as argument

        The data argument needs to be accessible by the name of the feature of this node.
        This is a FeatureNode, the end of a branch of a tree, so its evaluation is simply the value of that
        feature in a specific datapoint.

        In the recursive call stack of the evaluation of a program, this is the terminal call and
        always return or the constant numerical value (if is_constant==True) or the value of the feature
        from the data passed as argument.

        Args:
            = data: Union[dict, pd.Series, pd.DataFrame, None]  (default: None)
                The data to be used to evaluate the program.
                If it's None, a TypeError is raised.
                If it's a dictionary, pd.Series or pd.DataFrame, the program is evaluated with the values of the features.

        Returns:
            = result: Union[int, float]
                The value of the feature in the datapoint passed as argument
        """

        result = None

        if self.is_constant:
            result = self.feature

        elif data is not None and (isinstance(data, pd.Series) or isinstance(
                data, pd.DataFrame) or isinstance(data, dict)):
            result = data[self.feature]

        else:
            raise TypeError(
                f'Non constants FeatureNodes need data to be evaluated')

        return result

    def _get_all_operations(self, all_operations: List = list) -> List:
        """ This method returns a list of all the operations in the program

        This is a FeatureNode, the end of a branch of a tree, so it returns an empty list.

        Args:
            = all_operations: List  (default: list)
                The list of all the operations in the program

        Returns:
            = all_operations: List
                The list of all the operations in the program
        """
        return all_operations

    def _get_complexity(self, base_complexity: int = 0) -> int:
        """ This method returns the complexity of the FeatureNode

        This is a FeatureNode, the end of a branch of a tree, so it returns the base complexity plus 1.

        Args:
            = base_complexity: int  (default: 0)
                The complexity of the program

        Returns:
            = base_complexity + 1: int
                The complexity of the FeatureNode
        """
        return base_complexity + 1

    def _get_constants(self, constants: List = list) -> List:
        """ This method returns a list of all the constants in the program

        This is a FeatureNode, the end of a branch of a tree, so it returns an empty list.

        Args:
            - constants: List  (default: list)
                The list of all the constants in the program

        Returns:
            - constants: List
                The list of all the constants in the program
        """
        return constants

    def _get_depth(self, base_depth: int = 0) -> int:
        """ This method returns the depth of the FeatureNode

        This is a FeatureNode, the end of a branch of a tree, so it returns the base depth plus 1.

        Args:
            = base_depth: int   (default: 0)
                The depth of the program

        Returns:
            = base_depth + 1: int
                The depth of the FeatureNode
        """
        return base_depth + 1

    def _get_features(self, features_list: list = list):
        return features_list

    def _get_operations_used(self, operations_used: List = list) -> List:
        """ This method returns a list of all the operations used in the program

        This is a FeatureNode, the end of a branch of a tree, so it returns an empty list.

        Args:
            - operations_used: List  (default: list)
                The list of all the operations used in the program

        Returns:
            - operations_used: List
                The list of all the operations used in the program
        """
        return operations_used

    def hash(self, hash_list: list = list) -> 'Hash':
        """ This method returns the hash of the FeatureNode

        This is a FeatureNode, the end of a branch of a tree, so it returns the hash of the feature of this node.

        Args:
            - hash_list: list  (default: list)
                The list of all the hashes of the nodes in the program

        Returns:
            - Hash(self.render()): 'Hash'
                The hash of the FeatureNode
        """

        if self.is_constant:
            h = Hash('C')
        else:
            h = Hash(self.render())

        h.ref = self

        return h

    @property
    def is_valid(self):
        """ This property returns True if the feature of this node is not NaN

        Returns:
            = not pd.isna(self.feature): bool
                True if the feature of this node is not NaN
        """
        return not pd.isna(self.feature)

    def render(self, data: Union[dict, pd.Series, pd.DataFrame, None] = None, format_tf: bool = False, format_diff: bool = False, format_result: bool = False) -> str:
        """ This method render the string representation of this FeatureNode

        If data is provided, the rendering consist of the value of the datapoint of the feature of this
        FeatureNode.
        If data is not provided, the rendering will be the name of the feature.
        If this node is a constant value (is_constant==True) then that numerical value is returned

        Args:
            = data: Union[dict, pd.Series, pd.DataFrame, None]  (default: None)
                The data to be used to evaluate the program.
                If it's None, a TypeError is raised.
                If it's a dictionary, pd.Series or pd.DataFrame, the program is rendered with the values of the features.
            = format_tf: bool  (default: False)
                If True, the rendering is in the format of a tensorflow program
            = format_diff: bool  (default: False)
                If True, the rendering is in the format of a differential program

        Returns:
            = result: str
                The string representation of this FeatureNode
        """

        if self.is_constant:
            if format_tf:
                return f'constants[{self.index}]'
            elif format_diff:
                return f'c{self.index}'
            return str(self.feature)

        if data is not None:  # Case in which I render the value of the feature in the datapoint instead of its name
            if format_tf:
                return f'X[{list(data.index).index(self.feature)}]'
            return self.evaluate(data=data)

        return self.feature


class InvalidNode(Node):
    """ This class represents an invalid node in a program
    """

    def __init__(self, father: Node = None) -> None:
        """ This method initializes an InvalidNode

        Args:
            = father: Node  (default: None)
                The father of this node
        """
        super().__init__(father=father)

        self.is_constant = True  # This is a constant node

    def _get_all_operations(self, all_operations: List = list) -> List:
        """ This method returns a list of all the operations in the program

        This is an InvalidNode, the end of a branch of a tree, so it returns the same list it received.

        Args:
            = all_operations: List  (default: list)
                The list of all the operations in the program

        Returns:
            = all_operations: List
                The list of all the operations in the program
        """
        return all_operations

    def _get_features(self, features_list: List = list) -> List:
        """ This method returns a list of all the features in the program

        This is an InvalidNode, the end of a branch of a tree, so it returns the same list it received.

        Args:
            = features_list: List  (default: list)
                The list of all the features in the program

        Returns:
            = features_list: List
                The list of all the features in the program
        """
        return features_list

    def _get_complexity(self, base_complexity: int = 0) -> int:
        """ This method returns the complexity of the InvalidNode

        This is an InvalidNode, the end of a branch of a tree, so it returns the base complexity plus 1.

        Args:
            = base_complexity: int  (default: 0)
                The complexity of the program

        Returns:
            = base_complexity + 1: int
                The complexity of the InvalidNode
        """
        return base_complexity + 1

    def _get_constants(self, constants: List['FeatureNode'] = list) -> List['FeatureNode']:
        """ This method returns a list of all the constants in the program

        This is an InvalidNode, the end of a branch of a tree, so it returns the same list it received.

        Args:
            - constants: List['FeatureNode']  (default: list)
                The list of all the constants in the program

        Returns:
            - constants: List['FeatureNode']
                The list of all the constants in the program
        """
        return constants

    def _get_depth(self, base_depth: int = 0) -> int:
        """ This method returns the depth of the InvalidNode

        This is an InvalidNode, the end of a branch of a tree, so it returns the base depth plus 1.

        Args:
            = base_depth: int  (default: 0)
                The depth of the program

        Returns:
            = base_depth + 1: int
                The depth of the InvalidNode
        """
        return base_depth + 1

    def _get_operations_used(self, base_operations_used: Dict = dict) -> Dict:
        """ This method returns the operations used in the program

        This is an InvalidNode, the end of a branch of a tree, so it returns the same dictionary it received.

        Args:
            = base_operations_used: Dict  (default: dict)
                The dictionary of all the operations used in the program.

        Returns:
            = base_operations_used: Dict
                The dictionary of all the operations used in the program
        """
        return base_operations_used

    def hash(self, hash_list: List = list) -> 'Hash':
        """ This method returns the hash of the InvalidNode

        This is an InvalidNode, the end of a branch of a tree, so it returns the hash of the feature of this node.

        Args:
            - hash_list: List  (default: list)
                The list of all the hashes of the nodes in the program

        Returns:
            - Hash('InvalidNode'): Hash
                The hash of the InvalidNode
        """
        h = Hash('InvalidNode')
        h.ref = self
        return h

    @property
    def is_valid(self):
        """ This property returns False because the InvalidNode is not valid by definition
        """
        return False

    def evaluate(self, data: Union[dict, pd.Series, pd.DataFrame, None] = None) -> None:
        """ This method evaluates the InvalidNode

        This is an InvalidNode, the end of a branch of a tree, so it returns np.inf.

        Args:
            = data: Union[dict, pd.Series, pd.DataFrame, None]  (default: None)
                The data to be used to evaluate the program.
                If it's a dictionary, pd.Series or pd.DataFrame, the program is rendered with the values of the features.

        Returns:
            = np.inf: float
                The InvalidNode is not valid, so it always returns np.inf
        """
        return np.inf

    @property
    def feature(self) -> float:
        return np.inf

    def render(self, data: Union[dict, pd.Series, None] = None, format_tf: bool = False, format_diff: bool = False, format_result: bool = False) -> str:
        """ This method render the string representation of this InvalidNode

        This is an InvalidNode, the end of a branch of a tree, so it returns 'InvalidNode'.

        Args:
            = data: Union[dict, pd.Series, pd.DataFrame, None]  (default: None)
                The data to be used to evaluate the program. The InvalidNode do not use the data.
            = format_tf: bool  (default: False)
                If True, the rendering is in the format of a tensorflow program
            = format_diff: bool  (default: False)
                If True, the rendering is in the format of a differential program

        Returns:
            = 'InvalidNode': str
                The string representation of this InvalidNode which is independent of the data
        """
        return 'InvalidNode'


class Hash(str):

    ref: Node = None

    def __new__(cls, string):
        """ This method creates a new Hash 

        This method creates a new Hash from a string. The Hash is a string with a reference to the object that created it.

        Args:
            = string: str
                The string to be hashed

        Returns:
            = instance: Hash
                The Hash of the string
        """
        instance = super().__new__(cls, string)
        instance.ref: Node = None
        return instance
