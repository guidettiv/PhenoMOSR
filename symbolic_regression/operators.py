import operator

import numpy as np


def _protected_exp(x):
    try:
        return np.exp(x)
    except OverflowError:
        return np.inf


def _protected_mul(x1, x2):
    """Closure of multiplication for zero arguments."""
    x1 = np.array(x1, dtype=np.float32)
    x2 = np.array(x2, dtype=np.float32)
    with np.errstate(over='ignore', under='ignore'):
        return np.multiply(x1, x2)


def _protected_division(x1, x2):
    """Division (x1/x2) without custom handling for zero denominator."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.divide(x1, x2)


def _protected_sqrt(x1):
    with np.errstate(invalid='ignore'):
        return np.sqrt(np.abs(x1))


def _protected_log(x1):
    """Closure of log for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log(np.abs(x1))


def _protected_inverse(x1):
    """Closure of inverse for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.divide(1., x1)


def _protected_pow(x1, x2):
    """Closure of pow for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        x1 = np.array(x1, dtype=np.float32)
        x2 = np.array(x2, dtype=np.float32)
        return np.power(np.abs(x1), x2)


def _sigmoid(x1):
    """Special case of logistic function to transform to probabilities."""
    with np.errstate(over='ignore', under='ignore'):
        return 1. / (1. + np.exp(-x1))


OPERATOR_ADD = {
    "func": operator.add,
    "format_tf": 'tf.add({}, {})',
    "arity": 2,
    "symbol": "+",
    "format_str": "({} + {})",
    "format_result": "({} + {})",
    "commutative": True,
}

OPERATOR_SUB = {
    "func": operator.sub,
    "format_tf": 'tf.subtract({}, {})',
    "arity": 2,
    "symbol": "-",
    "format_str": "({} - {})",
    "format_result": "({} - {})",
    "commutative": False
}

OPERATOR_MUL = {
    "func": _protected_mul,
    "format_tf": 'tf.multiply({}, {})',
    "arity": 2,
    "symbol": "*",
    "format_str": "({} * {})",
    "format_result": "({} * {})",
    "commutative": True
}

OPERATOR_DIV = {
    "func": _protected_division,
    "format_tf": 'tf.divide({}, {})',
    "arity": 2,
    "symbol": "/",
    "format_str": "({} / {})",
    "format_result": "({} / {})",
    "commutative": False
}

OPERATOR_INV = {
    "func": _protected_inverse,
    "format_tf": 'tf.pow({}, -1)',
    "arity": 1,
    "symbol": "1/",
    "format_str": "(1 / {})",
    "format_result": "(1 / {})",
    "commutative": False
}

OPERATOR_NEG = {
    "func": operator.neg,
    "format_tf": 'tf.negative({})',
    "arity": 1,
    "symbol": "-",
    "format_str": "-({})",
    "format_result": "-({})",
    "commutative": False
}

OPERATOR_ABS = {
    "func": np.abs,
    "format_tf": 'tf.abs({})',
    "arity": 1,
    "symbol": "Abs",
    "format_str": "Abs({})",
    "format_result": "Abs({})",
    "commutative": False
}

OPERATOR_SIN = {
    "func": np.sin,
    "format_tf": 'tf.sin({})',
    "arity": 1,
    "symbol": "sin",
    "format_str": "sin({})",
    "format_result": "sin({})",
    "commutative": False
}

OPERATOR_COS = {
    "func": np.cos,
    "format_tf": 'tf.cos({})',
    "arity": 1,
    "symbol": "cos",
    "format_str": "cos({})",
    "format_result": "cos({})",
    "commutative": False
}

OPERATOR_LOG = {
    "func": _protected_log,
    "format_tf": 'tf.math.log({})',
    "arity": 1,
    "symbol": "log",
    "format_str": "log(Abs({}))",
    "format_result": "log(Abs({}))",
    "commutative": False
}

OPERATOR_EXP = {
    "func": _protected_exp,
    "format_tf": 'tf.exp({})',
    "arity": 1,
    "symbol": "exp",
    "format_str": "exp({})",
    "format_result": "exp({})",
    "commutative": False
}

OPERATOR_POW = {
    "func": _protected_pow,
    "format_tf": 'tf.pow({}, {})',
    "arity": 2,
    "symbol": "^",
    "format_str": "(Abs({}) ** {})",
    "format_result": "(Abs({}) ** {})",
    "commutative": False
}

OPERATOR_SQRT = {
    "func": _protected_sqrt,
    "format_tf": 'tf.sqrt({})',
    "arity": 1,
    "symbol": "sqrt",
    "format_str": "sqrt(Abs({}))",
    "format_result": "sqrt(Abs({}))",
    "commutative": False}

OPERATOR_MAX = {
    "func": np.maximum,
    "format_tf": 'tf.maximum({}, {})',
    "arity": 2,
    "symbol": "max",
    "format_str": "Max({}, {})",
    "format_result": "Max({}, {})",
    "commutative": True
}

OPERATOR_MIN = {
    "func": np.minimum,
    "format_tf": 'tf.minimum({}, {})',
    "arity": 2,
    "symbol": "min",
    "format_str": "Min({}, {})",
    "format_result": "Min({}, {})",
    "commutative": True
}

OPERATOR_SIGMOID = {
    "func": _sigmoid,
    "format_tf": 'tf.sigmoid({})',
    "arity": 1,
    "symbol": "sigmoid",
    "format_str": "sigmoid({})",
    "format_result": "sigmoid({})",
    "commutative": False
}

# LOGIC OPERATORS


def _logic_and(x1, x2):
    return x1 and x2


OPERATOR_LOGIC_AND = {
    "func": _logic_and,
    "format_tf": 'tf.math.logical_and({}, {})',
    "arity": 2,
    "symbol": "and",
    "format_str": "and({}, {})",
    "format_result": "and({}, {})",
    "commutative": True
}


def _logic_or(x1, x2):
    return x1 or x2


OPERATOR_LOGIC_OR = {
    "func": _logic_or,
    "format_tf": 'tf.math.logical_or({}, {})',
    "arity": 2,
    "symbol": "or",
    "format_str": "or({}, {})",
    "format_result": "or({}, {})",
    "commutative": True
}


def _logic_xor(x1):
    return not x1


OPERATOR_LOGIC_XOR = {
    "func": _logic_xor,
    "format_tf": 'tf.math.logical_xor({}, {})',
    "arity": 2,
    "symbol": "xor",
    "format_str": "xor({}, {})",
    "format_result": "xor({}, {})",
    "commutative": True
}


def _logic_not(x1):
    return not x1


OPERATOR_LOGIC_NOT = {
    "func": _logic_not,
    "format_tf": 'tf.math.logical_not({})',
    "arity": 1,
    "symbol": "not",
    "format_str": "not({})",
    "format_result": "not({})",
    "commutative": False
}


def _equal(x1, x2):
    return x1 == x2


OPERATOR_EQUAL_THAN = {
    "func": _equal,
    "format_tf": 'tf.Equal({}, {})',
    "arity": 2,
    "symbol": "==",
    "format_str": "equal({}, {})",
    "format_result": "equal({}, {})",
    "commutative": True
}


def _less_than(x1, x2):
    return x1 < x2


OPERATOR_LESS_THAN = {
    "func": _less_than,
    "format_tf": 'tf.Less({}, {})',
    "arity": 2,
    "symbol": "<",
    "format_str": "less({}, {})",
    "format_result": "less({}, {})",
    "commutative": False
}


def _less_equal_than(x1, x2):
    return x1 <= x2


OPERATOR_LESS_EQUAL_THAN = {
    "func": _less_equal_than,
    "format_tf": 'tf.LessEqual({}, {})',
    "arity": 2,
    "symbol": "<=",
    "format_str": "lessEqual({}, {})",
    "format_result": "lessEqual({}, {})",
    "commutative": False
}


def _greater_than(x1, x2):
    return x1 <= x2


OPERATOR_GREATER_THAN = {
    "func": _greater_than,
    "format_tf": 'tf.Greater({}, {})',
    "arity": 2,
    "symbol": ">",
    "format_str": "less({}, {})",
    "format_result": "less({}, {})",
    "commutative": False
}


def _greater_equal_than(x1, x2):
    return x1 <= x2


OPERATOR_GREATER_EQUAL_THAN = {
    "func": _greater_equal_than,
    "format_tf": 'tf.GreaterEqual({}, {})',
    "arity": 2,
    "symbol": ">=",
    "format_str": "lessEqual({}, {})",
    "format_result": "lessEqual({}, {})",
    "commutative": False
}