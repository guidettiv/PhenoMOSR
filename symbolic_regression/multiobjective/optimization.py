import traceback
import warnings
from typing import Union

import numpy as np
import pandas as pd
import sympy as sym
import copy
from scipy.optimize import minimize
from sympy.utilities.lambdify import lambdify
from warnings import catch_warnings
from warnings import simplefilter
from sklearn.gaussian_process import GaussianProcessRegressor


from symbolic_regression.multiobjective.fitness.Regression import \
    create_regression_weights
from symbolic_regression.preprocessing import check_assumptions
warnings.filterwarnings("ignore")


def SGD(program, data: Union[dict, pd.Series, pd.DataFrame], target: str, weights: str, constants_optimization_conf: dict, task: str, bootstrap: bool = False):
    '''
    Stochastic Gradient Descent with analytic derivatives

    Args:
        - program: Program
            Program to be optimized
        - data: dict, pd.Series, pd.DataFrame
            Data to be used for optimization
        - target: str
            Name of the target column
        - weights: str
            Name of the weights column
        - constants_optimization_conf: dict
            Dictionary with the following
            - learning_rate: float
                Learning rate for the optimization
            - batch_size: int
                Batch size for the optimization
            - epochs: int
                Number of epochs for the optimization
            - gradient_clip: float
                Gradient clipping value
            - l1_param: float
                L1 regularization parameter
            - l2_param: float
                L2 regularization parameter
        - task: str
            Task to be performed. Can be 'regression' or 'classification'
        - bootstrap: bool
            When bootstrapping is used, the weights are recalculated each time

    Returns:
        - list
            List of the optimized constants
        - list
            List of the loss values
        - list
            List of the accuracy values
    '''
    if task == 'regression:cox':
        batch_size = data.shape[0]
        status = constants_optimization_conf['status']
        unique_target = np.sort(
            data[target].loc[data[status] == True].unique())
        powers = [len(np.where(data[target] == unique_target[el])[0])
                  for el in range(len(unique_target))]
        RJs_indices = [np.where(data[target] >= unique_target[el])[
            0] for el in range(len(unique_target))]
        DJs_indices = [np.where((data[target] == unique_target[el]) *
                                (data[status] == True))[0] for el in range(len(unique_target))]
    else:
        batch_size = constants_optimization_conf['batch_size']

    learning_rate = constants_optimization_conf['learning_rate']
    epochs = constants_optimization_conf['epochs']
    gradient_clip = constants_optimization_conf.get('gradient_clip', None)
    l1_param = constants_optimization_conf.get('l1_param', 0)
    l2_param = constants_optimization_conf.get('l2_param', 0)

    if not program.is_valid:  # No constants in program
        return [], [], []

    n_features = len(program.features)
    constants = np.array(
        [item.feature for item in program.get_constants(return_objects=True)])
    n_constants = constants.size

    if n_constants == 0:  # No constants in program
        return [], [], []

    # Initialize symbols for variables and constants
    x_sym = ''
    for f in program.features:
        x_sym += f'{f},'
    x_sym = sym.symbols(x_sym, real=True)
    c_sym = sym.symbols('c0:{}'.format(n_constants),real=True)

    # Initialize ground truth and data arrays
    y_true = np.reshape(data[target].to_numpy(), (data[target].shape[0], 1))
    X_data = data[program.features].to_numpy()

    if weights:
        if bootstrap:
            if task == 'regression:wmse' or task == 'regression:wrrmse':
                w = create_regression_weights(
                    data=data, target=target, bins=10)
            elif task == 'binary:logistic':
                w = np.where(y_true == 1, 1./(2*y_true.mean()),
                             1./(2*(1-y_true.mean())))
            w = np.reshape(w, (w.shape[0], 1))
        else:
            w = np.reshape(data[weights].to_numpy(),
                           (data[weights].shape[0], 1))
    else:
        w = np.ones_like(y_true)

    # convert program render into sympy formula (symplify?)
    p_sym = program.program.render(format_diff=True)

    # compute program analytic gradients with respect to the constants to be optimized
    grad = []
    for i in range(n_constants):
        grad.append(sym.diff(p_sym, f'c{i}'))

    # define gradient and program python functions from sympy object
    try:
        pyf_grad = lambdify([x_sym, c_sym], grad)
        pyf_prog = lambdify([x_sym, c_sym], p_sym)
    except KeyError:  # When the function doesn't have sense
        return [], [], []

    # Define batches
    n_batches = int(X_data.shape[0] / batch_size)
    X_batch = np.array_split(X_data, n_batches, 0)
    y_batch = np.array_split(y_true, n_batches, 0)
    w_batch = np.array_split(w, n_batches, 0)

    log, loss = [], []  # lists to store learning process

    # initialize variance
    var = 0.

    for _ in range(epochs):
        for i in range(n_batches):
            split_X_batch = np.split(X_batch[i], n_features, 1)
            split_c_batch = np.split(
                constants*np.ones_like(y_batch[i]), n_constants, 1)

            # Define current batch weights, and compute numerical values of pyf_grad pyf_prog
            y_pred = pyf_prog(tuple(split_X_batch), tuple(split_c_batch))
            num_grad = pyf_grad(tuple(split_X_batch), tuple(split_c_batch))

            if task == 'regression:wmse':
                av_loss = np.nanmean(w_batch[i] * (y_pred - y_batch[i])**2)
                av_grad = np.array([
                    np.nanmean(2. * w_batch[i] * (y_pred - y_batch[i]) * g)
                    for g in num_grad
                ])
            elif task == 'regression:wrrmse':
                y_av = np.mean(y_batch[i]*w_batch[i])+1e-20

                sq_term = np.sqrt(np.nanmean(
                    w_batch[i] * (y_pred - y_batch[i])**2))
                av_loss = sq_term*100./y_av
                av_grad = np.array(
                    [100./(y_av*sq_term) * np.nanmean(w_batch[i] * (y_pred - y_batch[i]) * g) for g in num_grad])

            elif task == 'binary:logistic':
                # compute average loss
                # w=np.where(y_batch[i]==1, 1./(2*y_batch[i].mean()),  1./(2*(1-y_batch[i].mean())))
                # av_loss=np.nanmean(-w*y_batch[i]*np.log(y_pred+1e-20)-w*(1.-y_batch[i])*np.log(1.-y_pred+1e-20))
                sigma = 1. / (1. + np.exp(-y_pred)
                              )  # numerical value of sigmoid(program)
                av_loss = np.nanmean(
                    -w_batch[i] *
                    (y_batch[i] * np.log(sigma + 1e-20) +
                     (1. - y_batch[i]) * np.log(1. - sigma + 1e-20)))
                # compute average gradients
                av_grad = np.array([
                    np.nanmean(w_batch[i] * (sigma - y_batch[i]) * g)
                    for g in num_grad
                ])
            elif task == 'regression:cox':
                DFs = [np.sum(y_pred[els]) for els in DJs_indices]
                MEs = [np.mean(np.exp(y_pred)[els]) for els in DJs_indices]
                REs = [np.sum(np.exp(y_pred)[els]) for els in RJs_indices]
                EGs = [g*np.exp(y_pred) for g in num_grad]
                MEGs = [np.array([np.mean(EG[els]) for EG in EGs])
                        for els in DJs_indices]
                REGs = [np.array([np.sum(EG[els]) for EG in EGs])
                        for els in RJs_indices]
                DGs = [np.array([np.sum((g*np.ones_like(y_pred))[els])
                                for g in num_grad]) for els in DJs_indices]
                F_TIDES = [np.sum(np.log((REs[el] -
                                         np.expand_dims(np.arange(powers[el]), 1)*MEs[el]))) for el in range(len(powers))]
                av_loss = - \
                    np.sum(np.array([DFs[el]-F_TIDES[el]
                           for el in range(len(powers))]))
                TIDEs = [np.sum((REGs[el]-np.expand_dims(np.arange(powers[el]), 1) * MEGs[el]) /
                                (REs[el]-np.expand_dims(np.arange(powers[el]), 1)*MEs[el]), 0) for el in range(len(powers))]
                av_grad = -np.sum(np.array([DGs[el]-TIDEs[el]
                                  for el in range(len(powers))]), 0)

            # try with new constants if loss is nan
            if np.isnan(av_loss):
                var += 0.2
                constants = np.random.normal(0.0, var, constants.shape)

            norm_grad = np.linalg.norm(av_grad)
            if gradient_clip and (norm_grad > 1.):  # normalize gradients
                av_grad = av_grad / (norm_grad + 1e-20)

            # Updating constants
            constants -= learning_rate * av_grad + 2 * learning_rate * l2_param * \
                constants + learning_rate * l1_param * np.sign(constants)

        log.append(list(constants))
        loss.append(av_loss)

    return constants, loss, log


def ADAM(program, data: Union[dict, pd.Series, pd.DataFrame], target: str, weights: str, constants_optimization_conf: dict, task: str, bootstrap: bool = False):
    ''' ADAM with analytic derivatives

    Args:
        - program: Program
            The program to optimize
        - data: dict, pd.Series, pd.DataFrame
            The data to fit the program
        - target: str
            The target column name
        - weights: str
            The weights column name
        - constants_optimization_conf: dict
            Dictionary with the following
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

        - task: str
            The task to optimize
        - bootstrap: bool
            When bootstrapping is used, the weights are recalculated each time

    Returns:
        - constants: np.array
            The optimized constants
        - loss: list
            The loss at each epoch
        - log: list
            The constants at each epoch
    '''
    if task == 'regression:cox':
        batch_size = data.shape[0]
        status = constants_optimization_conf['status']
        unique_target = np.sort(
            data[target].loc[data[status] == True].unique())
        powers = [len(np.where(data[target] == unique_target[el])[0])
                  for el in range(len(unique_target))]
        RJs_indices = [np.where(data[target] >= unique_target[el])[
            0] for el in range(len(unique_target))]
        DJs_indices = [np.where((data[target] == unique_target[el]) *
                                (data[status] == True))[0] for el in range(len(unique_target))]
    else:
        batch_size = constants_optimization_conf['batch_size']

    learning_rate = constants_optimization_conf['learning_rate']
    epochs = constants_optimization_conf['epochs']
    gradient_clip = constants_optimization_conf['gradient_clip']
    beta_1 = constants_optimization_conf['beta_1']
    beta_2 = constants_optimization_conf['beta_2']
    epsilon = constants_optimization_conf['epsilon']
    l1_param = constants_optimization_conf.get('l1_param', 0)
    l2_param = constants_optimization_conf.get('l2_param', 0)

    if not program.is_valid:  # No constants in program
        return [], [], []

    n_features = len(program.features)
    constants = np.array(
        [item.feature for item in program.get_constants(return_objects=True)])
    n_constants = constants.size

    if n_constants == 0:  # No constants in program
        return [], [], []

    # Initialize symbols for variables and constants
    x_sym = ''
    for f in program.features:
        x_sym += f'{f},'
    x_sym = sym.symbols(x_sym, real=True)
    c_sym = sym.symbols('c0:{}'.format(n_constants),real=True)

    # Initialize ground truth and data arrays
    y_true = np.reshape(data[target].to_numpy(), (data[target].shape[0], 1))
    X_data = data[program.features].to_numpy()

    if weights:
        if bootstrap:
            if task == 'regression:wmse' or task == 'regression:wrrmse':
                w = create_regression_weights(
                    data=data, target=target, bins=10)
            elif task == 'binary:logistic':
                w = np.where(y_true == 1, 1./(2*y_true.mean()),
                             1./(2*(1-y_true.mean())))
            w = np.reshape(w, (w.shape[0], 1))
        else:
            w = np.reshape(data[weights].to_numpy(),
                           (data[weights].shape[0], 1))
    else:
        w = np.ones_like(y_true)

    # convert program render into sympy formula (symplify?)
    p_sym = program.program.render(format_diff=True)

    # compute program analytic gradients with respect to the constants to be optimized
    grad = []
    try:
        for i in range(n_constants):
            grad.append(sym.diff(p_sym, f'c{i}'))
    except:
        return [], [], []

    # define gradient and program python functions from sympy object

    try:
        pyf_grad = lambdify([x_sym, c_sym], grad)
        pyf_prog = lambdify([x_sym, c_sym], p_sym)
    except:  # When the function doesn't have sense
        return [], [], []

    # Define batches
    n_batches = int(X_data.shape[0] / batch_size)
    X_batch = np.array_split(X_data, n_batches, 0)
    y_batch = np.array_split(y_true, n_batches, 0)
    w_batch = np.array_split(w, n_batches, 0)

    log, loss = [], []  # lists to store learning process

    # Initialize Adam variables
    m = 0
    v = 0
    t = 1
    var = 0

    for _ in range(epochs):
        for i in range(n_batches):

            split_X_batch = np.split(X_batch[i], n_features, 1)
            split_c_batch = np.split(
                constants*np.ones_like(y_batch[i]), n_constants, 1)

            # Define current batch weights, and compute numerical values of pyf_grad pyf_prog
            try:
                y_pred = pyf_prog(tuple(split_X_batch), tuple(split_c_batch))
                num_grad = pyf_grad(tuple(split_X_batch), tuple(split_c_batch))
            except KeyError:
                return [], [], []
            except ValueError:
                return [], [], []

            if task == 'regression:wmse':
                av_loss = np.nanmean(w_batch[i] * (y_pred - y_batch[i])**2)
                av_grad = np.array([
                    np.nanmean(2 * w_batch[i] * (y_pred - y_batch[i]) * g)
                    for g in num_grad
                ])

            elif task == 'regression:wrrmse':
                y_av = np.mean(y_batch[i]*w_batch[i])+1e-20

                sq_term = np.sqrt(np.nanmean(
                    w_batch[i] * (y_pred - y_batch[i])**2))
                av_loss = sq_term*100./y_av
                av_grad = np.array(
                    [100./(y_av*sq_term) * np.nanmean(w_batch[i] * (y_pred - y_batch[i]) * g) for g in num_grad])

            elif task == 'binary:logistic':
                # compute average loss

                # numerical value of sigmoid(program)
                sigma = 1. / (1. + np.exp(-y_pred))
                av_loss = np.nanmean(
                    -w_batch[i] *
                    (y_batch[i] * np.log(sigma + 1e-20) +
                     (1. - y_batch[i]) * np.log(1. - sigma + 1e-20)))
                # compute average gradients
                av_grad = np.array([
                    np.nanmean(w_batch[i] * (sigma - y_batch[i]) * g)
                    for g in num_grad
                ])
            elif task == 'regression:cox':
                DFs = [np.sum(y_pred[els]) for els in DJs_indices]
                MEs = [np.mean(np.exp(y_pred)[els]) for els in DJs_indices]
                REs = [np.sum(np.exp(y_pred)[els]) for els in RJs_indices]
                EGs = [g*np.exp(y_pred) for g in num_grad]
                MEGs = [np.array([np.mean(EG[els]) for EG in EGs])
                        for els in DJs_indices]
                REGs = [np.array([np.sum(EG[els]) for EG in EGs])
                        for els in RJs_indices]
                DGs = [np.array([np.sum((g*np.ones_like(y_pred))[els])
                                for g in num_grad]) for els in DJs_indices]
                F_TIDES = [np.sum(np.log((REs[el] -
                                         np.expand_dims(np.arange(powers[el]), 1)*MEs[el]))) for el in range(len(powers))]
                av_loss = - \
                    np.sum(np.array([DFs[el]-F_TIDES[el]
                           for el in range(len(powers))]))
                TIDEs = [np.sum((REGs[el]-np.expand_dims(np.arange(powers[el]), 1) * MEGs[el]) /
                                (REs[el]-np.expand_dims(np.arange(powers[el]), 1)*MEs[el]), 0) for el in range(len(powers))]
                av_grad = -np.sum(np.array([DGs[el]-TIDEs[el]
                                  for el in range(len(powers))]), 0)

            # try with new constants if loss is nan
            if np.isnan(av_loss):
                var += 0.2
                constants = np.random.normal(0.0, var, constants.shape)

            norm_grad = np.linalg.norm(av_grad)
            if gradient_clip and (norm_grad > 1.):  # normalize gradients
                av_grad = av_grad / (norm_grad + 1e-20)

            # Updating momentum variables
            m = beta_1 * m + (1 - beta_1) * av_grad
            v = beta_2 * v + (1 - beta_2) * np.power(av_grad, 2)
            m_hat = m / (1 - np.power(beta_1, t))
            v_hat = v / (1 - np.power(beta_2, t))
            t += 1

            # Update constants
            constants -= learning_rate * m_hat / \
                (np.sqrt(v_hat) + epsilon) + 2 * learning_rate * l2_param * \
                constants + learning_rate * l1_param * np.sign(constants)

        log.append(list(constants))
        loss.append(av_loss)

    return constants, loss, log



def ADAM2FOLD(program, 
              data: Union[dict, pd.Series, pd.DataFrame], 
              target: list, 
              weights: list, 
              constants_optimization_conf: dict, 
              task: str, 
              bootstrap: bool = False):
    ''' ADAM with analytic derivatives for 2-fold programs

    Args:
        -program: Program
            The program to optimize
        -data: dict, pd.Series, pd.DataFrame
            The data to fit
        -target: list
            The targets to fit
        -weights: list
            The weights to fit
        -constants_optimization_conf: dict
            Dictionary with the following
            - learning_rate: float
                The learning rate
            - batch_size: int
                The batch size
            - epochs: int
                The number of epochs
            - gradient_clip: bool
                Whether to clip the gradients
            - beta_1: float
                The beta_1 parameter for Adam
            - beta_2: float
                The beta_2 parameter for Adam
            - epsilon: float
                The epsilon parameter for Adam
        -task: str
            The task to optimize
        -bootstrap: bool
            When bootstrapping is used, the weights are recalculated each time

    Returns:
        -constants: list
            The optimized constants
        -loss: list
            The loss at each epoch
        -log: list
            The constants at each epoch
    '''
    # print('using ADAM2FOLD')
    learning_rate = constants_optimization_conf['learning_rate']
    batch_size = constants_optimization_conf['batch_size']
    epochs = constants_optimization_conf['epochs']
    gradient_clip = constants_optimization_conf['gradient_clip']
    beta_1 = constants_optimization_conf['beta_1']
    beta_2 = constants_optimization_conf['beta_2']
    epsilon = constants_optimization_conf['epsilon']

    if not program.is_valid:  # No constants in program
        return [], [], []

    n_features = len(program.features)
    constants = np.array(
        [item.feature for item in program.get_constants(return_objects=True)])
    n_constants = constants.size

    if n_constants == 0:  # No constants in program
        return [], [], []

    # Initialize symbols for variables and constants
    x_sym = ''
    for f in program.features:
        x_sym += f'{f},'
    x_sym = sym.symbols(x_sym, real=True)
    c_sym = sym.symbols('c0:{}'.format(n_constants),real=True)

    # Initialize ground truth and data arrays
    y_true_1 = np.reshape(data[target[0]].to_numpy(),
                          (data[target[0]].shape[0], 1))
    y_true_2 = np.reshape(data[target[1]].to_numpy(),
                          (data[target[1]].shape[0], 1))
    X_data = data[program.features].to_numpy()

    if weights:
        if bootstrap:
            if task == 'regression:wmse' or task == 'regression:wrrmse':
                w1 = create_regression_weights(
                    data=data, target=target[0], bins=10)
                w2 = create_regression_weights(
                    data=data, target=target[1], bins=10)
            elif task == 'binary:logistic':
                w1 = np.where(y_true_1 == 1, 1./(2*y_true_1.mean()),
                              1./(2*(1-y_true_1.mean())))
                w2 = np.where(y_true_2 == 1, 1./(2*y_true_2.mean()),
                              1./(2*(1-y_true_2.mean())))
            w1 = np.reshape(w1, (w1.shape[0], 1))
            w2 = np.reshape(w2, (w2.shape[0], 1))
        else:
            w1 = np.reshape(data[weights[0]].to_numpy(),
                            (data[weights[0]].shape[0], 1))
            w2 = np.reshape(data[weights[1]].to_numpy(),
                            (data[weights[1]].shape[0], 1))
    else:
        w1 = np.ones_like(y_true_1)
        w2 = np.ones_like(y_true_2)

    # convert program render into sympy formula (symplify?)
    p_sym = program.program.render(format_diff=True)

    # compute program analytic gradients with respect to the constants to be optimized
    grad = []
    for i in range(n_constants):
        grad.append(sym.diff(p_sym, f'c{i}'))

    # define gradient and program python functions from sympy object

    try:
        pyf_grad = lambdify([x_sym, c_sym], grad)
        pyf_prog = lambdify([x_sym, c_sym], p_sym)
    except KeyError:  # When the function doesn't have sense
        return [], [], []
    # Define batches
    n_batches = int(X_data.shape[0] / batch_size)
    X_batch = np.array_split(X_data, n_batches, 0)
    y1_batch = np.array_split(y_true_1, n_batches, 0)
    y2_batch = np.array_split(y_true_2, n_batches, 0)
    w1_batch = np.array_split(w1, n_batches, 0)
    w2_batch = np.array_split(w2, n_batches, 0)

    log, loss = [], []  # lists to store learning process

    # Initialize Adam variables
    m = 0
    v = 0
    t = 1
    var = 0

    samples = 100

    for _ in range(epochs):
        for i in range(n_batches):
            # sample lambdas from distribution
            lambda1 = np.random.uniform(low=0.0, high=1.0, size=(1, samples))

            split_X_batch = np.split(X_batch[i], n_features, 1)
            split_c_batch = np.split(
                constants*np.ones_like(y1_batch[i]), n_constants, 1)

            # Define current batch weights, and compute numerical values of pyf_grad pyf_prog
            y_pred = pyf_prog(tuple(split_X_batch), tuple(split_c_batch))
            num_grad = pyf_grad(tuple(split_X_batch), tuple(split_c_batch))

            if task == 'regression:wmse':  # (N,1)
                av_loss = np.nanmean(lambda1*(w1_batch[i] * (y_pred - y1_batch[i])**2)
                                     + (1-lambda1)*(w2_batch[i] * (y_pred - y2_batch[i])**2))
                av_grad = np.array([
                    np.nanmean(2 * (lambda1*(w1_batch[i]*(y_pred - y1_batch[i]))+ (1-lambda1)*(w2_batch[i]*(y_pred - y2_batch[i]))) * g) for g in num_grad
                ])

            # try with new constants if loss is nan
            if np.isnan(av_loss):
                var += 0.2
                constants = np.random.normal(0.0, var, constants.shape)

            norm_grad = np.linalg.norm(av_grad)
            if gradient_clip and (norm_grad > 1.):  # normalize gradients
                av_grad = av_grad / (norm_grad + 1e-20)

            # Updating momentum variables
            m = beta_1 * m + (1 - beta_1) * av_grad
            v = beta_2 * v + (1 - beta_2) * np.power(av_grad, 2)
            m_hat = m / (1 - np.power(beta_1, t))
            v_hat = v / (1 - np.power(beta_2, t))
            t += 1

            # Update constants
            constants -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        log.append(list(constants))
        loss.append(av_loss)

    return constants, loss, log


def SCIPY(program, 
          data: Union[dict, pd.Series, pd.DataFrame], 
          target: str, 
          weights: str, 
          constants_optimization_conf: dict, 
          task: str, 
          bootstrap: bool = False):
    ''' SCIPY library for optimization

    Args:
        - program: Program
            The program to optimize
        - data: dict, pd.Series, pd.DataFrame
            The data to fit the program
        - target: str
            The target column name
        - weights: str
            The weights column name
        - constants_optimization_conf: Not required

        - task: str
            The task to optimize
        - bootstrap: bool
            When bootstrapping is used, the weights are recalculated each time

    Returns:
        - constants: np.array
            The optimized constants
        - loss: list
            The loss at each epoch
        - log: list
            The constants at each epoch
    '''

    if not program.is_valid:  # No constants in program
        return [], [], []
    
    constants = np.array(
        [item.feature for item in program.get_constants(return_objects=True)])
    n_constants = constants.size

    if n_constants == 0:  # No constants in program
        return [], [], []
    
    y_true = np.reshape(data[target].to_numpy(), (data[target].shape[0], 1))
    X_data = data[program.features].to_numpy()

    pyf_prog = program.lambdify()

    if not pyf_prog:
        program._override_is_valid = False
        return constants, None, None
   
    if weights:
        if bootstrap:
            if task == 'regression:wmse' or task == 'regression:wrrmse':
                w = create_regression_weights(
                    data=data, target=target, bins=10)
            elif task == 'binary:logistic':
                w = np.where(y_true == 1, 1./(2*y_true.mean()),
                             1./(2*(1-y_true.mean())))
            w = np.reshape(w, (w.shape[0], 1))
        else:
            w = np.reshape(data[weights].to_numpy(),
                           (data[weights].shape[0], 1))
    else:
        w = np.ones_like(y_true)

    try:
        if task == 'regression:wmse' or task == 'regression:wrrmse':
            res = minimize(nll_min_regression, x0=constants, args=(
                y_true, X_data, pyf_prog, w), method='L-BFGS-B')
            constants = res.x
        
        elif task == 'binary:logistic':
            res = minimize(nll_min_binary, x0=constants, args=(
                y_true, X_data, pyf_prog, w), method='L-BFGS-B')
            constants = res.x
        
        elif task=='regression:cox':
            status=constants_optimization_conf['status']  
            event_indicators=data[status]
            event_times=data[target]
            res = minimize(nll_min_CoxBreslow, x0=constants, args=(
                y_true, X_data, pyf_prog, event_times,event_indicators), method='L-BFGS-B')
            constants = res.x

        elif task=='regression:cox_efron':
            status=constants_optimization_conf['status']  
            event_indicators=data[status]
            event_times=data[target]
            res = minimize(nll_min_CoxEfron, x0=constants, args=(
                y_true, X_data, pyf_prog, event_times,event_indicators), method='L-BFGS-B')
            constants = res.x

        elif task=='regression:finegray':
            status=constants_optimization_conf['status']  
            event_indicators=data[status]
            event_times=data[target]
            res = minimize(nll_min_FineGrayBreslow, x0=constants, args=(
                y_true, X_data, pyf_prog, event_times,event_indicators), method='L-BFGS-B')
            constants = res.x
    except ValueError:
        program._override_is_valid = False
        return constants, None, None
    
    return constants, None, None



def nll_min_regression(c, y, X, pyf_prog, weights=None):
    n_features = X.shape[1]
    n_constants = len(c)
    split_X = np.split(X, n_features, 1)
    split_c = np.split(c*np.ones_like(y), n_constants, 1)
    y_pred = pyf_prog(tuple(split_X), tuple(split_c))
    residual = (y-y_pred)

    if weights is not None:
        return np.mean(weights*residual**2)
    return np.mean(residual**2)

def nll_min_binary(c, y, X, pyf_prog, weights=None):
    n_features = X.shape[1]
    n_constants = len(c)
    split_X = np.split(X, n_features, 1)
    split_c = np.split(c*np.ones_like(y), n_constants, 1)
    y_pred = pyf_prog(tuple(split_X), tuple(split_c))
    sigma = 1. / (1. + np.exp(-y_pred))

    if weights is not None:
        return -np.mean(weights*(y*np.log(sigma+1e-20) + (1.-y)*np.log(1.-sigma+1e-20)))
    return -np.mean(y*np.log(sigma+1e-20) + (1.-y)*np.log(1.-sigma+1e-20))

def nll_min_CoxEfron(c, y, X, pyf_prog,event_times,event_indicators):
    n_features = X.shape[1]
    n_constants = len(c)
    split_X = np.split(X, n_features, 1)
    split_c = np.split(c*np.ones_like(y), n_constants, 1)
    y_pred = pyf_prog(tuple(split_X), tuple(split_c))

    unique_times, event_counts = np.unique(event_times[event_indicators == 1], return_counts=True)

    LogLikelihood = 0
    for t, d_k in zip(unique_times, event_counts):
        risk_set = np.where(event_times >= t)[0]
        event_set = np.where((event_times == t) & (event_indicators == 1))[0]
        
        sum_beta_X_i = np.sum(y_pred[event_set])
        sum_exp_beta_X_j = np.sum(np.exp(y_pred[risk_set]))
        
        adjusted_sum = 0
        for j in range(d_k):
            adjusted_sum +=  np.log(sum_exp_beta_X_j - j / d_k * np.sum(np.exp(y_pred[event_set])))
        
        LogLikelihood += sum_beta_X_i - adjusted_sum

    nll = -LogLikelihood
    return nll

def nll_min_CoxBreslow(c, y, X, pyf_prog,event_times,event_indicators):
    n_features = X.shape[1]
    n_constants = len(c)
    split_X = np.split(X, n_features, 1)
    split_c = np.split(c*np.ones_like(y), n_constants, 1)
    y_pred = pyf_prog(tuple(split_X), tuple(split_c))

    unique_times, event_counts = np.unique(event_times[event_indicators == 1], return_counts=True)

    LogLikelihood = 0
    for t, d_k in zip(unique_times, event_counts):
        risk_set = np.where(event_times >= t)[0]
        event_set = np.where((event_times == t) & (event_indicators == 1))[0]
        
        sum_fx_i = np.sum(y_pred[event_set])
        sum_exp_fx_j = np.sum(np.exp(y_pred[risk_set]))
        
        LogLikelihood += sum_fx_i - d_k * np.log(sum_exp_fx_j)
    nll = -LogLikelihood
    return nll

def nll_min_FineGrayBreslow(c, y, X, pyf_prog,times,indicators):
    n_features = X.shape[1]
    n_constants = len(c)
    split_X = np.split(X, n_features, 1)
    split_c = np.split(c*np.ones_like(y), n_constants, 1)
    y_pred = pyf_prog(tuple(split_X), tuple(split_c))

    # Ensure times and indicators are numpy arrays for indexing
    times = np.asarray(times)
    indicators = np.asarray(indicators)

    # Calculate weights using vectorized approach
    weights = calculate_weights(times, indicators)

    # Filter events and get unique event times
    unique_times, event_counts = np.unique(times[indicators==1.], return_counts=True)

    # Create weight mask based on event times and censoring status
    mask = (np.expand_dims((indicators != 2),-1))&(np.expand_dims(times,-1)<np.expand_dims(unique_times,0))
    weights[mask] = 0

    # Vectorized calculation for exp(pred) * weights
    A=np.exp(y_pred)
    A_w=(A*weights)
    risk_sets = A_w.sum(axis=0)

    log_likelihood = np.sum([
            np.sum(y_pred[times == t][indicators[times == t] == 1]) - d_k * np.log(risk_sets[i])
            for i, t, d_k in zip(range(len(unique_times)), unique_times, event_counts)
        ])
    nll = -log_likelihood

    return nll

def calculate_weights(times, indicators):
    """
    Calculate weights w_ij using the Kaplan-Meier estimator.

    times: array of observation times
    indicators: array of censoring/event indicators (0=censored, 1=event, 2=competing event)

    Returns:
    weights: matrix of weights w_ij
    """
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter()
    kmf.fit(times, event_observed=(indicators == 0))

    # Extract unique event times and survival function at these times
    unique_times = np.unique(times[indicators == 1])
    survival_function = kmf.survival_function_at_times(unique_times).values

    # Vectorized weight calculation
    weights = np.ones((len(times), len(unique_times)))

    # Use broadcasting for efficient calculation of weights
    times_broadcast = np.minimum(np.expand_dims(times,-1),np.expand_dims(unique_times,0))
    original_shape=times_broadcast.shape

    weights = np.reshape(survival_function,-1) / (np.reshape(kmf.survival_function_at_times(times_broadcast.flatten()),original_shape) + 1e-20)

    return weights


# surrogate or approximation for the objective function
def surrogate_model(model):
    def surrogate(x):
        # catch any warning generated when making a prediction
        with catch_warnings():
        # ignore generated warnings
            simplefilter("ignore")
        if len(np.array(x).shape)==1:
            x=np.expand_dims(np.array(x),0)
        return model.predict(x, return_std=False) ## metti un - se stai studiando il portafoglio
    return surrogate



def GaussProcess(program, 
             data: Union[dict, pd.Series, pd.DataFrame], 
             target: str, 
             weights: str, 
             constants_optimization_conf: dict, 
             task: str,
             bootstrap: bool = False):
    ''' Gaussian process with analytic derivatives
    DA SISTEMARE!!!
    Args:
        - program: Program
            The program to optimize
        - data: dict, pd.Series, pd.DataFrame
            The data to fit the program
        - target: str
        - weights: str
            Usless but kept to keep notation uniform
        - constants_optimization_conf: dict
            Dictionary with the following
            - learning_rate: float
                The learning rate
            - batch_size: int
                The batch size
            - epochs: int
                The number of epochs
        - task: str
            The task to optimize (Useless at the moment, useful when having more than one trading algorithm)

    Returns:
        - constants: np.array
            The optimized constants
        - loss: list
            The loss at each epoch
        - log: list
            The constants at each epoch
    '''
    
    iters=constants_optimization_conf['iters']
        
    if not program.is_valid:  # Not valid
        return [], [], []

    n_constants = len(program.get_constants())

    if n_constants == 0:  # No constants in program
        return [], [], []

    ## sample n=iters random arrays to be used as constants
    cmin,cmax=program.const_range
    constants_sampled=np.reshape(np.random.uniform(low=cmin, high=cmax,  size=iters*n_constants),(iters,n_constants))
    constants_sampled=[list(el) for el in list(constants_sampled)]

    hv_ftn = [ftn for ftn in program.fitness_functions if (ftn.minimize and (ftn.label != 'ModComplex') and (ftn.label != 'Complexity'))]
    neg_hv_list=[]
    for constants in constants_sampled:
        p=copy.deepcopy(program)
        p.set_constants(constants)
        hv_dims=[ftn.hypervolume_reference-ftn.evaluate(p,data,validation=True) for ftn in hv_ftn]
        neg_hv_list.append(-np.prod(hv_dims))
    assert len(constants_sampled) == len(neg_hv_list), "len of constants_sampled and hv_list should be the same"
    constants_sampled = np.array(constants_sampled)  # Convert to NumPy array (2D)
    neg_hv_list = np.array(neg_hv_list)  # Convert to NumPy array (1D)

    mask = ~np.isnan(neg_hv_list)  # Create a mask based on neg_hv_list
    # Apply mask to both neg_hv_list and constants_sampled
    neg_hv_list = neg_hv_list[mask]
    constants_sampled = constants_sampled[mask]

    # Convert back to lists
    neg_hv_list = neg_hv_list.tolist()
    constants_sampled = constants_sampled.tolist()

    new_constants=[]
    try:
        if len(neg_hv_list)>5:
            #try: 
            #identify our best guess based on sampled constants
            guess_constants=constants_sampled[np.nanargmin(neg_hv_list)]

            #once we collected more than n_constants*iter samples of constants, estimate new program constants with gaussian process
            model = GaussianProcessRegressor()
            # fit the model
            model.fit(np.array(constants_sampled), np.expand_dims(neg_hv_list,1))
            surrogate=surrogate_model(model)
            res=minimize(surrogate, np.array(guess_constants), method='BFGS')
            new_constants=list(res.x)
        return new_constants, None, None
    except ValueError:
        return new_constants, None, None

    


