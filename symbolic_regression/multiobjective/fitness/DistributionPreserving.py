from typing import Dict
import numpy as np
import pandas as pd
from astropy import stats

from symbolic_regression.multiobjective.fitness.Base import BaseFitness
from symbolic_regression.Program import Program


class Average_Wasserstein(BaseFitness):

    def __init__(self, F_ys:list, **kwargs) -> None:
        """ This method requires the following arguments:
        - data: pd.DataFrame
        - target: str
        - weights: str  # will be calculated, do not provide!
        - bins: int

        """
        self.F_ys=F_ys
        super().__init__(**kwargs)

    def evaluate(self, program: Program, data: pd.DataFrame, validation: bool = False, pred=None,  inject: Dict = dict()) -> float:

        if not hasattr(self, 'F_ys'):
            self.F_y =[get_cumulant_hist(data=data, target=t, bins=self.bins) for t in self.target]

        features = program.features

        try:
            y_pred = np.array(program.evaluate(data[features]))
        except KeyError:
            return np.inf

        try:
            rescaled_y_pred = (y_pred-np.min(y_pred)) / \
                    (np.max(y_pred)-np.min(y_pred))
            wasserstein_list = []
            
            for F_y in self.F_ys:
                try:
                    # we add -1 so that wasserstein distance belongs to [0,1]
                    dy = 1./(F_y.shape[0]-1)               
                    # compute density function histogram based on target optimal one
                    pd_y_pred_grid, _ = stats.histogram(
                        rescaled_y_pred, bins=F_y.shape[0], density=True)
                    # compute optimal cumulative histogram
                    F_y_pred = np.sum(dy*pd_y_pred_grid *
                                      np.tril(np.ones(pd_y_pred_grid.size), 0), 1)       
                except:
                    F_y_pred = np.ones_like(F_y)            
                wasserstein_list.append(dy*np.sum(np.abs(F_y_pred-F_y)))
                
            return sum(wasserstein_list) / len(wasserstein_list)         
        except:            
            return np.inf



class Wasserstein(BaseFitness):

    def __init__(self, **kwargs) -> None:
        """ This method requires the following arguments:
        - data: pd.DataFrame
        - target: str
        - weights: str  # will be calculated, do not provide!
        - bins: int

        """
        super().__init__(**kwargs)

    def evaluate(self, program: Program, data: pd.DataFrame, validation: bool = False, pred: pd.DataFrame = None, inject: Dict = dict()) -> float:

        pred = inject.get('pred', pred)

        if pred is None:
            if not hasattr(self, 'F_y'):
                self.F_y = get_cumulant_hist(
                    data=data, target=self.target, bins=self.bins)

            features = program.features

            try:
                pred = np.array(program.evaluate(data[features]))
            except KeyError:
                return np.inf

        # we add -1 so that wasserstein distance belongs to [0,1]
        dy = 1./(self.F_y.shape[0]-1)

        # rescale between [0,1]
        try:
            rescaled_y_pred = (pred-np.min(pred)) / \
                (np.max(pred)-np.min(pred))
            # compute density function histogram based on target optimal one
            pd_y_pred_grid, _ = stats.histogram(
                rescaled_y_pred, bins=self.F_y.shape[0], density=True)
            # compute optimal cumulative histogram
            F_y_pred = np.sum(dy*pd_y_pred_grid *
                              np.tril(np.ones(pd_y_pred_grid.size), 0), 1)
        except:
            F_y_pred = np.ones_like(self.F_y)

        return dy*np.sum(np.abs(F_y_pred-self.F_y))


def get_cumulant_hist(data: pd.DataFrame, target: str, bins: int = None) -> np.array:

    y_true = np.array(data[target])

    # rescale
    rescaled_y_true = (y_true-np.min(y_true)) / \
        (np.max(y_true)-np.min(y_true))

    # compute optimal density function histogram
    if not bins:
        pd_y_true_grid, y_grid = stats.histogram(
            rescaled_y_true, bins='knuth', density=True)
    else:
        pd_y_true_grid, y_grid = stats.histogram(
            rescaled_y_true, bins=bins, density=True)

    # compute grid steps
    dy = y_grid[1]-y_grid[0]

    # compute optimal cumulative histogram
    F_y = np.sum(dy*pd_y_true_grid *
                 np.tril(np.ones(pd_y_true_grid.size), 0), 1)

    return F_y
