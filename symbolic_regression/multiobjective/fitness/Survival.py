
import numpy as np
import pandas as pd

from symbolic_regression.multiobjective.fitness.Base import BaseFitness
from symbolic_regression.Program import Program

from typing import Dict
from lifelines import KaplanMeierFitter


class CoxEfron(BaseFitness):

    def __init__(self, **kwargs) -> None:
        """ This fitness requires the following arguments:

        - target: str
        - weights: str

        """
        super().__init__(**kwargs)

        assert hasattr(
            self, 'status'), "Status must be specified. In a Cox model, it is whether the event happened or not."

    def evaluate(self, program: Program, data: pd.DataFrame, validation: bool = False, pred=None, inject: Dict = dict()) -> float:

        if not program.is_valid:
            return np.nan

        if not validation:
            program = self.optimize(program=program, data=data)

        pred = program.evaluate(data=data)

        if hasattr(pred, 'to_numpy'):
            pred = pred.to_numpy()
        if hasattr(pred, 'shape') and pred.size != 1:
            pred = np.reshape(pred, (len(data), 1))
        elif isinstance(pred, float) or (hasattr(pred, 'shape') and pred.size == 1):
            pred = pred*np.ones((len(data), 1))

        assert pred.shape == (len(data), 1), "wrong shape of prediction array"
  
        event_indicators=data[self.status]
        event_times=data[self.target]

        unique_times, event_counts = np.unique(event_times[event_indicators == 1], return_counts=True)
    
        LogLikelihood = 0
        for t, d_k in zip(unique_times, event_counts):
            risk_set = np.where(event_times >= t)[0]
            event_set = np.where((event_times == t) & (event_indicators == 1))[0]
            
            sum_beta_X_i = np.sum(pred[event_set])
            sum_exp_beta_X_j = np.sum(np.exp(pred[risk_set]))
            
            adjusted_sum = 0
            for j in range(d_k):
                adjusted_sum +=  np.log(sum_exp_beta_X_j - j / d_k * np.sum(np.exp(pred[event_set])))
            
            LogLikelihood += sum_beta_X_i - adjusted_sum
        try:
            
            nll = -LogLikelihood
            return nll
        except TypeError:
            return np.inf
        except ValueError:
            return np.inf



class CoxBreslow(BaseFitness):

    def __init__(self, **kwargs) -> None:
        """ This fitness requires the following arguments:

        - target: str
        - weights: str

        """
        super().__init__(**kwargs)

        assert hasattr(
            self, 'status'), "Status must be specified. In a Cox model, it is whether the event happened or not."

    def evaluate(self, program: Program, data: pd.DataFrame, validation: bool = False, pred=None, inject: Dict = dict()) -> float:

        if not program.is_valid:
            return np.nan

        if not validation:
            program = self.optimize(program=program, data=data)

        pred = program.evaluate(data=data)

        if hasattr(pred, 'to_numpy'):
            pred = pred.to_numpy()
        if hasattr(pred, 'shape') and pred.size != 1:
            pred = np.reshape(pred, (len(data), 1))
        elif isinstance(pred, float) or (hasattr(pred, 'shape') and pred.size == 1):
            pred = pred*np.ones((len(data), 1))

        assert pred.shape == (len(data), 1), "wrong shape of prediction array"
  
        event_indicators=data[self.status]
        event_times=data[self.target]

        unique_times, event_counts = np.unique(event_times[event_indicators == 1], return_counts=True)
    
        LogLikelihood = 0
        for t, d_k in zip(unique_times, event_counts):
            risk_set = np.where(event_times >= t)[0]
            event_set = np.where((event_times == t) & (event_indicators == 1))[0]
            
            sum_fx_i = np.sum(pred[event_set])
            sum_exp_fx_j = np.sum(np.exp(pred[risk_set]))
            
            LogLikelihood += sum_fx_i - d_k * np.log(sum_exp_fx_j)
        try:
            #nconstants = len(program.get_constants())
            nll = -LogLikelihood
            #AIC = 2*(nconstants+nll)
            return nll
        except TypeError:
            return np.inf
        except ValueError:
            return np.inf
        


class FineGrayBreslow(BaseFitness):

    def __init__(self, **kwargs) -> None:
        """ This fitness requires the following arguments:

        - target: str
        - weights: str

        """
        super().__init__(**kwargs)
        assert hasattr(self, 'status'), "Status must be specified. In a Cox model, it is whether the event happened or not."
        
    def evaluate(self, program: Program, data: pd.DataFrame, validation: bool = False, pred=None, inject: Dict = dict()) -> float:

        if not program.is_valid:
            return np.nan

        if not validation:
            program = self.optimize(program=program, data=data)

        pred = program.evaluate(data=data)

        if hasattr(pred, 'to_numpy'):
            pred = pred.to_numpy()
        if hasattr(pred, 'shape') and pred.size != 1:
            pred = np.reshape(pred, (len(data), 1))
        elif isinstance(pred, float) or (hasattr(pred, 'shape') and pred.size == 1):
            pred = pred*np.ones((len(data), 1))

        assert pred.shape == (len(data), 1), "wrong shape of prediction array"

        # Ensure times and indicators are numpy arrays for indexing
        indicators = data[self.status].values
        times = data[self.target].values

        # Calculate weights using vectorized approach
        weights = FineGrayBreslow.calculate_weights(times, indicators)

        # Filter events and get unique event times
        unique_times, event_counts = np.unique(times[indicators==1.], return_counts=True)

        # Create weight mask based on event times and censoring status
        mask = (np.expand_dims((indicators != 2),-1))&(np.expand_dims(times,-1)<np.expand_dims(unique_times,0))
        weights[mask] = 0

        # Vectorized calculation for exp(pred) * weights
        A=np.exp(pred)
        A_w=(A*weights)

        risk_sets = A_w.sum(axis=0)

        log_likelihood = np.sum([
                np.sum(pred[times == t][indicators[times == t] == 1]) - d_k * np.log(risk_sets[i])
                for i, t, d_k in zip(range(len(unique_times)), unique_times, event_counts)
            ])

        # AIC calculation
        try:
            #nconstants = len(program.get_constants())
            nll = -log_likelihood
            #AIC = 2 * (nconstants + nll)
            return nll
        except (TypeError, ValueError):
            return np.inf
    
    @staticmethod
    def calculate_weights(times, indicators):
        """
        Calculate weights w_ij using the Kaplan-Meier estimator.

        times: array of observation times
        indicators: array of censoring/event indicators (0=censored, 1=event, 2=competing event)

        Returns:
        weights: matrix of weights w_ij
        """
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

