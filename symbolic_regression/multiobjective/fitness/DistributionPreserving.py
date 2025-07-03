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




class BinningDiversity(BaseFitness):

    def __init__(self, grade: str, method:str, N:int, **kwargs) -> None:
        """
        Calcola la diversità di binning tra l'output di un programma e la colonna 'IC_grade' di un dataset.

        Args:
            grade (str): MJ grade
            method (str): 'quantile' o 'optimal' per il tipo di binning
            N (int): numero di bin

        Returns:
            float: diversità media normalizzata tra i due binning
        """
        self.grade = grade
        self.method = method
        self.N = N
        super().__init__(**kwargs)

    def evaluate(self, program: Program, data: pd.DataFrame, validation: bool = False, pred: pd.DataFrame = None, inject: Dict = dict()) -> float:

        pred = inject.get('pred', pred)

        if pred is None:

            if not program.is_valid:
                return np.nan

            if not validation:
                program = self.optimize(program=program, data=data)

            try:
                pred = np.array(program.evaluate(data))
            except KeyError:
                return np.inf

        try:
            # Calcola il risultato del programma e assicurati che sia una Series 1D
            pred = pred.squeeze()

            target = data[self.grade].squeeze()

            if self.method == 'quantile':
                res_binned = pd.qcut(pred, q=self.N, labels=False, duplicates='drop') + 1
                grade_binned = pd.qcut(target, q=self.N, labels=False, duplicates='drop') + 1

            elif self.method == 'optimal':
                res_binned = assign_bins(pred, optimal_histogram(pred, self.N))
                grade_binned = assign_bins(target, optimal_histogram(target, self.N))

            else:
                raise ValueError("method must be either 'quantile' or 'optimal'")

            diversities = abs((res_binned - grade_binned) / self.N)
            return diversities.mean()
        except:
            return np.nan





#compute_sse calcola l'errore quadratico del vettore colonna 'data' dal valore start al valore end (start < end devono essere due valore di 'data'), 
# prima ne calcola la media e poi somma i quadrati delle differenze elemento-media
def compute_sse(data, start, end):
    subset = data[(data >= start) & (data <= end)]
    avg_val = np.mean(subset)
    sse = np.sum((subset - avg_val) ** 2)
    return sse

#optimal_histogram divide 'data' (colonna) in B bucket minimizzando SSE, l'output sarà una lista di B tuple
def optimal_histogram(data, B):
    data = np.sort(data) #ordino tutta la colonna
    n = len(data)
    dp = np.zeros((B + 1, n)) #inizializzo la matrice dei costi SSE al variare dell'indice che scelgo per tagliare
    #dp[k][j]: costo minimo per suddividere i primi j elementi in k bucket.
    partitions = np.zeros((B + 1, n), dtype=int) #inizializzo la matrice dei punti di separazione
    #partitions[k][j]: indice (rispetto al vettore colonna) del punto in cui terminare il bucket precedente quando si vogliono k bucket fino all'indice j.

    for j in range(n):
        dp[1][j] = compute_sse(data, data[0], data[j]) #inizializzo la prima riga, k=1 quindi 1 bucket unico dall'inizio alla j-esima e salvo i costi (SSE)

    for k in range(2, B + 1):
        for j in range(1, n):
            dp[k][j] = float('inf') #inizializzo il costo a infinito
            for i in range(j): #scorro tutti gli indici fino al j-esimo 
                cost = dp[k - 1][i] + compute_sse(data, data[i], data[j]) #aggiorno il costo (dp[k - 1][i] è il costo minimo per suddividere i primi i elementi in k-1 bucket) 
                #quindi aggiungo l'errore dell'intervallo dall'indice i a j
                if cost < dp[k][j]: 
                    dp[k][j] = cost #aggiorno mano mano se il costo scende, alla fine ottengo il costo minimo
                    partitions[k][j] = i  #salvo l'indice per dividere in maniera ottimale la colonna fino al j-esimo indice in k buckets
    buckets = []
    idx = n - 1 #parto dalla fine
    for k in range(B, 0, -1): 
        start = data[partitions[k][idx]] #indici iniziali dei bucket (inizia dall'ultimo) (nota: questi poi diventano i finali con l'ultimo aggiornamento)
        #questo perchè parto volendo dividere tutto in B buckets, poi tolgo l'ultimo intervallo, a quel punto il resto voglio dividerlo in B-1 etc
        end = data[idx] #indici finali dei bucket
        buckets.append((start, end)) #creo la lista di 2-uple con (valore iniziale,valore finale) 
        idx = partitions[k][idx]

    buckets.reverse() #riordino le tuple per averle dalla prima all'ultima in ordine 
    return buckets 


def assign_bins(data, buckets):
    bin_labels = np.zeros(len(data), dtype=int) #inizializzo la colonna categorizzata
    for i, (start, end) in enumerate(buckets):
        bin_labels[(data >= start) & (data <= end)] = i + 1 #se il valore rientra nell'intervallo dell'i-esimo indice della lista bucket allora assegno
        # l'indice i+1 (perchè parte da 0)
    return bin_labels

def binning_dataset(df, B): #serve solo per generalizzare ad un dataset (df) con più colonne
    binned_df = pd.DataFrame()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]): #nel caso ci siano colonne non numeriche distingue i casi categorizzando solo quelle numeriche
            data = df[col].dropna().values #toglie i nan nel caso ci siano
            buckets = optimal_histogram(data, B)
            binned_df[col] = assign_bins(df[col].values, buckets)
            #per plottare
            #plt.hist(data, bins=30, alpha=0.5, color='gray', label=f'Distribuzione originale {col}')
            #for i, (start, end) in enumerate(buckets):
                #plt.axvline(start, color='red', linestyle='--', label=f"Inizio bucket {i + 1}" if i == 0 else None)
                #plt.axvline(end, color='blue', linestyle='--', label=f"Fine bucket {i + 1}" if i == len(buckets) - 1 else None)
                #plt.fill_betweenx([0, plt.ylim()[1]], start, end, alpha=0.2, label=f"Bucket {i + 1}")

            #plt.xlabel(col)
            #plt.ylabel('Frequenza')
            #plt.title(f'Istogramma con Bucket Ottimali per {col}')
            #plt.legend()
            #plt.show()
        else:
            binned_df[col] = df[col]
    return binned_df