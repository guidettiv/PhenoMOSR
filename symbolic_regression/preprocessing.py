import numpy as np
import pandas as pd
from pyHSICLasso import HSICLasso



def check_assumptions(df):
    assumptions = {
        "real": lambda x: np.isreal(x),
        "finite": lambda x: np.isfinite(x),
        "positive": lambda x: x > 0,
        "negative": lambda x: x < 0,
        "nonpositive": lambda x: x <= 0,
        "nonnegative": lambda x: x >= 0,
    }

    result = {}
    for col in df.columns:
        col_data = df[col]
        col_result = {}
        for name, func in assumptions.items():
            try:
                passed = func(col_data)
                if isinstance(passed, pd.Series):
                    # If all values in the column pass the test (ignoring NaNs)
                    col_result[name] = bool(passed.dropna().all())
                else:
                    col_result[name] = bool(passed.all())
            except Exception:
                col_result[name] = False  # Handle errors gracefully
                print('raised exception in feature assumption')
        result[col] = col_result

    return result


def select_feature_probabilities_regression(data,features,target):
    
    from sklearn.ensemble import RandomForestRegressor
    regr = RandomForestRegressor(max_depth=4, random_state=42)
    regr.fit(data[features], data[target])
    scores=regr.feature_importances_
    scores=scores/np.sum(scores)
    scores = np.round(scores, decimals=4).astype(np.float64)
    id_max=np.argmax(scores)
    max_val=scores[id_max]
    scores[id_max]=1-(scores.sum()-max_val)
    return scores.tolist()


def select_feature_probabilities_regression_HSIC(data,features,target):
    hsic_lasso = HSICLasso()
    X = data[features].values
    Y = data[target].values
    hsic_lasso.input(X,Y,featname=features)
    hsic_lasso.regression(len(features)-1)


    sorted_scores=hsic_lasso.get_index_score()
    sorted_features=hsic_lasso.get_features()
    sorted_features=sorted_features+list(set(features)-set(sorted_features))
    sorted_scores=np.append(sorted_scores,sorted_scores[-1])
    sorted_scores=sorted_scores/np.sum(sorted_scores)
    sorted_scores = np.round(sorted_scores, decimals=4).astype(np.float64)
    sorted_scores[0]=1-sorted_scores[1:].sum()

    feature_prob_dict=dict([(key,val) for key, val in zip(sorted_features,sorted_scores)])

    probs=[]
    for f in features:
        probs.append(feature_prob_dict[f])

    probs=np.array(probs)
    return probs.tolist()


def select_reduced_features_for_regression(data,features,target):
    def find_knee_value(sorted_array,sorted_features):
        # 1) Sort the array
        n = len(sorted_array)
        
        # If there's only one point or two points, the concept of a "knee" might not be meaningful;
        # we'll just handle the trivial cases directly:
        if n < 3:
            return sorted_features
        
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
        return sorted_features[:knee_index+1]
    
    from pyHSICLasso import HSICLasso

    hsic_lasso = HSICLasso()
    X = data[features].values
    Y = data[target].values
    hsic_lasso.input(X,Y,featname=features)
    hsic_lasso.regression(len(features)-1)
    selected_features = find_knee_value(hsic_lasso.get_index_score(),hsic_lasso.get_features())
    return selected_features


def select_best_program(sr,regression_ftn):    
    index=-100
    val=np.inf
    for i,p in enumerate(sr.first_pareto_front):
        new_val=p.fitness[regression_ftn.label]
        if new_val<val:
            index=i
            val=new_val
    best_prog=sr.first_pareto_front[index]
    string_best_prog=best_prog.program.render(format_result=True) 
    return best_prog, string_best_prog