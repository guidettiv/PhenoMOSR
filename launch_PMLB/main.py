import argparse
import logging
import sys
import pandas as pd
# to include the SR code without installing in the environment
sys.path.append('../')
from symbolic_regression.SymbolicRegressor import SymbolicRegressor, decompress
from symbolic_regression.preprocessing import select_feature_probabilities_regression, select_reduced_features_for_regression
from symbolic_regression.operators import *
from symbolic_regression.multiobjective.fitness.Regression import WeightedMeanSquaredError, Complexity, ModuleDepthWeigthedComplexity, WMSEAkaike, WMSEBayes, RegressionMinimumDescriptionLength
from symbolic_regression.callbacks.CallbackSave import MOSRCallbackSaveCheckpoint
from symbolic_regression.callbacks.CallbackStatistics import MOSRHistory, MOSRStatisticsComputation

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


def one_SR_run():
    parser = argparse.ArgumentParser(description="Configuration parameters.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--complexity", type=str, required=True, help="Complexity: cstd or cmod")
    parser.add_argument("--pheno_freq", type=str, required=True, help="brgflow, bootstrap, additive mutation freq list  e.g. '[0,2,0]'")
    parser.add_argument("--genetic_mutations", type=str, required=True, help="use std genetic mutation?")
    parser.add_argument("--feature_probability", type=str, required=False, default=False, help="use non trivial feature probability?")
    parser.add_argument("--seed", type=str, required=True, help="np seed")
    parser.add_argument("--n_jobs", type=str, required=True, help="number of jobs")
    args = parser.parse_args()

    genetic_mutations=(args.genetic_mutations=='True')
    bool_feature_probability=(args.feature_probability=='True')
    seed=int(args.seed)
    n_jobs=int(args.n_jobs)

    import ast
    pheno_freq = ast.literal_eval(args.pheno_freq)
    pheno_freq = [int(x) for x in pheno_freq]

    
    ####################
    ## SET RANDOM SEED
    ####################
    np.random.seed(seed=seed)

    ################
    ## OPERATORS
    ################
    operations = [
        OPERATOR_ADD,
        OPERATOR_SUB,
        OPERATOR_MUL,
        OPERATOR_DIV,
        OPERATOR_ABS,
        # OPERATOR_MOD,
        # OPERATOR_NEG,
        # OPERATOR_INV,
        OPERATOR_LOG,
        OPERATOR_EXP,
        OPERATOR_POW,
        OPERATOR_SQRT,
        #OPERATOR_MAX,
        #OPERATOR_MIN,
        OPERATOR_SIN
    ]

    ################
    ## DATASET
    ################
    from sklearn.model_selection import train_test_split
    from pmlb import fetch_data

    # Returns a pandas DataFrame
    data = fetch_data(args.dataset)
    if len(data)<=1000:
        train_hrs=1
    else: 
        train_hrs=10

    features=list(data.columns)
    features.remove('target')
    print(features)
    target='target'
    weights='w'
    
    train, test = train_test_split(data)
    train['w']=1.
    test['w']=1.

    #####################################
    ## STATIC/DYNAMIC FEATURE PROBABILITY
    #####################################
    if bool_feature_probability:
        feature_probability=select_feature_probabilities_regression(train,features,target)
        bool_update_feature_probability=True
    else:
        feature_probability=[1./len(features)]*len(features)
        bool_update_feature_probability=False


    #######################
    ## FITNESS FUNCTIONS 
    #######################
    constants_optimization = 'scipy'
    constants_optimization_conf = {'task': 'regression:wmse'}

    regression_ftn=WMSEAkaike(label='AIC', target=target,
                    weights=weights, minimize=True,  
                    constants_optimization=constants_optimization, 
                    constants_optimization_conf=constants_optimization_conf)
    if args.complexity=='cstd':
        complexity_ftn=Complexity(label='Complex', one_minus=False, minimize=True)
    elif args.complexity=='cmod':
        complexity_ftn=ModuleDepthWeigthedComplexity(label='ModComplex', one_minus=False, minimize=True)
    
    fitness_functions = [regression_ftn,complexity_ftn]

    #######################
    ## GENETIC OPERATORS 
    #######################

    if genetic_mutations:
        genetic_operators_frequency = {
                                    'crossover': 1,
                                    'mutation': 1,
                                    'insert_node': 1,
                                    'delete_node': 1,
                                    'mutate_leaf': 1,
                                    'mutate_operator': 1
                                }
    else:
        genetic_operators_frequency = { }

    genetic_operators_frequency['regularize_brgflow']=pheno_freq[0]
    genetic_operators_frequency['regularize_bootstrap']=pheno_freq[1]
    genetic_operators_frequency['additive_expansion']=pheno_freq[2]

    #############################
    ## CALLBACKS AND PARAMETERS 
    #############################

    file_name = f"./{args.dataset}_{args.complexity}_brg_{pheno_freq[0]}_boot_{pheno_freq[1]}_add_{pheno_freq[2]}_genetic_mut_{str(genetic_mutations)}_nontrivial_feat_prob_{str(bool_feature_probability)}_seed_{seed}"
    print(f'file name: {file_name}')
    
    callbacks = [
        MOSRCallbackSaveCheckpoint(
            checkpoint_file=file_name, checkpoint_frequency=1, checkpoint_overwrite=True),
        MOSRStatisticsComputation(),
        MOSRHistory(history_fpf_frequency=-1)]

    POPULATION_SIZE = 100
    GENERATIONS = 100
    TOURNAMENT_SIZE = 3
    TIMEOUT_MINS = 60 * train_hrs
    const_range = (-1, 1)

    logging.info(f'Running with POPULATION_SIZE {POPULATION_SIZE}')
    logging.info(f'Running with TOURNAMENT_SIZE {TOURNAMENT_SIZE}')

    ###############################
    ## SR INITIALIZATION AND FIT 
    ###############################

    sr = SymbolicRegressor(
        client_name='client',
        const_range=const_range,
        parsimony=.8,
        parsimony_decay=.85,  # Expected depth = parsimony / (1-parsimony_decay)
        population_size=POPULATION_SIZE,
        tournament_size=TOURNAMENT_SIZE,
        genetic_algorithm='NSGA-II', #'NSGA-II', 'SMS-EMOEA'
        genetic_operators_frequency=genetic_operators_frequency,
        callbacks=callbacks
    )


    sr.fit(
        data=train,
        val_data=test,
        features=features,
        feature_probability=feature_probability,
        bool_update_feature_probability=bool_update_feature_probability,
        operations=operations,
        fitness_functions=fitness_functions,
        generations_to_train=GENERATIONS,
        training_timeout_mins=TIMEOUT_MINS*100,
        n_jobs=n_jobs,
        stop_at_convergence=False,
        convergence_rolling_window=None,
        verbose=1  # The output could be very verbose. Consider using 0, 1, or 2 to reduce the verbosity
    )

    ##########################
    ## SELECT BEST AND SAVE
    ##########################
    
    best_prog, string_best_prog = select_best_program(sr,regression_ftn)

    import sympy as sym
    from sklearn.metrics import r2_score
    import pickle

    num_nodes = sum(1 for _ in sym.preorder_traversal(sym.simplify(string_best_prog)))
    simplicity = round(-np.log(num_nodes)/np.log(5),1)

    y_pred = best_prog.evaluate(test)
    r2 = r2_score(test[target], y_pred)

    data_to_save = [best_prog, string_best_prog, simplicity, r2]

    # Save to a file named "my_data.pkl" (you can change the file name as desired)
    best_res_file=f'./best_prog_{args.dataset}_{args.complexity}_brg_{pheno_freq[0]}_boot_{pheno_freq[1]}_add_{pheno_freq[2]}_genetic_mut_{str(genetic_mutations)}_nontrivial_feat_prob_{str(bool_feature_probability)}_seed_{seed}'
    with open(best_res_file+".pkl", "wb") as f:
        pickle.dump(data_to_save, f)


    print('End')


if __name__ == "__main__":
    one_SR_run()
