"""
SymbolicRegressor program by Minas Badikyan, Research Scholar, AFRL, Summer 2020

REQUIREMENTS:
Must provide the filepath of a .csv file with no missing data,
where the first row is composed of column labels, and every other entry
is a number.

DEPENDENCIES:
gplearn, sklearn, numpy, pandas

REFERENCES:

gplearn documentation: 
https://gplearn.readthedocs.io/en/stable/

Scikit Learn:
https://scikit-learn.org/stable/
"""

import argparse
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


def median(List):
    n = len(List)
    lst = List[:]
    lst.sort()
    if n % 2 == 0:
        return(lst[int(n/2)-1])
    else:
        return(lst[int(n/2)])
    

### function to parse command-line arguments ###
def create_parser():
    parser = argparse.ArgumentParser(
    description='Generate K (cross-validation splits) symbolic regression models using the gplearn package.')

    parser.add_argument('-train', '--train_filepath', type=str, required=True,
                    help='Input file in CSV format to be used for training.')

    parser.add_argument('-test', '--test_set', type=float, required=False, default=0.2,
                    help='The fraction of the original data to be set aside as a test data set.')

    parser.add_argument('-preds', '--predictors', nargs='*', type=int, required=False, default=None,  
                    help='Indices of predictors. Defaults to all columns but the last. Note that first column has index 0.')

    parser.add_argument('-targ', '--target', nargs=1, type=int, required=False, default=None,         
                    help='Index of target variable. Defaults to last column in data set.') 

    parser.add_argument('-k', '--k_fold', type=int, required=False, default=2,
                    help='The value k for k-fold cross validation. Defaults to k=2.')

    parser.add_argument('-pop', '--population_size', type=int, required=False, default=1000,
                    help='The number of programs in each generation.')

    parser.add_argument('-gen', '--generations', type=int, required=False, default=20,
                   help='The number of generations to evolve.')

    parser.add_argument('-tns', '--tournament_size', type=int, required=False, default=20,
                    help='Number of programs that will compete to become part of the next generation.')

    parser.add_argument('-stc', '--stopping_criteria', type=float, required=False, default=0.0,
                    help='The required metric value required in order to stop evolution early.')

    parser.add_argument('-cor', '--const_range', nargs=2, type=float, required=False, default=[-1.0, 1.0],
                    help='The range of values in which to search for any constant terms in the model.')

    parser.add_argument('-ind', '--init_depth', nargs=2, type=int, required=False, default=[2, 6],
                    help='The range of tree depths for the initial population of naive formulas.')

    parser.add_argument('-inm', '--init_method', type=str, choices=['grow','full','half and half'],
                    required=False, default='half and half',
                    help = 'Dictates reproduction--see gplearn documentation')

    function_set = ['add','sub','mul','div','sqrt','log','abs','neg','inv','max','min','sin','cos','tan','exp','all']
    default_set = ['add','sub','mul','div']
    parser.add_argument('-fns', '--function_set', nargs='*', type=str, choices=function_set, required=False, default=default_set,
                    help='The functions to use when building or evolving programs.')

    metrics = ['mean absolute error','mse','rmse','pearson','spearman']
    parser.add_argument('-met', '--metric', type=str, choices=metrics, required=False, default='mean absolute error',
                    help='The error metric used to compare evolved models to test data.')

    parser.add_argument('-par','--parsimony_coefficient', type=float, required=False, default=0.001,
                    help='Determines model complexity, higher values yield smaller models.')

    parser.add_argument('-pxr','--p_crossover', type=float, required=False, default=0.9,
                    help='The probability of a cross-over type mating occuring between two parents.')

    parser.add_argument('-psm', '--p_subtree_mutation', type=float, required=False, default=0.01,
                    help='The probability of a random subtree mutation of any given offspring.')

    parser.add_argument('-phm', '--p_hoist_mutation', type=float, required=False, default=0.01,
                    help='The probability of a random hoist mutation of any given offspring.')

    parser.add_argument('-ppm', '--p_point_mutation', type=float, required=False, default=0.01,
                    help='The probability of a random point mutation of any given offspring.')

    parser.add_argument('-ppr','--p_point_replace', type=float, required=False, default=0.05,
                    help='The probability of a random point replacement of any given offspring.')

    parser.add_argument('-mxs','--max_samples', type=float, required=False, default=1.0,
                    help='The fraction of samples to draw from X to evaluate each program on.')

    parser.add_argument('-ftn','--feature_names', nargs='*', type=str, required=False, default=None,
                    help='Optional list of feature names, defaults are X0, X1, etc...')

    parser.add_argument('-wrm','--warm_start', type=bool, required=False, default=False,
                     help='When set to True, reuse the solution of the previous call to fit and add more generations to the evolution.')

    parser.add_argument('-low','--low_memory', type=bool, required=False, default=False,
                     help='When True, only the current generation is retained, reduces memory usage.')

    parser.add_argument('-jobs','--n_jobs', type=int, required=False, default=1,
                     help='The number of jobs to run in parallel. If set to -1, number of jobs is set to number of cores.')

    parser.add_argument('-vrb','--verbose', type=int, required=False, default=1,
                     help='Controls the verbosity of the evolution building process.')

    parser.add_argument('-rnd','--random_state', type=int, required=False, default=0,
                     help='Included value is used to seed the random number generator.')

    return(parser)

##==================================================================================##
# Initialize SymbolicRegressor function with command-line parameters.

def init_GP(args): #args must be a .parse_args() object with all the above parameters

    if 'all' in args.function_set:
        function_set = ['add','sub','mul','div','sqrt','log','abs','neg','inv','max','min','sin','cos','tan','exp']
        args.function_set = function_set[:]
    

    if 'exp' in args.function_set:    #if exponential function is included in function set
        
        def _protected_exponent(x):   #custom defines an exp function (not included in gplearn)
            with np.errstate(over='ignore'):  #that protects against overflow
                return np.where(np.abs(x)<100, np.exp(x), 0.)
            
        exp = make_function(_protected_exponent, 'exp', arity=1, wrap=True)
        args.function_set.remove('exp')
        args.function_set.append(exp)
        
        est_gp = SymbolicRegressor(population_size = args.population_size, 
                                   generations = args.generations,
                                   tournament_size = args.tournament_size,
                                   stopping_criteria = args.stopping_criteria,
                                   const_range = tuple(args.const_range), 
                                   init_depth = tuple(args.init_depth),
                                   init_method = args.init_method, 
                                   function_set = tuple(args.function_set), 
                                   metric = args.metric, 
                                   parsimony_coefficient = args.parsimony_coefficient,
                                   p_crossover = args.p_crossover, 
                                   p_subtree_mutation = args.p_subtree_mutation,
                                   p_hoist_mutation = args.p_hoist_mutation, 
                                   p_point_mutation = args.p_point_mutation,
                                   p_point_replace = args.p_point_replace, 
                                   max_samples = args.max_samples,
                                   feature_names = args.feature_names, 
                                   warm_start = args.warm_start,
                                   low_memory = args.low_memory, 
                                   n_jobs = args.n_jobs,
                                   verbose = args.verbose,
                                   random_state = args.random_state)

    else:
        est_gp = SymbolicRegressor(population_size = args.population_size, #if exp is not listed in the function set
                                   generations = args.generations,
                                   tournament_size = args.tournament_size,
                                   stopping_criteria = args.stopping_criteria,
                                   const_range = tuple(args.const_range), 
                                   init_depth = tuple(args.init_depth),
                                   init_method = args.init_method, 
                                   function_set = tuple(args.function_set), 
                                   metric = args.metric, 
                                   parsimony_coefficient = args.parsimony_coefficient,
                                   p_crossover = args.p_crossover, 
                                   p_subtree_mutation = args.p_subtree_mutation,
                                   p_hoist_mutation = args.p_hoist_mutation, 
                                   p_point_mutation = args.p_point_mutation,
                                   p_point_replace = args.p_point_replace, 
                                   max_samples = args.max_samples,
                                   feature_names = args.feature_names, 
                                   warm_start = args.warm_start,
                                   low_memory = args.low_memory, 
                                   n_jobs = args.n_jobs,
                                   verbose = args.verbose,
                                   random_state = args.random_state)
    return(est_gp)

##====================================================================================##

def prep_data(data, args):    #takes a pandas dataframe and returns a preprocessed numpy ndarray
    n_rows = data.shape[0]

    subset = np.random.choice(range(n_rows), n_rows) #randomly shuffle observations
    data = data.iloc[subset, :]

    if args.predictors and args.target:
        data_preds = data.iloc[:, args.predictors]
        data_targ = data.iloc[:, args.target]
        data = pd.concat([data_preds, data_targ], axis=1)

    if args.predictors and not args.target:
        data_preds = data.iloc[:, args.predictors]
        data = pd.concat([data_preds, data[:, -1]], axis=1)

    n_cols = data.shape[1]    
    print("Predictors and target(last column) are {}".format(data.columns))
    
    data = data.to_numpy()   #converts pandas DataFrame to numpy array

    subset = np.random.choice(range(n_rows), int(n_rows*args.test_set))
    test_data = data[subset, :]
    train_indices = np.setdiff1d(range(n_rows), subset)
    train_data = data[train_indices, :]
	
    scaler = StandardScaler()  #scale the data to be standard normally distributed
    train_data = scaler.fit_transform(train_data)
    
    return(train_data, test_data, n_cols)

##========================================================================================##

def main():
    parser = create_parser()  
    args = parser.parse_args() #parse command line arguments into variables

    est_gp = init_GP(args)    #initialize gp regressor using variables

    data = pd.read_csv(args.train_filepath)  #preprocess data
    train_data, test_data, n_cols = prep_data(data, args)
    print("Setting aside {}% of the data as test data.".format(int(args.test_set*100)))

    X_train = train_data[:, 0:n_cols-1]  #split training data into predictors and target variable
    Y_train = train_data[:, -1]
    del data

    kf = KFold(n_splits = args.k_fold,
               random_state = args.random_state,
               shuffle = True)

    models = []
    R2vals = []
    for train_index, test_index in kf.split(X_train):
        x_train, x_test = X_train[train_index], X_train[test_index]
        y_train, y_test = Y_train[train_index], Y_train[test_index]

        est_gp.fit(x_train, y_train)
        r2 = est_gp.score(x_test, y_test)
        models.append(est_gp)
        R2vals.append(r2)
		
	
    best_gp = models[R2vals.index(median(R2vals))] #pick the model with median R-squared value
    X_test = test_data[:, 0:n_cols-1]
    Y_test = test_data[:, -1]

    print("==============================")
    print(best_gp._program)
    print(best_gp.score(X_test, Y_test))
    print("==============================")

    return()


if __name__ == "__main__":
    main()
