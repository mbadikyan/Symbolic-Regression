### SymbolicRegressor program by Minas Badikyan, Research Scholar, AFRL, Summer 2020

### REQUIREMENTS:
### Must provide the filepath of a .csv file with no missing data,
### where the first row is composed of column labels, and every other entry
### is a number.

### DEPENDENCIES:
### gplearn, sklearn, numpy, pandas, mpi4py (for parallel computing)

#from mpi4py import MPI
import argparse
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
#from sklearn.cross_validation import StratifiedKFold
import numpy as np
import pandas as pd
from S_run_aifeynman import run_aifeynman


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
    description='Run AI-Feynman on the given data.')

    parser.add_argument('--pathdir', type=str, required=True,
                    help='Path to the directory containing the data file.')

    parser.add_argument('--filename', type=str, required=True,
                    help='The name of the file containing the data.')

    parser.add_argument('--BR_try_time', type=int, required=False, default=60,
                    help='Time limit for each brute force call.')

    parser.add_argument('--BF_ops_file_type', type=str, required=True,
                    help='File containing the symbols to be used in the brute force code.')

    parser.add_argument('--polyfit_deg', type=int, required=False, default=4,
                    help='Maximum degree of the polynomial tried by the polynomial fit routine.')

    parser.add_argument('--NN_epochs', type=int, required=False, default=4000,
                    help='Number of epochs for the training set.')

    parser.add_argument('--vars_name', nargs='*', type=str, required=True,
                    help='Name of the variables appearing in the equation (plus target, in order).')

    parser.add_argument('--test_percentage', type=int, required=False, default=0,
                    help='Percentage of the input data to be kept aside as test set.')
    
    return(parser)

##==================================================================================##
# Initialize SymbolicRegressor function with command-line parameters.

def run_AIFeynman(args): #args must be a .parse_args() object with all the above parameters

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

    '''
    skf = StratifiedKFold(y_data,
                          n_folds = args.k_fold,
                          shuffle = True,
                          random_state = args.random_state)
    '''
    kf = KFold(n_splits = args.k_fold,
               random_state = args.random_state,
               shuffle = True)

    models = []
    R2vals = []
    #print("Displaying {} models from K-fold cross-validation...".format(args.k_fold))
    for train_index, test_index in kf.split(X_train):
        x_train, x_test = X_train[train_index], X_train[test_index]
        y_train, y_test = Y_train[train_index], Y_train[test_index]

        est_gp.fit(x_train, y_train)
        r2 = est_gp.score(x_test, y_test)
        #print("====================")
        #print(est_gp._program)       #prints fittest program for an individual k-fold
        #print(r2)  #prints R-squared value of model on test set
        #print("====================")
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




    
    
















