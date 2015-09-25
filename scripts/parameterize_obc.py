from bqff.models import model
__author__ = 'Patrick B. Grinaway'

"""
This script will use Bayesian optimization to parameterize an OBC forcefield. Options are specified in a yaml file
"""

import GPy
import GPyOpt

import numpy as np
import yaml
import pickle

# def _get_parameter_bounds(list_of_parameter_names):
#     """
#     Gets a list of parameter bounds (alphabetical) from the parameter names
#
#     Arguments
#     ---------
#     list_of_parameter_names : list of string
#        names of parameters
#
#     Returns
#     -------
#     bounds : list of tuples
#         Bounds for the parameters of interest
#     """
#     list_of_parameter_names.sort()
#     bounds = []
#     for name in list_of_parameter_names:
#         parmtype = name.split('_')[1]
#         if parmtype == 'radius':
#             bounds.append((0.5, 3.0))
#         elif parmtype == 'scalingFactor':
#             bounds.append((-0.8, 1.5))
#         elif parmtype == 'sigma':
#             bounds.append((0, 1000))
#         else:
#             raise ValueError
#     return bounds

def read_gbsa_parameters(filename):
        """
        Read a GBSA parameter set from a file.

        ARGUMENTS

        filename (string) - the filename to read parameters from

        RETURNS

        parameters (dict) - parameters[(atomtype,parameter_name)] contains the dimensionless parameter

        TODO

        * Replace this with a standard format?

        """

        parameters = dict()

        infile = open(filename, 'r')
        for line in infile:
            # Strip trailing comments
            index = line.find('%')
            if index != -1:
                line = line[0:index]

            # Parse parameters
            elements = line.split()
            if len(elements) == 3:
                [atomtype, radius, scalingFactor] = elements
                parameters['%s_%s' % (atomtype,'radius')] = float(radius)
                parameters['%s_%s' % (atomtype,'scalingFactor')] = float(scalingFactor)
            elif len(elements) == 6:
                [atomtype, radius, scalingFactor, alpha, beta, gamma] = elements
                parameters['%s_%s' % (atomtype,'radius')] = float(radius)
                parameters['%s_%s' % (atomtype,'scalingFactor')] = float(scalingFactor)
                parameters['%s_%s' % (atomtype,'alpha')] = float(alpha)
                parameters['%s_%s' % (atomtype,'beta')] = float(beta)
                parameters['%s_%s' % (atomtype,'gamma')] = float(gamma)
        return parameters

# def _param_list_to_dict(parameter_list, list_of_param_names):
#     """
#     Convert the list of parameters to a dictionary for the model
#
#     Arguments
#     ---------
#     parameter_list : list of floats
#         The current set of parameters that GPyOpt wants to try
#     list_of_param_names : list of string
#         The list of the names of the parameters.
#
#     Returns
#     -------
#     parameter_dict_list : dict
#         A dictionary of the format {parameter_name : parameter_value}
#
#     """
#
#     list_of_param_names.sort()
#     parameter_dict = {}
#     for (i, val) in enumerate(parameter_list):
#         parameter_dict[list_of_param_names[i]] = val
#     return parameter_dict


def obc_model_posterior(parameters, model=None, verbose=True):
    """
    This is a function that is callable by GPyOpt. Specifically, it
    takes in a list of parameters (alphabetical order--GPyOpt uses lists instead of dicts)
    and creates a dict with the appropriate parameters for the model.

    Arguments
    ---------
    parameters : list of float
        The current set of parameters that GPyOpt wants to try
    model : GBFFModel
        An instantiated GBFFmodel that can calculate the log prior and likelihood

    Returns
    -------
    ln_post : float
        The log posterior of the model
    """
    ln_unnormalized = []
    for parameter_set in parameters:
        ln_unnormalized.append([model.objective_func(parameter_set,verbose)])
    return np.array(ln_unnormalized)

def gpy_f_factory(model):
    """
    returns a function callable by GPyOpt.

    Since GPyOpt.methods.BayesianOptimization seeks the minimum, we flip the sign
    of the posterior.

    """
    def gpy_f(*parameters):
        return -obc_model_posterior(*parameters, model=model)
    return gpy_f

if __name__ == "__main__":
    import os

    #read in configurations
    yamlfile = open('configs.yaml','r')
    configs = yaml.load(yamlfile)
    yamlfile.close()

    prepared_database_file = configs['database_file']
    parameter_file = configs['parameter_file']
    max_iter = configs['max_iter']
    n_molecules = configs['n_molecules']

    print(os.getcwd())

    #read in the parameters
    parameters = read_gbsa_parameters(parameter_file)
    parameters['model_sigma'] = 100
    parameter_names = parameters.keys()

    #get initial set of parameters in list form:
    parameter_names.sort()
    initial_parameter_list = []
    for (i, name) in enumerate(parameter_names):
        initial_parameter_list.append(parameters[name])
    initial_parameter_array = np.array([initial_parameter_list])

    #load the database
    db_file = open(prepared_database_file,'r')
    database = pickle.load(db_file)
    db_file.close()

    #create a subset for debugging:
    subset_size = n_molecules
    cid_list = database.keys()
    max_num = len(cid_list)
    np.random.seed(0)
    mol_indices = np.random.choice(max_num, subset_size)
    mols_to_use = [cid_list[k] for k in mol_indices]
    database = dict((k, database[k]) for k in mols_to_use)

    #instantiate the model
    model = model.GBFFmodel(database, parameters)

    #generate the bounds for each variable
    bounds = model._parameter_bounds

    #generate the GPyOpt callable
    posterior = gpy_f_factory(model)

    #get GPyOpt started
    #initial_parameter_arrays = [initial_parameter_array + 0.01* np.random.rand(len(bounds)) for _ in range(10)]
    #initial_func_vals = np.hstack([posterior(params) for params in initial_parameter_arrays])

    #get a kernel
    kernel = GPy.kern.RBF(25, lengthscale=0.001)

    BO = GPyOpt.methods.BayesianOptimization
    gpyopt = BO(posterior, bounds,
                numdata_initial_design=100, kernel=kernel)

    gpyopt.run_optimization(max_iter)

    result = gpyopt.x_opt

    result_dict = _param_list_to_dict(result, parameter_names)

    print(result_dict)

    gpyopt.save_report()
