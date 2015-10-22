'''
Baseline: draw samples using affine-invariant ensemble MCMC,
as implemented in emcee.
'''


__author__='Josh Fass'

import pickle

import numpy as np
import yaml
import emcee


if __name__=='__main__':
    ## setup stuff is mostly copied from parameterize_obc
    import os
    import bqff.models as model

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
    modelobj = model.GBFFmodel(database, parameters)

    ## instantiate the sampler
    dim = modelobj._num_params
    nwalkers = dim*2
    lnpostfn=lambda theta:modelobj.callable_lnpostfn(theta,verbose=True)

    sampler = emcee.EnsembleSampler(nwalkers=nwalkers,dim=dim,lnpostfn=lnpostfn)
    bounds = np.array(modelobj._parameter_bounds)
    print(bounds.shape)
    left_bounds,right_bounds = bounds[:,0],bounds[:,1]
    scale=right_bounds-left_bounds

    p0=[(np.random.rand(dim)*scale)+left_bounds for _ in range(nwalkers)]

    sampler.run_mcmc(p0,100)
