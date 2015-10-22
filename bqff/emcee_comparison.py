'''
Baseline: draw samples using affine-invariant ensemble MCMC,
as implemented in emcee.
'''


__author__='Josh Fass'

import pickle

import numpy as np
import yaml
import time
import emcee

from models import model

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

if __name__=='__main__':
    ## setup stuff is mostly copied from parameterize_obc
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

    #save the molecules we used so we can compare to test set
    mols_used = open('mols_used_%s.txt' % str(time.time()))
    for mol in mols_to_use:
        mols_used.write(mol+"\n")
    mols_used.close()

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
    list_of_param_names = sorted(parameters.keys())
    p1 = []
    for parm in list_of_param_names:
        p1.append(parameters[parm])
    p0=[np.array(p1)+np.random.rand(dim)*scale/10 for _ in range(nwalkers)]
    #p0=[(np.random.rand(dim)*scale)+left_bounds for _ in range(nwalkers)]

    sample_trace = open('/cbio/jclab/projects/pgrinaway/gbff_aies/bqff/bqff/samples.dat','w')
    sample_trace.close()
    for result in sampler.sample(p0, iterations=50000, storechain=False):
        position = result[0]
        f = open("/cbio/jclab/projects/pgrinaway/gbff_aies/bqff/bqff/samples.dat", "a")
        for k in range(position.shape[0]):
            f.write("{0:d} {1:s}\n".format(k, " ".join(position[k])))
        f.close()
