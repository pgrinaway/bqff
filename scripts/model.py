

__author__ = 'Patrick B. Grinaway'

import scipy.stats as stats
import energies

class GBFFmodel(object):
    """
    This class contains the implementation for the GBFF model, without using PyMC

    Arguments
    ---------
    prepared_database : dict
        A prepared FreeSolv database
    """

    def __init__(self, prepared_database, initial_parameter_dict):
        self._database = prepared_database
        self._initial_parameters = initial_parameter_dict
        self._parameter_names = initial_parameter_dict.keys().sort()
        self._parameter_bounds = self._get_parameter_bounds()
        self._num_params = len(self._parameter_names)

    def ln_prior(self, parameters):
        """
        Calculate the log prior probability of the FF parameters
        """
        lnprior = 0.0
        for (key, value) in parameters.iteritems():
            parmtype = key.split("_")[1]
            if parmtype == 'radius':
                lnprior += stats.distributions.uniform.logpdf(value, 0.5, 2.5)
            elif parmtype == 'scalingFactor':
                lnprior += stats.distributions.uniform.logpdf(value, -0.8, 2.3)
            elif parmtype == 'sigma':
                lnprior += stats.distributions.invgamma.logpdf(value,1.0,1.0)
        return lnprior


    def ln_likelihood(self, parameters, verbose=True):
        """
        This calculates the log-likelihood of the given parameters.

        Arguments
        ---------
        parameters : dict
            A dictionary of the current set of parameters

        Returns
        -------
        ln_like : float
            The log-likelihood of the parameters
        """
        ln_like = 0.0

        #calculate free energies--returns a dictionary of {cid : DeltaG}
        delta_g = energies.compute_hydration_energies_sequentially(self._database, parameters)

        #iterate through and add each molecule's contribution to the loglikelihood
        for cid in delta_g.iterkeys():
            dg_gbsa = delta_g[cid]
            dg_exp = self._database[cid]['expt']
            ddg_expt = self._database[cid]['d_expt']
            normal_sigma = parameters['model_sigma']**2 + ddg_expt **2

            if verbose==True:
                #print('The calculated dG for %s is %s, and the experimental is %f' % (self._database[cid]['iupac'], str(dg_gbsa), dg_exp))
                print(self._database[cid]['iupac'])
                try:
                    print('\tCalculated dG: {0:.3f}\n\tExperimental dG: {1:.3f}'.format(dg_gbsa, dg_exp))
                except:
                    print('dg_gbsa',dg_gbsa,type(dg_gbsa))
                    print('dg_exp',dg_exp,type(dg_exp))
                print('')
            ln_like += stats.distributions.norm.logpdf(dg_gbsa, dg_exp, normal_sigma)

        return ln_like

    def ln_posterior(self,parameter_dict,verbose=True):
        return self.ln_prior(parameter_dict) + self.ln_likelihood(parameter_dict,verbose)

    def _get_parameter_bounds(self):
        """
        Gets a list of parameter bounds (alphabetical) from the parameter names

        Arguments
        ---------
        list_of_parameter_names : list of string
           names of parameters

        Returns
        -------
        bounds : list of tuples
            Bounds for the parameters of interest
        """

        bounds = []
        for name in self._parameter_names:
            parmtype = name.split('_')[1]
            if parmtype == 'radius':
                bounds.append((0.5, 3.0))
            elif parmtype == 'scalingFactor':
                bounds.append((-0.8, 1.5))
            elif parmtype == 'sigma':
                bounds.append((0, 1000))
            else:
                raise ValueError
        return bounds

    def _parse_params(self,param_vector):
        """
        Convert the param_vector to a dictionary for the model

        Arguments
        ---------
        param_vector : array-like of floats, shape=(n,) or (n,1) where n=self._num_params

        Returns
        -------
        parameter_dict_list : dict
            A dictionary of the format {parameter_name : parameter_value}

        """
        assert(len(param_vector)==self._num_params)
        parameter_dict = dict()

        for (i, val) in enumerate(param_vector):
            parameter_dict[self._parameter_names[i]] = val
        return parameter_dict

    def objective_func(self,param_vector,verbose=False):
        parameter_dict = self._parse_params(param_vector)
        return self.ln_posterior(parameter_dict,verbose)
