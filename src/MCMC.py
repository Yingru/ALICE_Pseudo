### Note, in this version, the extra_sys_error are added in the physical space, instead of the PC space
import numpy as np
from scipy.linalg import lapack
from scipy import linalg
import matplotlib.pyplot as plt
import h5py
import os, sys
import logging

import emcee
from gaussian_emulator import Emulator


logging.basicConfig(
    stream=sys.stdout,
    format='[%(levelname)s][%(module)s] %(message)s',
    level=os.getenv('LOGLEVEL', 'info').upper()
)


def mvn_loglike(y, cov):
    """
    Calculate multi-varaite-normal log-likelihood
        log_p = -1/2 * [(y^T).(C^-1).y + log(det(C))] + const

    To normalize the likelihood, const = -n/2*log(2*pi), which is omitted here

    Args:
        y -- (n)
        cov -- shape (n, n)
    Returns:
        log_p
    """

    L, info = lapack.dpotrf(cov, clean=False)
    if info != 0:
        raise ValueError('lapack dpotrf error: illegal value for info!')

    alpha, info = lapack.dpotrs(L, y)
    if info != 0:
        raise ValueError('lapack dpotrf error: illegal value for info! {}'.format(info))

    return -.5*np.dot(y, alpha) - np.log(L.diagonal()).sum()



def credible_interval(samples, ci=0.9):
    """
    compute the highest-posterior density (HPD) credible interval
    """
    nci = int((1-ci) * samples.size)
    argp = np.argpartition(samples, [nci, samples.size - nci])
    cil = np.sort(samples[argp[:nci]])
    cih = np.sort(samples[argp[:nci]])
    ihpd = np.argmin(cih - cil)

    return cil[ihpd], cih[ihpd]





class LoggingEnsembleSampler(emcee.EnsembleSampler):
    def run_mcmc(self, X0, nsteps, status=None, **kwargs):
        """
        run_mcmc wrarpper with additional arg `status`
        """
        logging.info('running %d walkers for %d steps', self.k, nsteps)
        
        if status == None:
            status = nsteps//5

        for n, result in enumerate(self.sample(X0, iterations=nsteps, **kwargs), start=1):
            if n%status == 0 or n == nsteps:
                af = self.acceptance_fraction
                logging.info('step %d: acceptance fraction: mean %.4f, std %.4f, min %.4f, max %.4f',
                                n, af.mean(), af.std(), af.min(), af.max())
        return result





class Chain():
    def __init__(self, emulator, Y_exp, Y_err, X_min, X_max, cov_on, extra_std_scale=0.05):
        """
        Args:
            emulator: --- gaussian_emulator.Emulator, already being trainedd
            Y_exp: ---  experimental/desired Y
            Y_err: --- (sys_err, stat_err)
            X_min: --- parameter lower limit
            X_max: --- parameter upper limit
        """
        self.min, self.max = X_min, X_max
        self.ndim = len(self.min)
        print(self.ndim)
        print('init: ', self.min)
        self.emulator = emulator 
        self._common_indices = list(range(emulator.ndim))

        self.Y_exp = Y_exp
        self.cov_exp = np.diag(Y_err[0])**2 + np.diag(Y_err[1])**2 # + np.diag(0.01*self,emulator.scaler.var_)

        self.cov_on = cov_on
        self.extra_std_scale = extra_std_scale 

    def _predict(self, X, **kwargs):
        """
        call each system emulator to predict model output at X
        """
        return self.emulator.predict(X[:, self._common_indices], **kwargs)


    def log_likelihood(self, X):
        """
        Evaluate the log-likelihood at position X
        """

        X = np.array(X, copy=False, ndmin=2)
        lp = np.zeros(X.shape[0])
        inside = np.all((X > self.min) & (X < self.max), axis=1)
        lp[~inside] = -np.inf

        nsamples = np.count_nonzero(inside)
        if nsamples > 0:
            extra_std = X[inside, -1]
            pred = self._predict(X[inside], return_cov=True, extra_std=extra_std)
            Y_pred, cov_pred, cov_pred2 = pred

            if self.extra_std_scale == 0:
                cov = cov_pred + self.cov_exp[np.newaxis,:]
            else:
                cov = cov_pred + self.cov_exp[np.newaxis,:] + (extra_std**2)[:,np.newaxis,np.newaxis] * (np.eye(cov_pred.shape[-1]))[np.newaxis,:]

            ## compute the loglikelihood at each point
            lp[inside] += list(map(mvn_loglike, Y_pred - self.Y_exp, cov))

            if self.cov_on:
                ## add prior for extra_std (emulator sys error)
                if self.extra_std_scale != 0:
                    lp[inside] += 2*np.log(extra_std) - extra_std/self.extra_std_scale

        return lp


    def random_pos(self, n=1):
        """
        generate n random positions in parameter space
        """
        return np.random.uniform(self.min, self.max, (n, self.ndim))


    @staticmethod
    def map(f, args):
        return f(args)


    def calibrate(self, nsteps, nwalkers, outputFile, status=None):
        """
        Run MCMC model calibration. If the chain already exists, continue from the last point;
                                    otherwise burn-in and start the chain
        """
        f = h5py.File(outputFile, 'a')
        try:
            dset = f['chain']
        except KeyError:
            print('calibration: ', self.ndim)
            burn = True
            dset = f.create_dataset('chain', dtype='f8',
                                    shape=(nwalkers, 0, self.ndim),
                                    chunks=(nwalkers, 1, self.ndim),
                                    maxshape=(nwalkers, None, self.ndim),
                                    compression='lzf')
        else:
            burn = False
            nwalkers = dset.shape[0]

        sampler = LoggingEnsembleSampler(nwalkers, self.ndim, self.log_likelihood, pool=self)
        if burn:
            logging.info('no existing chain found, starting initial burn-in')

            # burn-in steps set as half of the steps, run first half of burn-in starting from random position,
            # and then run second half of burn-in. This significantly accelerates burn-in, help prevent stuck walker
            nburnsteps = nsteps//2

            #### first half of burn-in
            nburn0 = nburnsteps//2
            initial_pos = self.random_pos(nwalkers)
            sampler.run_mcmc(initial_pos, nburn0, status=status)

            #### second half of burn-in
            logging.info('resamling walker positions')
            X0 = sampler.flatchain[np.unique(sampler.flatlnprobability, return_index=True)[1][-nwalkers:]]
            sampler.reset()
            X0 = sampler.run_mcmc(X0, nburnsteps - nburn0, storechain=False, status=status)[0]
            sampler.reset()
            logging.info('burn-in complete, starting production')


        else:
            logging.info('restarting from last point of existing chain')
            X0 = dset[:,-1,:]

        sampler.run_mcmc(X0, nsteps, status=status)
        logging.info('writing chain to file')
        dset.resize(dset.shape[1] + nsteps, 1)
        dset[:, -nsteps:, :] = sampler.chain

        f.close()



    def samples(self, outputFile, n=1):
        """
        emulator's predict of model's output at n parameters randomly draw from the chain
        """
        try:
            d = h5py.File(outputFile, 'r')['chain']
            X = np.array([d[i] for i in zip(*[np.random.randint(s, size=n) for s in d.shape[:2]]) ])
            return (X, self._predict(X))

        except KeyError:
            logging.info('Warning! No existing sampler')



    def samplesCR(self, outputFile, n=1, CR=90):
        """
        emulator's prediction of model's output but with % CR
        """
        try:
            limit = (100 - CR)//2
            d = h5py.File(outputFile, 'r')['chain'].value
            d = d.reshape((d.shape[0]*d.shape[1], d.shape[2]))
            lower = np.percentile(d, limit, axis=0)
            upper = np.percentile(d, 100-limit, axis=0)
            inside = np.all((d > lower) & (d < upper), axis=1)
            dCR = d[inside]
            print(d.shape, dCR.shape, inside.shape)
            X = np.array([dCR[i] for i in np.random.randint(dCR.shape[0], size=n)])
            return (X, self._predict(X))

        except KeyError:
            logging.info('Warning! No existing sampler')



