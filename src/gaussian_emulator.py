"""
Gaussian Emulator
"""

import logging
import sys, os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process import kernels
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import QuantileTransformer

from scipy import linalg
import matplotlib.pyplot as plt
import pandas as pd


logging.basicConfig(
    stream = sys.stdout,
    format = '[%(levelname)s][%(module)s] %(message)s',
    level = os.getenv('LOGLEVEL', 'info').upper()
)



class Emulator:
    """
    Multi-dimensional Gaussian process emulator using principal component analysis
    model Y_train ---> (scaler-standardize) ---> (singular value decomposition) PCA
    """

    def __init__(self, X_train, Y_train, X_train_min=None, X_train_max=None, npc=10, nrestarts=0):
        """
        Initialize the emulator with X_train, Y_train, X_train_range, npc, nrestarts
        NOTE: if X_train_min == None, X_min=0.9*(X_min)
              if X_train_max == None, X_max=1.1*(X_max)
        """
        logging.info(
            'training emulator for system with (%d PC, %d restarts)', npc, nrestarts)


        self.npc = npc 
        (_, self.ndim) = X_train.shape
        (self.nsamples, self.nobs) = Y_train.shape

        ### standardize X_train
        if X_train_min is None:
            X_train_min = 0.9 * np.min(X_train, axis=0)
        if X_train_max is None:
            X_train_max = 1.1 * np.max(X_train, axis=0)
        self.X_min = np.array(X_train_min)
        self.X_max = np.array(X_train_max)
        self.Y_train = np.copy(Y_train)
        self.X_train = (X_train - X_train_min)/(X_train_max - X_train_min)

        ### standardScaler transformation
        ### Args: Y; returns: (Y - Y.mean)/Y.std
        self.scaler = StandardScaler(copy=False)


        ### principal components analysis 
        ### Args: Y; Y -- (svd) --> Y = U.S.Vt,
        ### Returns:   if Whiten, Z = Y.V/S * np.sqrt(nsamples -1) = U * np.sqrt(nsamples -1)
        ###            else,      Z = Y.V = U.S
        self.pca = PCA(copy=False, whiten=True, svd_solver='full')
        Z = self.pca.fit_transform(self.scaler.fit_transform(self.Y_train))[:, :npc]


        ### Kernel and Gaussian Process Emulators
        k0 = 1.*kernels.RBF(length_scale=np.ones(self.ndim),length_scale_bounds=np.outer(np.ones(self.ndim),(.1,10)))
        k1 = kernels.ConstantKernel()    # ConstantKernel doesn't help
        k2 = kernels.WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-8,10.))
        kernel = (k0 + k2)

        self.GPs = [GPR(kernel=kernel, alpha=0, n_restarts_optimizer=nrestarts, copy_X_train=False).fit(self.X_train, z) for z in Z.T]

        ## construct the full linear transform matrix
        self._trans_matrix = self.pca.components_ * np.sqrt(self.pca.explained_variance_[:, np.newaxis]) * self.scaler.scale_
        ## in-order to propagate the predictive variance: https://en.wikipedia.org/wiki/Propagation_of_uncertainty
        A = self._trans_matrix[:npc]
        self._var_trans = np.einsum('ki,kj->kij', A, A, optimize=False).reshape(npc, self.nobs**2)
        B = self._trans_matrix[npc:]
        ## covariance matrix for the remaining neglected PCs
        self._cov_trunc = np.dot(B.T, B)
        self._cov_trunc.flat[::self.nobs + 1] += 1e-4 * self.scaler.var_



    def _inverse_transform(self, Z):
        """
        Inverse transformation from PC spaec to physical space
        Args:
            Z -- shape (None, npc)
        Returns:
            Y -- shape (None, nobs)
        """
        Y = np.dot(Z, self._trans_matrix[: Z.shape[-1]])
        Y += self.scaler.mean_

        return Y


    def predict(self, X, return_cov=False, extra_std=0):
        """
        Predict model output at arbitrary design X
        Args:
            X -- 2D array-like, shape (None, ndim): need to be normalized before-hand!!!
        Returns:
            if return_cov: tuple (mean, cov)
            else:   mean -- shape (None, nobs)
        """
        X = np.atleast_2d(X)
        X = (X - self.X_min)/(self.X_max - self.X_min)
        gp_mean = [gp.predict(X, return_cov=return_cov) for gp in self.GPs]
        if return_cov:
            gp_mean, gp_cov = zip(*gp_mean)

        mean = self._inverse_transform(np.concatenate([m[:, np.newaxis] for m in gp_mean], axis=1))

        if return_cov:
            gp_var = np.concatenate([c.diagonal()[:, np.newaxis] for c in gp_cov], axis=1)
            cov = np.dot(gp_var, self._var_trans).reshape(X.shape[0], self.nobs, self.nobs)
            cov += self._cov_trunc

            ### additional uncertainties added to the emulator, to account for the model discrepancy
            extra_std = np.array(extra_std, copy=False).reshape(-1,1)
            gp_var += extra_std**2
            cov2 = np.dot(gp_var, self._var_trans).reshape(X.shape[0], self.nobs, self.nobs)

            return (mean, cov, cov2)
        else:
            return mean


    def sample_y(self, X, n_samples=1, random_state=None):
        """
        sample model output at X
        """
        return self._inverse_transform(np.concatenate([gp.sample_y(X, n_samples=n_samples, random_state=random_state)[:, :, np.newaxis] for gp in self.GPs]) \
             +  [np.random.standard_normal((X.shape[0], n_samples, self.pca.n_components_ - self.npc))], axis=1)





if __name__ == '__main__':
    param_columns = ['alphaS', 'qhatMin', 'qhatSlope', 'qhatPower']
    obs_columns = ['CMS-0-100-Raa', 'CMS-30-50-v2', 'ALICE-0-10-Raa', 'ALICE-30-50-v2']

    ### read in data 
    df = pd.read_pickle('../Data_run90_PbPb5020_alphaS_afterUrQMD_CMS-ALICE_cumulant_RUN2_MIMIC_RUN3_forWK.pkl')
    X_train = df[param_columns]
    Y_train = [[] for i in range(100)]

    for idx in range(100):
        for col in obs_columns:
            Y_train[idx].extend(df[col][idx])

    Y_train = np.array(Y_train)

    
    X_min = np.array([0.1, 0.1, 0.0, 0.1])
    X_max = np.array([0.5, 7.0, 5.0, 0.6])

    ### create and train the emulator 
    #emulator = Emulator(X_train, Y_train, X_min, X_max, npc=6)
    emulator = Emulator(X_train, Y_train, npc=6)
    print('{} PCs (out of {} obs) explains {:.5f} of variance'.format(emulator.npc, emulator.nobs, emulator.pca.explained_variance_ratio_[:emulator.npc].sum()))

    print(X_train.shape)
    Y_train_predict = emulator.predict(X_train, return_cov=False)
    fig, axes = plt.subplots(1, 2, figsize=(8,4))
    for i in range(Y_train_predict.shape[0]):
    #for i in range(10):
        axes[0].plot(Y_train_predict[i], Y_train[i], 'o', markersize=3)
        axes[1].plot(Y_train_predict[i] / Y_train[i], 'o', markersize=3)
    axes[0].set_xlabel('model calculation')
    axes[0].set_xlabel('emulator prediction')

    axes[1].set_xlabel('observables')
    axes[1].set_ylabel('prediction / model calculation')
    axes[1].set_ylim(0.5, 1.5)
    plt.show()
