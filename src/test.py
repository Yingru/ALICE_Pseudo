import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from gaussian_emulator import Emulator
from MCMC import Chain



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = 'Bayesian Calibration using multi-dimensional Gaussian process'
    )

    parser.add_argument('--npc', type=int, help='number of PCs')
    parser.add_argument('--nsteps', type=int, help='number of MC steps')
    parser.add_argument('--nwalkers', type=int, help='number of walkers')
    parser.add_argument('--cov', type=bool, help='including cov error or not')

    args = parser.parse_args()

    if args.npc is None:
        print('No npc provided, using default npc = 8')
        args.npc = 6

    if args.nsteps is None:
        print('No nsteps provided, using default nsteps = 1000')
        args.nsteps = 1000

    if args.nwalkers is None:
        print('No nwalkers provided, using default nwalker = 10')
        args.nwalkers = 100

    if args.cov is None:
        print('No cov provided, using default cov=False')
        args.cov = False

    covKeys = {True: 1, False: 0, None: 0}


    #### >>>>>>>>>>>>>>>>>>>>>>>>>>>  step 1: loading in training data file
    param_columns = ['alphaS', 'qhatMin', 'qhatSlope', 'qhatPower']
    obs_columns = ['CMS-0-100-Raa', 'CMS-30-50-v2', 'ALICE-0-10-Raa', 'ALICE-30-50-v2']

    df = pd.read_pickle('../Data_run90_PbPb5020_alphaS_afterUrQMD_CMS-ALICE_cumulant_RUN2_MIMIC_RUN3_forWK.pkl')
    X_train = df[param_columns]
    Y_train = [[] for i in range(100)]

    for idx in range(100):
        for col in obs_columns:
            Y_train[idx].extend(df[col][idx])

    Y_train = np.array(Y_train)

    X_min = np.array([0.1, 0.1, 0.0, 0.1])
    X_max = np.array([0.5, 7.0, 5.0, 0.6])


    
    #### >>>>>>>>>>>>>>>>>>>>>>>>>  step 2: loading the experimental data file 
    df_exp = pd.read_pickle('../Exp_run90_PbPb5020_CMS-ALICE_RUN2_MIMIC_RUN3_forWK.pkl')

    Y_exp = []
    stat_error = []
    sys_error = []
    for col in obs_columns:
        Y_exp.extend(df_exp[col])
        stat_error.extend(df_exp['{}-stat'.format(col)])
        sys_error.extend(df_exp['{}-sys'.format(col)])

    Y_exp = np.array(Y_exp)
    stat_error = np.array(stat_error)
    sys_error = np.array(sys_error)

    fig = plt.figure()
    for i in range(len(Y_train)):
        plt.plot(Y_train[i], 'b-', alpha=0.5)

    plt.errorbar(range(len(Y_exp)), Y_exp, yerr=[stat_error, sys_error], fmt='o', markersize=4, color='red')
    plt.show()

    #### >>>>>>>>>>>>>>>>>>>>>>>>>>> step 3: create and train the emulator
    emulator = Emulator(X_train, Y_train,
                        X_train_min = X_min,
                        X_train_max = X_max,
                        npc = args.npc)
    nwalkers = args.nwalkers * emulator.ndim 
    print('{} PCs (out of {} obs) explains {:.5f} of variance'.format(emulator.npc, emulator.nobs, emulator.pca.explained_variance_ratio_[:emulator.npc].sum()))



    #### >>>>>>>>>>>>>>>>>>>>>>>>>> step 4: perform the MCMC random walk and calibrate to experimental value
    outputFile = 'posterior/test.hdf5'
    BAchain = Chain(emulator, Y_exp = Y_exp,
                              Y_err = (sys_error, stat_error),
                              X_min = np.array(list(X_min) + [0.]),
                              X_max = np.array(list(X_max) + [1.]), 
                              cov_on = args.cov)

    BAchain.calibrate(nwalkers = nwalkers,
                      nsteps = args.nsteps,
                      outputFile = outputFile)




    (test, testPred) = BAchain.samples(outputFile = outputFile, n = 200)
    print(test.shape)

    fig = plt.figure()
    for i in range(len(testPred)):
        plt.plot(testPred[i], 'g-', zorder=-5, alpha=0.1)

    plt.errorbar(range(len(Y_exp)), Y_exp, yerr=[stat_error, sys_error], fmt='o', markersize=4, color='red')
    plt.show()


    
