import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def read_Raa_Dmeson(folder, ycut, cen_range, urqmd='after', pT_range=(0,99)):
    """
    compute the Dmeson Raa for certain centrality
    Argument:
        folder -- str: path of the result files
        ievent -- int: event ID
        cen_range -- (int, int): tuple of (cen_low, cen_high)
        pT_range -- (float, float): tuple of (pT_low, pT_high)
    Returns:
        (pT, Raa)
    """

    pT, AAcN = np.loadtxt('PbPb5020/multi_AAcN_{}Decay_cD_y{}.dat'.format(urqmd, ycut), unpack=True)
    multiFile = '{}/multi_AAcY_{}UrQMD_cD_y{}_cen{}.dat'.format(folder, urqmd, ycut, cen_range)

    pT, AAcY = np.loadtxt(multiFile, unpack=True)
    ## PbPb 5020GeV cross-section(PbPb->c) / (pp->c)
    Raa = 0.7387133684626412 * AAcY / AAcN

    idxL = list(pT).index(pT_range[0])
    idxH = list(pT).index(pT_range[1])

    return (pT[idxL:idxH+1], Raa[idxL:idxH+1])




def read_v2_Dmeson(folder, ycut, cen_range, urqmd='after', case='cumulant', pT_range=(0,99)):
    """
    compute the Dmeson Raa for centain centrality
    Argument:
        folder -- str: path of the result files
        ievent -- int: event ID
        cen_range -- (int, int): tuple of (cen_low, cen_high)
        case -- 'EP2', 'cumulant'
    Returns:
        (pT, Raa)
    """

    if case == 'cumulant':
        v2File = '{}/v22_AAcY_{}UrQMD_cD_y{}_cen{}_cumulant.dat'.format(folder, urqmd, ycut, cen_range)
        pT, v2 = np.loadtxt(v2File, unpack=True)

    elif case == 'EP':
        v2File = '{}/v2_AAcY_{}UrQMD_cD_y{}_cen{}.dat'.format(folder, urqmd, ycut, cen_range)
        pT, v2 = np.loadtxt(v2File, unpack=True)
        v2 = -v2 


    idxL = list(pT).index(pT_range[0])
    idxH = list(pT).index(pT_range[1])

    return (pT[idxL:idxH+1], v2[idxL:idxH+1])




def read_Design(folder, ievent):
    """
    read in design points 
    Argument:   
        folder -- str: folder for the design points
        ievent -- int: event ID
    Returns:
        (alphaS, qhatMin, qhatSlope, qhatPower)
    """

    designFile = '{}/{}'.format(folder, str(ievent).zfill(3))

    with open(designFile, 'r') as f:
        inputline = f.readline().split()
        alphas = float(inputline[-1])

        inputline = f.readline().split()
        qhatMin  = float(inputline[2].split('=')[1])
        qhatSlope = float(inputline[3].split('=')[1])
        qhatPower = float(inputline[4].split('=')[1])

    return {'alphaS': alphas,
            'qhatMin': qhatMin,
            'qhatSlope': qhatSlope,
            'qhatPower': qhatPower}



def read_data(folder, experiment_obs, urqmd, v2Case):
    df_result = pd.DataFrame()
    expResult = {}

    eID = experiment_obs.get('ID')
    for ievent in range(100):
        ires = pd.Series(
                    read_Design('/var/phy/project/nukeserv/yx59/spring2018/run90_PbPb5020_alpha/design_run87/main', ievent))

        for cen_range in experiment_obs.get('Raa'):
            Raa_ycut = experiment_obs.get('Raa_ycut')
            ifolder = '{}/bmsap_{}'.format(folder, ievent)
            (dum_pT, dum_Raa) = read_Raa_Dmeson(ifolder, Raa_ycut, cen_range, urqmd)
            f = interp1d(dum_pT, dum_Raa)
            exp_data = np.loadtxt('JW_PbPb5020/{}-PbPb5020-Raa-{}.dat'.format(eID, cen_range))
            exp_data = pd.DataFrame(exp_data, columns = ['pT_L', 'pT_H', 'Raa', 'stat_err', 'sys_err'])
            exp_pT = 0.5 * (exp_data['pT_L'] + exp_data['pT_H'])
            ires['{}-{}-Raa-pT'.format(eID, cen_range)] = exp_pT.values
            ires['{}-{}-Raa'.format(eID, cen_range)] = f(exp_pT.values)


            ### rescale the uncertainty for the prejected experimental data from JW
            if ievent == 0:
                (dum_xl, dum_xh, dum_y) = np.loadtxt('Current_PbPb5020/{}-PbPb5020-Raa-{}.dat'.format(eID, cen_range), usecols=(0,1,2), unpack=True)
                dum_x = 0.5 * (dum_xl + dum_xh)
                f = interp1d(dum_x, dum_y)
                expResult.update({
                    '{}-{}-Raa-pTL'.format(eID, cen_range): exp_data['pT_L'],
                    '{}-{}-Raa-pTH'.format(eID, cen_range): exp_data['pT_H'],
                    '{}-{}-Raa'.format(eID, cen_range): f(exp_pT.values),
                    '{}-{}-Raa-sys'.format(eID, cen_range): exp_data['stat_err'] * f(exp_pT.values)/exp_data['Raa'],
                    '{}-{}-Raa-stat'.format(eID, cen_range): exp_data['sys_err'] * f(exp_pT.values)/exp_data['Raa']
                    }
                )


        for cen_range in experiment_obs.get('v2'):
            v2_ycut = experiment_obs.get('v2_ycut')
            ifolder = '{}/bmsap_{}'.format(folder, ievent)
            (dum_pT, dum_v2) = read_v2_Dmeson(ifolder, v2_ycut, cen_range, urqmd, v2Case)
            f = interp1d(dum_pT, dum_v2)
            exp_data = np.loadtxt('JW_PbPb5020/{}-PbPb5020-v2-{}.dat'.format(eID, cen_range))
            if exp_data.shape[1] == 5:
                exp_data = pd.DataFrame(exp_data, columns=['pT_L', 'pT_H', 'v2', 'stat_err', 'sys_err'])
            elif exp_data.shape[1] == 6:
                exp_data = pd.DataFrame(exp_data, columns=['pT_L', 'pT_H', 'v2', 'stat_err', 'sys_err', 'sys_err2'])
            exp_pT = 0.5 *(exp_data['pT_L'] + exp_data['pT_H'])
            ires['{}-{}-v2-pT'.format(eID, cen_range)] = exp_pT.values
            ires['{}-{}-v2'.format(eID, cen_range)] = f(exp_pT.values)


            if ievent == 0:
                (dum_xl, dum_xh, dum_y) = np.loadtxt('Current_PbPb5020/{}-PbPb5020-v2-{}.dat'.format(eID, cen_range), usecols=(0,1,2), unpack=True)
                dum_x = 0.5 * (dum_xl + dum_xh)
                f = interp1d(dum_x, dum_y)
                expResult.update({
                    '{}-{}-v2-pTL'.format(eID, cen_range): exp_data['pT_L'],
                    '{}-{}-v2-pTH'.format(eID, cen_range): exp_data['pT_H'],
                    '{}-{}-v2'.format(eID, cen_range): f(exp_pT.values),
                    '{}-{}-v2-sys'.format(eID, cen_range): exp_data['stat_err'] * f(exp_pT.values)/exp_data['v2'],
                    '{}-{}-v2-stat'.format(eID, cen_range): exp_data['sys_err'] * f(exp_pT.values)/exp_data['v2']
                    }
                )

        df_result = df_result.append(ires, ignore_index=True)

    return df_result, expResult



if __name__ == '__main__':
    CMS_obs = {'Raa': ['0-100'],
                'v2': ['30-50'],
                'ID': 'CMS',
                'Raa_ycut': 1.0,
                'v2_ycut': 1.0}

    ALICE_obs = {'Raa': ['0-10'],
                'v2': ['30-50'],
                'ID': 'ALICE',
                'Raa_ycut': 0.5,
                'v2_ycut': 0.8}



    urqmd = 'after'
    v2Case = 'cumulant'

    (result_CMS, expCMS) = read_data('/var/phy/project/nukeserv/yx59/spring2018/run90_PbPb5020_alpha/RESULT', CMS_obs, urqmd, v2Case)
    (result_ALICE, expALICE) = read_data('/var/phy/project/nukeserv/yx59/spring2018/run90_PbPb5020_alpha/RESULT', ALICE_obs, urqmd, v2Case)
   
    result = pd.merge(result_CMS, result_ALICE, on=['alphaS', 'qhatMin', 'qhatSlope', 'qhatPower'])
    expALICE.update(expCMS)
    df_exp = pd.Series(expALICE)

   
    result.to_pickle('Data_run90_PbPb5020_alphaS_{}UrQMD_CMS-ALICE_{}_RUN2_MIMIC_RUN3_forWK.pkl'.format(urqmd, v2Case))
    df_exp.to_pickle('Exp_run90_PbPb5020_CMS-ALICE_RUN2_MIMIC_RUN3_forWK.pkl')

   
    print(result.columns)
    print(df_exp.index)

    fig = plt.figure()
    for i in range(len(result)):
        dum = list(result['CMS-0-100-Raa'][i]) + list(result['CMS-30-50-v2'][i]) +  list(result['ALICE-0-10-Raa'][i]) +  list(result['ALICE-30-50-v2'][i])

        plt.plot(dum, 'g-', alpha=0.5, zorder=-1)

    plt.errorbar(range(len(dum)),
        list(df_exp['CMS-0-100-Raa']) + list(df_exp['CMS-30-50-v2']) + list(df_exp['ALICE-0-10-Raa']) + list(df_exp['ALICE-30-50-v2']),
        yerr = [
        list(df_exp['CMS-0-100-Raa-sys']) + list(df_exp['CMS-30-50-v2-sys']) + list(df_exp['ALICE-0-10-Raa-sys']) + list(df_exp['ALICE-30-50-v2-sys']),
        list(df_exp['CMS-0-100-Raa-stat']) + list(df_exp['CMS-30-50-v2-stat']) + list(df_exp['ALICE-0-10-Raa-stat']) + list(df_exp['ALICE-30-50-v2-stat'])
        ],
        color='r', fmt='o')

    plt.show()
