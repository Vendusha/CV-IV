"""This script is provides tools to analyze CVIV measurements
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from matplotlib import rc
import read_cviv


def listdir_fullpath(directory):
    """lists full path of a directory
    """
    return [os.path.join(directory, f) for f in os.listdir(directory)]


# rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)


def check_breakdown(v_det, capacitance, i_det):
    """Finds the point of breakdown accoring to https:/
    /cds.cern.ch/record/2281851/files/PabloMatorras2017.pdf
    """
    for i in range(1, len(i_det)):
        k_bd = (i_det[i]-i_det[i-1])/(v_det[i]-v_det[i-1])*(v_det[i]/i_det[i])
        if k_bd > 4:
            i_det = i_det[:i-1]
            v_det = v_det[:i-1]
            capacitance = capacitance[:i-1]
            break
    return v_det, capacitance


def dep_voltage(v_det, capacitance, i_det):
    """finds the depletion voltage, based on https:/
    /stackoverflow.com/questions/29382903
    /how-to-apply-piecewise-linear-fit-in-python?rq=1
    """
    v_det, capacitance = check_breakdown(v_det, capacitance, i_det)
    # extract depletion voltage
    dys = np.gradient(1/capacitance**2, v_det)
    rgr = DecisionTreeRegressor(max_leaf_nodes=2)  # fitting two segments
    rgr.fit(v_det.reshape(-1, 1), dys.reshape(-1, 1))
    dys_dt = rgr.predict(v_det.reshape(-1, 1)).flatten()
    ys_sl = np.ones(len(v_det)) * np.nan
    coeff = []
    for item in np.unique(dys_dt):
        msk = dys_dt == item
        lin_reg = LinearRegression()
        lin_reg.fit(v_det[msk].reshape(-1, 1),
                    (1/capacitance**2)[msk].reshape(-1, 1))
        ys_sl[msk] = lin_reg.predict(v_det[msk].reshape(-1, 1)).flatten()
        coeff.append(lin_reg.coef_[0], lin_reg.intercept_[0])
        # uncertainty.append
        # print(lin_reg._residues)
        # plt.plot([v_det[msk][0], v_det[msk][-1]],
        #        # [ys_sl[msk][0], ys_sl[msk][-1]],
        #        # color='r', zorder=1)
    v_depletion = np.abs((coeff[1]-coeff[3])/(coeff[0]-coeff[2]))
    # read_cviv.plot_cv(v_det, 1/capacitance**2, "test", v_dep)
    plt.show()
    return v_depletion


FOLDER = "../data/CIS16-EPI-09-50-DS-54/"
FOLDER = listdir_fullpath(FOLDER)
# ###########################IV FILES#######################
# ##########################################################
for DATA in FOLDER:
    sample_name = os.path.basename(DATA)[0:-16]
    sample_graph_label = sample_name[6:]  # prints for example EPI-09
    if DATA.endswith("iv"):
        V_bias, I_tot, I_pad = read_cviv.read_iv(DATA)
        V_bias = np.abs(V_bias)
        I_pad = np.abs(I_pad)
        I_tot = np.abs(I_tot)
        V_gate = I_tot-I_pad
        # read_cviv.plot_iv(V_bias, V_gate, I_pad, sample_graph_label)
        # plt.savefig(sample_graph_label+"IV.png")
for DATA in FOLDER:
    # ###########################CV FILES#######################
    # ##########################################################
    # elif DATA.endswith("cv"):
    #     V_det, C, Sigma, V_bias, I_ps = read_cviv.read_cv(DATA)
    #     V_det = np.abs(V_det)
    #     V_bias = np.abs(V_bias)
    #     I_ps = np.abs(I_ps)
    #     # d_C = []
    #     # for j in range(1, V_det):
    #     #     d_C.append((C[j]-C[j-1])/(V_det[j]-V_det[j-1]))
    #     read_cviv.plot_cv(V_bias, C, sample_name)
    #     # plt.savefig(sample_graph_label+"CV.png")
    # ###########################CVF FILES#######################
    # ##########################################################
    if DATA.endswith("cvf"):
        Frequency = []
        V_dep = []
        sample_name = os.path.basename(DATA)[0:-17]
        sample_graph_label = sample_name[6:]  # prints for example EPI-09
        Data = read_cviv.read_cvf(DATA)
        for frequency in Data:
            v_dep = dep_voltage(frequency.v_det, frequency.c, I_pad)
            read_cviv.plot_cv(frequency.v_det, frequency.c, sample_graph_label
                              + " "+str(frequency.frequency)+" Hz", v_dep)
            Frequency.append(frequency.frequency)
        plt.savefig(sample_graph_label+"CVF.png")
plt.close()
# plt.show()
