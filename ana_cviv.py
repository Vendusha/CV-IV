"""This script is provides tools to analyze CVIV measurements
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
import read_cviv


def listdir_fullpath(directory):
    """lists full path of a directory
    """
    return [os.path.join(directory, f) for f in os.listdir(directory)]


# rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)


FOLDER = "../data/CIS16-EPI-09-50-DS-54"
BASENAME = (os.path.split(FOLDER))[-1]
if BASENAME[15] == "-":
    thickness = int(BASENAME[13:15])
else:
    thickness = int(BASENAME[13:16])
FOLDER = listdir_fullpath(FOLDER)
sample_graph_label = BASENAME[6:]  # prints for example EPI-09
# ###########################IV FILES#######################
# ##########################################################
for DATA in FOLDER:
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
        Data = read_cviv.read_cvf(DATA)
        Frequency, N_eff, V_dep = read_cviv.plot_cvf(Data,
                                                     sample_graph_label,
                                                     I_pad, thickness)
        # plt.savefig(sample_graph_label+"CVF.png")
plt.close()
# plt.show()
