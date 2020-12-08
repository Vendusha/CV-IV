"""This script is provides tools to analyze CVIV measurements
"""
import os
import numpy as np
from matplotlib import rc
import read_cviv


def listdir_fullpath(directory):
    """lists full path of a directory
    """
    return [os.path.join(directory, f) for f in os.listdir(directory)]


# rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)


def data_ana(sample_folder):
    """Analyses a folder of a sample generated
    by the CVIV labview.
    """
    print(sample_folder)
    basename = (os.path.split(sample_folder))[-1]
    if basename[15] == "-":
        thickness = int(basename[13:15])
    else:
        thickness = int(basename[13:16])
    sample_folder = listdir_fullpath(sample_folder)
    sample_graph_label = basename[6:]  # prints for example EPI-09
    # ###########################IV FILES#######################
    # ##########################################################
    for data in sample_folder:
        if data.endswith("iv"):
            v_bias, i_tot, i_pad = read_cviv.read_iv(data)
            v_bias = np.abs(v_bias)
            i_pad = np.abs(i_pad)
            i_tot = np.abs(i_tot)
            # v_gate = i_tot-i_pad
            # read_cviv.plot_iv(V_bias, V_gate, I_pad, sample_graph_label)
            # plt.savefig(sample_graph_label+"IV.png")
    for data in sample_folder:
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
        if data.endswith("cvf"):
            _no_line, date = read_cviv.find_start(data)
            data_frequency = read_cviv.read_cvf(data)
            _Frequency, _N_eff, _V_dep = read_cviv.plot_cvf(data_frequency,
                                                         sample_graph_label,
                                                         i_pad, thickness,
                                                            date, v_bias)
            # plt.savefig(sample_graph_label+"CVF.png")
    return


DIRECTORY = "../data"
DIRECTORY = [f.path for f in os.scandir(DIRECTORY) if f.is_dir()]
for folder in DIRECTORY:
    data_ana(folder)
