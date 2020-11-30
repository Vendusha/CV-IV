"""This script is provides tools to analyze CVIV measurements
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc


rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)


def read_iv(filename):
    """read the files IV extension, returns the bias, total current and pad current
    """
    data = np.genfromtxt(filename, dtype=float, skip_header=65,
                         skip_footer=1, unpack=True)
    return data


def read_cv(filename):
    """read the files IV extension, returns the detector Voltage, Capacitance,
    Conductivity, Bias and Current in Power Supply
    """
    data = np.genfromtxt(filename, dtype=float, skip_header=70,
                         skip_footer=1, unpack=True)
    return data


def read_cvf(filename):
    """read the files IV extension, returns the detector Voltage, Capacitance,
    Conductivity, Bias and Current in Power Supply
    """
    data = np.genfromtxt(filename, dtype=float, skip_header=70,
                         skip_footer=1, unpack=True)
    return data


# FOLDER = "data/CIS16-EPI-09-50-DS-54/CIS16-EPI-09-50-DS-54_2020-11-26_1.cv"
DATA = "../data/CIS16-EPI-09-50-DS-54/CIS16-EPI-09-50-DS-54_2020-11-26_1.iv"
sample_name = os.path.basename(DATA)[0:21]
sample_graph_label = sample_name[6:12]  # prints for example EPI-09
print(sample_graph_label)
if DATA.endswith("iv"):
    V_bias, I_tot, I_pad = read_iv(DATA)
    V_gate = I_tot-I_pad
if DATA.endswith("cv"):
    V_det, C, Sigma, V_bias, I_ps = read_cv(DATA)
if DATA.endswith("cvf"):
    V_det, f, C, Sigma, V_bias, I_ps = read_cvf(DATA)

try:
    plt.figure()
    plt.plot(V_bias, I_pad, label="sample_graph_label")
    plt.xlabel("V_B [V]")
    plt.ylabel("I_{pad} [A]")
    # plt.show()
    plt.figure()
    plt.plot(V_gate, I_pad)
    plt.xlabel("V_B [V]")
    plt.ylabel("I_{gate} [A]")
except NameError:
    print("IV measurement not found.")
plt.show()

# try:
#     plt.figure()
#     plt.plot(V_bias, C, label="sample_graph_label")
#     plt.xlabel("V_B [V]")
#     plt.ylabel("I_{pad} [A]")
#     # plt.show()
#     plt.figure()
#     plt.plot(V_gate, I_pad)
#     plt.xlabel("V_B [V]")
#     plt.ylabel("I_{gate} [A]")
# except NameError:
#     print("CV measurement not found.")

