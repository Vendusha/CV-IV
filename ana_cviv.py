"""This script is provides tools to analyze CVIV measurements
"""
import os
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

# rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)


def dep_voltage(v_det, capacitance, i_det):
    """finds the depletion voltage, based on
    http://cds.cern.ch/record/2639974/files/
    automatic-method-extracting-final-Martin-Petek.pdf
    Another approach is by Pablo: https://cds.cern.ch/
    record/2281851/files/PabloMatorras2017.pdf
    """
    # Setting Pablo approach as default
    plot_cv(v_det, capacitance, "test")
    v_det_new = []
    try:
        i = 1
        while True:
            k_bd = (i_det[i]-i_det[i-1])/(v_det[i]-v_det[i-1])*(v_det[i]/i_det[i])
            i += 1
            if k_bd > 4:
                print(k_bd)
                print(i)
                if i > 20000:
                    break
    except NameError:
        print("The cv measurements are not nice.")
    return v_det, capacitance


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
    data = np.genfromtxt(filename, dtype=float, skip_header=76,
                         skip_footer=1, unpack=True)
    frequency_object = namedtuple("Frequency_object",
                                  'v_det, frequency,c, sigma, v_bias, i_ps')
    no_of_frequencies = 0
    freq_init = data[1][0]
    frequency_array = []
    while True:
        no_of_frequencies += 1
        if data[1][no_of_frequencies] == freq_init:
            break
    for i in range(no_of_frequencies):
        v_det = np.abs(data[0][i::no_of_frequencies])
        capacitance = data[2][i::no_of_frequencies]
        sigma = data[3][i::no_of_frequencies]
        v_bias = np.abs(data[4][i::no_of_frequencies])
        i_ps = data[5][i::no_of_frequencies]
        frequency = frequency_object(v_det=v_det, frequency=data[1][i],
                                     c=capacitance, sigma=sigma, v_bias=v_bias,
                                     i_ps=i_ps)
        frequency_array.append(frequency)
    return frequency_array


def plot_iv(v_b, v_g, i_p, graph_label):
    """Plotting of the IV characteristics
    """
    try:
        plt.figure()
        plt.plot(v_b, i_p, label=graph_label)
        plt.xlabel(r"$V_B$ [V]")
        plt.ylabel(r"$I_{pad}$ [A]")
        plt.legend()
        # plt.figure()
        # plt.plot(v_g, i_p, label=graph_label)
        # plt.xlabel(r"$V_B$ [V]")
        # plt.ylabel(r"$I_{gate}$ [A]")
        # plt.legend()
    except NameError:
        print("IV measurement not found.")
        plt.close()


def plot_cv(v_d, capacitance, graph_label):
    """Plotting of the CV characteristics
    """
    try:

        _fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(v_d, c*10**(12),
                 label=str(graph_label))
        ax1.set_xlabel("$V_{det}$ [V]")
        ax1.set_ylabel("C [pF]")
        ax1.legend()

        ax2.plot(v_d, 1/((capacitance*10**(12))**2),
                 label=str(graph_label))
        ax2.set_xlabel("$V_{det}$ [V]")
        ax2.set_ylabel("1/C$^2$ [pF$^{-1}$)]")
        ax2.legend()

    except NameError:
        print("CV measurements not found.")
        plt.close()


def plot_cvf(data_frequency, graph_label):
    """Plotting of the CVF characteristics
    """
    try:

        _fig, (ax1, ax2) = plt.subplots(1, 2)
        # fig.suptitle('Horizontally stacked subplots')
        # ax1.plot(x, y)
        # ax2.plot(x, -y)
        for freq in data_frequency:
            ax1.plot(freq.v_det, freq.c*10**(12),
                     label=str(freq.frequency)+" Hz")
        ax1.set_xlabel("$V_{det}$ [V]")
        ax1.set_ylabel("C [pF]")
        ax1.legend()

        for freq in data_frequency:
            ax2.plot(freq.v_det, 1/((freq.c*10**(12))**2),
                     label=str(freq.frequency)+" Hz "+str(graph_label))
        ax2.set_xlabel("$V_{det}$ [V]")
        ax2.set_ylabel("1/C$^2$ [pF$^{-1}$)]")
        ax2.legend()

    except NameError:
        print("CVF measurements not found.")
        plt.close()


# FOLDER = "data/CIS16-EPI-09-50-DS-54/CIS16-EPI-09-50-DS-54_2020-11-26_1.cv"
# DATA = "../data/CIS16-EPI-09-50-DS-54/CIS16-EPI-09-50-DS-54_2020-11-26_1.iv"
# DATA = "../data/CIS16-EPI-04-100-DS-95/CIS16-EPI-04-100-DS-95_2020-11-30_1.cvf"
# DATA = "../data/CIS16-EPI-09-50-DS-60/CIS16-EPI-09-50-DS-60_2020-12-01_2.iv"
DATA = "../data/CIS16-EPI-15-100-DS-61/CIS16-EPI-15-100-DS-61_2020-12-01_1.cvf"
sample_name = os.path.basename(DATA)[0:-17]
# sample_graph_label = sample_name[6:12]  # prints for example EPI-09
sample_graph_label = sample_name[6:]  # prints for example EPI-09
print(sample_graph_label)
# ###########################IV FILES#######################
# ##########################################################
if DATA.endswith("iv"):
    V_bias, I_tot, I_pad = read_iv(DATA)
    V_bias = np.abs(V_bias)
    I_pad = np.abs(I_pad)
    I_tot = np.abs(I_tot)
    V_gate = I_tot-I_pad
    d_I = []
    # for j in range(1, V_bias):
        # d_I.append((I_pad[j]-I_pad[j-1])/(V_bias[j]-V_bias[j-1]))
    plot_iv(V_bias, V_gate, I_pad, sample_graph_label)
    plt.savefig(sample_graph_label+"IV.png")
# ###########################CV FILES#######################
# ##########################################################
elif DATA.endswith("cv"):
    V_det, C, Sigma, V_bias, I_ps = read_cv(DATA)
    V_det = np.abs(V_det)
    V_bias = np.abs(V_bias)
    d_C = []
    for j in range(1, V_det):
        d_C.append((C[j]-C[j-1])/(V_det[j]-V_det[j-1]))
    plot_cv(V_bias, C, sample_name)
    plt.savefig(sample_graph_label+"CV.png")
# ###########################CVF FILES#######################
# ##########################################################
elif DATA.endswith("cvf"):
    Data = read_cvf(DATA)
    # frequency = Data[1]
    # dep_voltage(frequency.v_det, frequency.c, frequency.i_ps)
    plot_cvf(Data, sample_graph_label)
    plt.savefig(sample_graph_label+"CVF.png")
plt.show()
