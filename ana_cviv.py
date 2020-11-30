"""This script is provides tools to analyze CVIV measurements
"""
import os
from collections import namedtuple
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
        v_det = data[0][i::no_of_frequencies]
        capacitance = data[2][i::no_of_frequencies]
        sigma = data[3][i::no_of_frequencies]
        v_bias = data[4][i::no_of_frequencies]
        i_ps = data[5][i::no_of_frequencies]
        frequency = frequency_object(v_det=v_det, frequency=data[1][i],
                                     c=capacitance, sigma=sigma, v_bias=v_bias,
                                     i_ps=i_ps)
        frequency_array.append(frequency)
    return frequency_array


def plot_iv(v_b, v_g, i_p):
    """Plotting of the IV characteristics
    """
    try:
        plt.figure()
        plt.plot(v_b, i_p, label="sample_graph_label")
        plt.xlabel(r"$V_B$ [V]")
        plt.ylabel(r"$I_{pad}$ [A]")
        plt.figure()
        plt.plot(v_g, i_p)
        plt.xlabel(r"$V_B$ [V]")
        plt.ylabel(r"$I_{gate}$ [A]")
    except NameError:
        print("IV measurement not found.")
        plt.close()


def plot_cv(v_b, capacitance):
    """Plotting of the CV characteristics
    """
    try:
        plt.figure()
        plt.plot(v_b, capacitance, label="sample_graph_label")
        plt.xlabel("$V_B$ [V]")
        plt.ylabel("Capacitance [A]")
    except NameError:
        print("CV measurement not found.")
        plt.close()


def plot_cvf(data_frequency):
    """Plotting of the CVF characteristics
    """
    try:
        plt.figure()
        for freq in data_frequency:
            plt.plot(freq.v_bias, freq.c,
                     label=freq.frequency)
        plt.xlabel("$V_{Bias}$ [V]")
        plt.ylabel("Capacitance [A]")
        plt.legend()
        plt.show()

        plt.figure()
        for freq in data_frequency:
            plt.plot(freq.v_bias, 1/(freq.c**2),
                     label=freq.frequency)
        plt.xlabel("$V_{Bias}$ [V]")
        plt.ylabel("1/C^2 [A]")
        plt.legend()

    except NameError:
        print("CVF measurements not found.")
        plt.close()


# FOLDER = "data/CIS16-EPI-09-50-DS-54/CIS16-EPI-09-50-DS-54_2020-11-26_1.cv"
# DATA = "../data/CIS16-EPI-09-50-DS-54/CIS16-EPI-09-50-DS-54_2020-11-26_1.iv"
DATA = "../data/CIS16-EPI-09-50-DS-54/CIS16-EPI-09-50-DS-54_2020-11-26_1.cvf"
sample_name = os.path.basename(DATA)[0:21]
sample_graph_label = sample_name[6:12]  # prints for example EPI-09
print(sample_graph_label)

if DATA.endswith("iv"):
    V_bias, I_tot, I_pad = read_iv(DATA)
    V_gate = I_tot-I_pad
    d_I = []
    for i in range(1, V_bias):
        d_I.append((I_pad[i]-I_pad[i-1])/(V_bias[i]-V_bias[i-1]))
    plot_iv(V_bias, V_gate, I_pad)
elif DATA.endswith("cv"):
    V_det, C, Sigma, V_bias, I_ps = read_cv(DATA)
    d_C = []
    for i in range(1, V_bias):
        d_C.append((C[i]-C[i-1])/(V_det[i]-V_det[i-1]))
    plot_cv(V_bias, C)
elif DATA.endswith("cvf"):
    Data = read_cvf(DATA)
    plot_cvf(Data)
plt.show()
