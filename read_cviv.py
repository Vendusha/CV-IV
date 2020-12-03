"""This script is contains functions to read the CVIV measurements
"""
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc
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


def plot_iv(v_b, _v_g, i_p, graph_label):
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


def plot_cv(v_d, capacitance, graph_label, v_dep=0):
    """Plotting of the CV characteristics
    """
    # try:

    _fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(v_d, capacitance,
             label=str(graph_label))
    ax1.set_xlabel("$V_{det}$ [V]")
    ax1.set_ylabel("C [pF]")
    ax1.legend()

    ax2.plot(v_d, 1/(capacitance**2),
             label=str(graph_label))
    ax2.set_xlabel("$V_{det}$ [V]")
    ax2.set_ylabel("1/C$^2$ [pF$^{-1}$)]")
    if v_dep != 0:
        ax2.axvline(v_dep, ymin=0.8, ymax=0.95, color='r', label="$v_{dep}$")

    ax2.legend()

    # except NameError:
    #     print("CV measurements not found.")
    #     plt.close()


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
