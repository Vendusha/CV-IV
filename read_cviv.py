"""This script is contains functions to read the CVIV measurements
"""
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from matplotlib import rc
rc('text', usetex=True)


def find_start(file_name):
    """Finds the "BEGIN" line.
    """
    line_no = 0
    with open(file_name, 'r') as read_obj:
        for num, line in enumerate(read_obj):
            if "BEGIN" in line:
                line_no = num+1
                break
    try:
        return line_no
    except NameError:
        print("String BEGIN not found in file.")


def n_eff_fcn(v_depletion, d_thickness):
    """Calculates the effective doping of a silicon sample.
    Returns the doping in 10**(-7).
    """
    epsilon_0 = 8.8541878128  # 10^{-12} F m^(-1)
    epsilon_si = 11.68  # relative permitivity
    q_0 = 1.602  # in 10^(-19)
    return np.abs(2*epsilon_si*epsilon_0**2*v_depletion/(q_0*d_thickness**2))


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
        coeff.extend([lin_reg.coef_[0], lin_reg.intercept_[0]])
        # uncertainty.append
        # print(lin_reg._residues)
        # plt.plot([v_det[msk][0], v_det[msk][-1]],
        #        # [ys_sl[msk][0], ys_sl[msk][-1]],
        #        # color='r', zorder=1)
    v_depletion = np.abs((coeff[1]-coeff[3])/(coeff[0]-coeff[2]))
    # read_cviv.plot_cv(v_det, 1/capacitance**2, "test", v_dep)
    # plt.show()
    return v_depletion


def read_iv(filename):
    """read the files IV extension, returns the bias, total current and pad current
    """
    no_line_start = find_start(filename)
    data = np.genfromtxt(filename, dtype=float, skip_header=no_line_start,
                         skip_footer=1, unpack=True)
    return data


def read_cv(filename):
    """read the files IV extension, returns the detector Voltage, Capacitance,
    Conductivity, Bias and Current in Power Supply
    """
    no_line_start = find_start(filename)
    data = np.genfromtxt(filename, dtype=float, skip_header=no_line_start,
                         skip_footer=1, unpack=True)
    return data


def read_cvf(filename):
    """read the files IV extension, returns the detector Voltage, Capacitance,
    Conductivity, Bias and Current in Power Supply
    """
    no_line_start = find_start(filename)
    data = np.genfromtxt(filename, dtype=float, skip_header=no_line_start,
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
    _fig, (axis1, axis2) = plt.subplots(1, 2)
    axis1.plot(v_d, capacitance,
               label=str(graph_label))
    axis1.set_xlabel("$V_{det}$ [V]")
    axis1.set_ylabel("C [pF]")
    axis1.legend()

    axis2.plot(v_d, 1/(capacitance**2),
               label=str(graph_label))
    axis2.set_xlabel("$V_{det}$ [V]")
    axis2.set_ylabel("1/C$^2$ [pF$^{-1}$)]")
    if v_dep != 0:
        axis2.axvline(v_dep, ymin=0.8, ymax=0.95,
                      color='r', label="$v_{dep}$")
    axis2.legend()


def plot_cvf(data, graph_label, i_pad, thickness):
    """Plotting of the CVF characteristics
    """
    frequency_array = []
    v_dep_array = []
    n_eff = []
    n_total = len(data) + 2
    cols = 3
    rows = int(n_total/cols)
    if n_total % cols > 0:
        rows += 1
    position = range(1, n_total + 1)
    fig = plt.figure(figsize=(15, 6))
    plt.title(label=str(graph_label))
    for index, frequency in enumerate(data):
        v_dep = dep_voltage(frequency.v_det, frequency.c, i_pad)
        # plot_cv(frequency.v_det, frequency.c, graph_label
        # + " "+str(frequency.frequency)+" Hz", v_dep)
        axis = fig.add_subplot(rows, cols, position[index])
        axis.plot(frequency.v_det, 1/(frequency.c**2),
                  label=str(frequency.frequency)+" Hz")
        axis.set_xlabel("$V_{det}$ [V]")
        axis.set_ylabel("1/C$^2$ [F$^{-1}$)]")
        if v_dep != 0:
            axis.axvline(v_dep, ymin=0.8, ymax=0.95,
                         color='r', label="$v_{dep}$")
        frequency_array.append(frequency.frequency)
        n_eff.append(n_eff_fcn(v_dep, thickness)[0])
        v_dep_array.append(v_dep[0])
    axis = fig.add_subplot(rows, cols, position[n_total-2])
    axis.plot(frequency_array, v_dep_array, 'x',
              label=str(graph_label))
    axis.set_ylabel("$V_{dep}$ [V]")
    axis.set_xlabel("Frequency [Hz]")
    axis = fig.add_subplot(rows, cols, position[n_total-1])
    axis.plot(frequency_array, n_eff, 'x',
              label=str(graph_label))
    axis.set_ylabel("Doping concentration [$10^7$] atoms")
    axis.set_xlabel("Frequency [Hz]")
    plt.legend()
    plt.savefig(str(graph_label)+"CVF.png")
    plt.show()
    return frequency_array, n_eff, v_dep
