"""This script is contains functions to read the CVIV measurements
"""
from collections import namedtuple
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from scipy import interpolate
from cycler import cycler
mpl.rc('text', usetex=True)
mpl.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')


def find_start(file_name):
    """Finds the "BEGIN" line.
    """
    line_no = 0
    with open(file_name, 'r') as read_obj:
        for num, line in enumerate(read_obj):
            if ":start" in line:
                datetime = next(read_obj)
            elif "BEGIN" in line:
                line_no = num+2
                break
    try:
        return line_no, datetime
    except NameError:
        print("String BEGIN not found in file.")


def n_eff_fcn(v_d, d_t):
    """Calculates the effective doping of a silicon sample.
    Returns the doping in 10**(13). Takes arguments
    depletion voltage and thickness
    """
    eps_0 = 8.8541878128  # 10^{-12} F m^(-1)
    eps_si = 11.68  # relative permitivity of Silicon
    q_0 = 1.602  # in 10^(-19)
    return np.abs(2*eps_si*eps_0*v_d/(q_0*d_t**2))


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


def dep_voltage_curvature(v_det, capacitance):
    """finds the depletion voltage, based on https:/
    /stackoverflow.com/questions/29382903
    /how-to-apply-piecewise-linear-fit-in-python?rq=1
    second derivation method based on window from the initial guess
    """
    capacitance = 1/(capacitance**2)
    tck = interpolate.splrep(v_det, capacitance, k=2, s=0)
    dev_1 = interpolate.splev(v_det, tck, der=1)
    dev_2 = interpolate.splev(v_det, tck, der=2)
    curvature = np.abs(((1+dev_1**2)**1.5)/dev_2)
    turning_point_mask = np.amin(curvature)
    print(turning_point_mask)
    # turning_point_mask = dev_2 == np.amin(dev_2)
    xnew = np.linspace(v_det[0], v_det[len(v_det)-1])
    _fig, axes = plt.subplots(3)
    axes[0].plot(v_det, capacitance, 'x', label='data')
    axes[0].plot(xnew, interpolate.splev(xnew, tck, der=0), label='Fit')
    axes[1].plot(v_det, interpolate.splev(v_det, tck, der=1), label='1st dev')
    axes[2].plot(v_det, dev_2, label='2st dev')
    axes[2].plot(v_det[turning_point_mask], dev_2[turning_point_mask],
                 'rx', label='Turning point')
    for axis in axes:
        axis.legend(loc='best')
    plt.show()
    return v_det[turning_point_mask]


def select_slice_vdep(v_det, capacitance, percentage, v_dep_init):
    """Selects a percentage slice of an array
    around a given point, percentage should be given in
    interval (0,1)
    """
    indeces = int(percentage * len(v_det))
    middle_index = np.searchsorted(v_det, v_dep_init)[0]
    v_det = v_det[middle_index-indeces:middle_index+indeces]
    capacitance = capacitance[middle_index-indeces:middle_index+indeces]
    return (v_det, capacitance)


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
    # v_det, capacitance = select_slice_vdep(v_det, capacitance,
    #                                        0.15, v_depletion)
    # v_depletion = dep_voltage_curvature(v_det, capacitance)
    # v_depletion = dep_voltage_curvature(v_depletion)
    # read_cviv.plot_cv(v_det, 1/capacitance**2, "test", v_dep)
    # plt.show()
    return v_depletion


def read_iv(filename):
    """read the files IV extension, returns the bias, total current and pad current
    """
    no_line_start, _datetime = find_start(filename)
    data = np.genfromtxt(filename, dtype=float, skip_header=no_line_start,
                         skip_footer=1, unpack=True)
    return data


def read_cv(filename):
    """read the files IV extension, returns the detector Voltage, Capacitance,
    Conductivity, Bias and Current in Power Supply
    """
    no_line_start, _datetime = find_start(filename)
    data = np.genfromtxt(filename, dtype=float, skip_header=no_line_start,
                         skip_footer=1, unpack=True)
    return data


def read_cvf(filename):
    """read the files IV extension, returns the detector Voltage, Capacitance,
    Conductivity, Bias and Current in Power Supply
    """
    no_line_start, _datetime = find_start(filename)
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


def plot_cvf(data, graph_label, i_pad, thickness, date, v_bias):
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
    fig = plt.figure(figsize=(16, 6))
    plt.title(str(graph_label))
    plt.box(False)
    plt.tick_params(left=False, labelleft=False)
    plt.tick_params(bottom=False, labelbottom=False)
    for index, frequency in enumerate(data):
        cmap = plt.get_cmap("tab10")
        v_dep = dep_voltage(frequency.v_det, frequency.c, i_pad)
        # plot_cv(frequency.v_det, frequency.c, graph_label
        # + " "+str(frequency.frequency)+" Hz", v_dep)
        axis = fig.add_subplot(rows, cols, position[index])
        axis.plot(frequency.v_det, 1/(frequency.c**2),
                  label="CV "+str(frequency.frequency)+" Hz", color=cmap(index))
        # ax2._get_lines.get_next_color()
        axis.set_xlabel("$V_{bias}$ [V]")
        axis.set_ylabel("1/C$^2$ [F$^{-1}$)]")
        plt.legend()
        if v_dep != 0:
            axis.axvline(v_dep, ymin=0.8, ymax=0.95,
                         color='r', label="$v_{dep}$")
        frequency_array.append(frequency.frequency)
        n_eff.append(n_eff_fcn(v_dep, thickness)[0])
        v_dep_array.append(v_dep[0])
    axis = fig.add_subplot(rows, cols, position[n_total-2])
    axis.plot(v_bias, i_pad, label="IV")
    axis.set_xlabel(r"$V_{bias}$ [V]")
    axis.set_ylabel(r"$I_{pad}$ [A]")
    plt.legend()
    axis = fig.add_subplot(rows, cols, position[n_total-1])
    for index, frequency in enumerate(frequency_array):
        axis.plot(frequency_array[index], v_dep_array[index], 'x',
                  label=str(graph_label), color=cmap(index))
    axis.set_ylabel("$V_{depletion}$ [V]")
    axis.set_xlabel("Frequency [Hz]")
    axis.set_xscale("log")
    axis2 = axis.twinx()
    for index, frequency in enumerate(frequency_array):
        axis2.plot(frequency_array[index], n_eff[index], 'x',
                   label=str(graph_label), color=cmap(index))
    axis2.set_ylabel("Doping concentration [$10^{13}$ atoms/cm$^3$]")

    # axis.set_xlabel("Frequency [Hz]")
    # axis.set_xscale("log")
    # plt.legend()
    # plt.show()
    plt.savefig("results/"+str(graph_label)+"-" +
                date[0:2]+"-"+date[3:5]+"-" +
                date[6:8]+"CVF.png")
    return frequency_array, n_eff, v_dep
