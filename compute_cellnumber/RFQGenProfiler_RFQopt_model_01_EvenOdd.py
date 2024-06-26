import copy
import random
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.special import iv as bessel1
import scipy.constants as consts
from particles import IonSpecies
import matplotlib.pyplot as plt

CLIGHT = consts.c

datafile = r"../data/full_with_cellnumber/RFQopt_model_01_Dataset_05_217k-samples.txt"
# df = pd.read_csv(datafile, delimiter=r"\s+", nrows=10)
df = pd.read_csv(datafile, delimiter=r"\s+")
df.drop(columns=df.columns[14], inplace=True)

# Fixed parameters
SetNumberOfDataPoints = 100  # number of data points in IN4 input table
RMSSetNumberOfDataPoints = 20  # number of RMS data points in IN4 input table
Xref = 180  # reference x-position for function parametrization [cm]
Xmax = 180  # maximum length of RFQ electrodes [cm]
Voltage = 0.022  # vane voltage [MV]
LFringeField = 1.4  # fringe field length [cm]
FieldMagnitude = 0.001084  # entrance gap field magnitude [MV]
NumberRMScells = 4  # number of RMS cells
Win = 15  # input energy [keV]
e = 1.602177e-19  # elemenary charge [C]
mp = 1.672622e-27  # proton mass [kg]
f = 32.8e6  # RFQ frequency [Hz]
NumPart = 40000  # Number of simulation particles in Parmteqm

# Beam
Alpha = 2.1
Beta = 17.0
# Phase = -91.86021751

ion = IonSpecies("H2+", 1.0, mass_mev=2.0 * 931.5, a=2, z=2, q=1)
ion.calculate_from_energy_mev(Win * 1e-3 / ion.a())

BetaLambdaHalf = np.sqrt(2 * Win * 1000 * e / (2 * mp)) / (2 * f)  # [m]
LRMS = NumberRMScells * BetaLambdaHalf * 100  # [cm]
RMSX1 = -(LRMS * 1.01)
RMSX3 = -1.0e-5
RMSBmin = 0.01

difference = []

for index, row in df.iterrows():
    # index, row = 14053, df.loc[14053]
    # print(row)
    # exit()
    # if True:
    ion.calculate_from_energy_mev(Win * 1e-3 / ion.a())
    print("Working on row {}".format(index))

    if row["Length"] >= Xmax:
        print("Skipping due to length")
        continue

    if row["Transmission"] < 50:
        print("Skipping due to transmission")
        continue

    Bmax = row["Bmax"]
    mX1 = row["mX1"]
    mX2 = row["mX2"]
    mY1 = row["mY1"]
    mY2 = row["mY2"]
    mtau1 = row["mtau1"]
    mtau2 = row["mtau2"]
    PhiY1 = row["PhiY1"]
    PhiY2 = row["PhiY2"]
    Phitau1 = row["Phitau1"]
    Phitau2 = row["Phitau2"]
    mY3ref = row["mY3ref"]
    PhiY3ref = row["PhiY3ref"]
    Eref = row["Eref"]

    print(row["Length"])

    PhiX1 = mX1
    PhiX2 = mX2

    num_cells_og = row["#Cells"]

    # generate IN4 table data (RFQ):
    # ------------------------------

    # Z-column:
    Zstep = Xmax / (SetNumberOfDataPoints - 1)
    Zet = []
    for loopcount in range(0, (SetNumberOfDataPoints + 1)):
        Zet.append((loopcount - 1) * Zstep)

    # B-column:
    B = []
    for loopcount in range(0, (SetNumberOfDataPoints + 1)):
        B.append(Bmax)

    B_interp = interp1d(np.array(Zet) * 0.01, B)

    # m-column:
    m = []
    for loopcount in range(0, (SetNumberOfDataPoints + 1)):
        if Zet[loopcount] <= mX1:
            m.append(((mY1 - 1) / mX1) * Zet[loopcount] + 1)
        if Zet[loopcount] > mX1:
            if Zet[loopcount] <= mX2:
                A2 = (mY1 - mY2) / (np.exp(mX1 / mtau1) - np.exp(mX2 / mtau1))
                A1 = mY2 - A2 * np.exp(mX2 / mtau1)
                m.append(A1 + A2 * np.exp(Zet[loopcount] / mtau1))
        if Zet[loopcount] > mX2:
            A2 = (mY2 - mY3ref) / (np.exp(mX2 / mtau2) - np.exp(Xref / mtau2))
            A1 = mY3ref - A2 * np.exp(Xref / mtau2)
            m.append(A1 + A2 * np.exp(Zet[loopcount] / mtau2))

    m_interp = interp1d(np.array(Zet) * 0.01, m)

    # Phi-column:
    Phi = []
    for loopcount in range(0, (SetNumberOfDataPoints + 1)):
        if Zet[loopcount] <= PhiX1:
            Phi.append(((PhiY1 + 90) / PhiX1) * Zet[loopcount] - 90)
        if Zet[loopcount] > PhiX1:
            if Zet[loopcount] <= PhiX2:
                A2 = (PhiY1 - PhiY2) / (
                    np.exp(PhiX1 / Phitau1) - np.exp(PhiX2 / Phitau1)
                )
                A1 = PhiY2 - A2 * np.exp(PhiX2 / Phitau1)
                Phi.append(A1 + A2 * np.exp(Zet[loopcount] / Phitau1))
        if Zet[loopcount] > PhiX2:
            A2 = (PhiY2 - PhiY3ref) / (np.exp(PhiX2 / Phitau2) - np.exp(Xref / Phitau2))
            A1 = PhiY3ref - A2 * np.exp(Xref / Phitau2)
            Phi.append(A1 + A2 * np.exp(Zet[loopcount] / Phitau2))

    phi_interp = interp1d(np.array(Zet) * 0.01, Phi)

    # my_rfq = PyRFQ(ion, Voltage, f)
    l_tot = 0.0
    cell_ct = 0
    tt = np.pi / 4.0  # RFQ Transit time factor
    lam = CLIGHT / f  # RF wavelength (m)
    v = Voltage  # inter-vane voltage (MV)

    # print(ion.total_kinetic_energy_mev())
    # print(Eref)

    # print("Cell    Wsyn     Phi       a       m         B         L         Z")

    while ion.total_kinetic_energy_mev() < Eref:
        cell_ct += 1
        _ion = copy.deepcopy(ion)

        # for i in range(5):
        ll = _ion.beta() * lam / 2.0

        # zz = l_tot  # Gather values at cell start
        zz = l_tot + ll * 0.5  # Gather values at cell center
        # zz = l_tot + ll  # Gather values at cell end

        mm = m_interp(zz)
        pp = np.deg2rad(phi_interp(zz))
        bb = B_interp(zz)

        r_0 = np.sqrt(
            ion.q() * v * lam**2.0 / (bb * ion.mass_mev())
        )  # radius at center of cell
        a = (
            2.0 * r_0 / (mm + 1.0)
        )  # TODO: Cave, this is not true for two term potential contour, but maybe good enough approx.?
        k = 2.0 * np.pi / _ion.beta() / lam  # 'wave number'
        aa = (mm**2.0 - 1.0) / (
            mm**2.0 * bessel1(0.0, k * a) + bessel1(0.0, k * mm * a)
        )
        e_0 = 2.0 * aa * v / _ion.beta() / lam
        dwdz = ion.q() * e_0 * tt * np.cos(pp)

        # energy = ion.total_kinetic_energy_mev() # Start
        # energy = ion.total_kinetic_energy_mev() + dwdz * ll * 0.5  # Center
        # energy = ion.total_kinetic_energy_mev() + dwdz * ll  # End
        # _ion.calculate_from_energy_mev(energy / ion.a())

        # Phi a m B L Z

        # print("{}      {:2.4f}      {:2.2f}    {:2.4f}      {:2.4f}      {:2.4f}     {:2.4f}     {:2.2f}".format(cell_ct + 4, _ion.total_kinetic_energy_mev(),
        #                                                                                                          np.rad2deg(pp), a*100, mm, bb, ll*100, (l_tot + ll)*100))

        # print(" {} with L = {} cm, m = {}, B = {}, Phi = {} Deg".format(cell_ct + 4, ll * 100, mm, bb, pp))
        #
        # print("Energy gain: {} keV/m, new ion energy = {} keV".format(dwdz*1e3, ion.total_kinetic_energy_mev()*1e3))

        # my_rfq.append_cell("regular",
        #                     voltage=v,
        #                     Wsyn=_ion.total_kinetic_energy_mev(),
        #                     Phi=pp,
        #                     a=a,
        #                     m=mm,
        #                     L=ll)

        # End of this cell/beginning of next:
        energy_cell = ion.total_kinetic_energy_mev() + dwdz * ll
        ion.calculate_from_energy_mev(energy_cell / ion.a())

        l_tot += ll

    print(
        "Number of cells OG = {:.0f}, number of cells this calc = {:.0f}".format(
            num_cells_og, cell_ct + NumberRMScells + 1
        )
    )
    difference.append(cell_ct + NumberRMScells + 1 - num_cells_og)

d = np.array(difference)

plt.hist(d, bins=[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
plt.title("(Predicted - Original) #Cells")
plt.show()

print(len(d[np.where(d == 0)]) / len(d))
