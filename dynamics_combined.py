import numpy as np
from pyatmos import nrlmsise00, coesa76
from scipy.interpolate import pchip_interpolate

# constants
g = 9.81
R = 287.1
specific_heat_ratio_air = 1.4
data0 = coesa76(0)
rho0 = data0.rho
# T0 = data0.T
# P0 = data0.P
# a0 = np.sqrt(gamma * R * T0)

# pursuer and evader surface areas (m^2)
SP = 2.3
SE = 28

# pursuer and evader masses (kg)
mP = 130
mE = 10000

# pursuer and evader drag coefficient vs Mach number data
MiP = np.array([0, 0.6, 0.8, 1, 1.2, 2, 3, 4, 5])
CdiP = np.array([0.016, 0.016, 0.0195, 0.045, 0.039, 0.0285, 0.024, 0.0215, 0.020])
MiE = np.array([0, 0.9, 1, 1.2, 1.6, 2])
CdiE = np.array([0.0175, 0.019, 0.05, 0.045, 0.043, 0.038])

evader_thrust0 = 76310


# turbojet thrust profile
def turbojet_thrust(T0, rho):
    return (rho / rho0) * T0


# boost-sustain solid rocket motor thrust profile
def SRM_thrust(t):
    if t < 10:
        return 10000
    elif t < 30:
        return 1800
    else:
        return 0


def dynamics_combined(x, t, nzE, nzP):
    dx = np.zeros(8)
    VP, gammaP, hP, dP, VE, gammaE, hE, dE = x

    # air density and speed of sound at current altitudes
    data_P = coesa76(hP / 1000)
    rhoP = data_P.rho
    TP = data_P.T
    PP = data_P.P
    aP = np.sqrt(specific_heat_ratio_air * R * TP)
    data_E = coesa76(hE / 1000)
    rhoE = data_E.rho
    TE = data_E.T
    PE = data_E.P
    aE = np.sqrt(specific_heat_ratio_air * R * TE)

    # calculate drag coefficients by interpolating measured data
    CdP = pchip_interpolate(MiP, CdiP, VP / aP)
    CdE = pchip_interpolate(MiE, CdiE, VE / aE)

    TE = turbojet_thrust(evader_thrust0, rhoE)
    TP = SRM_thrust(t)

    # saturate normal accelerations at 40g and 10g
    nzP = np.clip(nzP, -40 * g, 40 * g)
    nzE = np.clip(nzE, -10 * g, 10 * g)

    dx[0] = (TP - 0.5 * rhoP * VP**2 * CdP * SP) / mP - g * np.sin(gammaP)
    dx[1] = -(1 / VP) * (nzP + g * np.cos(gammaP))
    dx[2] = VP * np.sin(gammaP)
    dx[3] = VP * np.cos(gammaP)

    dx[4] = (TE - 0.5 * rhoE * VE**2 * CdE * SE) / mE - g * np.sin(gammaE)
    dx[5] = -(1 / VE) * (nzE + g * np.cos(gammaE))
    dx[6] = VE * np.sin(gammaE)
    dx[7] = VE * np.cos(gammaE)
    return dx, aP, rhoP
