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


def rocket_dynamics(x, t, nz):
    dx = np.zeros(4)
    V, gamma, h, d = x

    # air density and speed of sound at current altitude
    data = coesa76(h / 1000)
    rho = data.rho
    T = data.T
    P = data.P
    a = np.sqrt(specific_heat_ratio_air * R * T)

    # calculate drag coefficient by interpolating measured data
    Cd = pchip_interpolate(MiP, CdiP, V / a)
    thrust = SRM_thrust(t)

    # saturate normal acceleration at 40g
    nz = np.clip(nz, -40 * g, 40 * g)

    dx[0] = (thrust - 0.5 * rho * V**2 * Cd * SP) / mP - g * np.sin(gamma)
    dx[1] = -(1 / V) * (nz + g * np.cos(gamma))
    dx[2] = V * np.sin(gamma)
    dx[3] = V * np.cos(gamma)

    return dx

def plane_dynamics(x, t, nz):
    dx = np.zeros(4)
    V, gamma, h, d = x

    # air density and speed of sound at current altitude
    data = coesa76(h / 1000)
    rho = data.rho
    T = data.T
    P = data.P
    a = np.sqrt(specific_heat_ratio_air * R * T)

    # calculate drag coefficient by interpolating measured data
    Cd = pchip_interpolate(MiE, CdiE, V / a)
    thrust = turbojet_thrust(evader_thrust0, rho)

    # saturate normal acceleration at 10g
    nz = np.clip(nz, -10 * g, 10 * g)

    dx[0] = (thrust - 0.5 * rho * V**2 * Cd * SE) / mE - g * np.sin(gamma)
    dx[1] = -(1 / V) * (nz + g * np.cos(gamma))
    dx[2] = V * np.sin(gamma)
    dx[3] = V * np.cos(gamma)

    return dx
