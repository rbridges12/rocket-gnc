import numpy as np
from pyatmos import nrlmsise00, coesa76
from scipy.interpolate import pchip_interpolate

# constants
g = 9.81
gas_constant = 287.1
specific_heat_ratio_air = 1.4
data0 = coesa76(0)
rho0 = data0.rho
# T0 = data0.T
# P0 = data0.P
# a0 = np.sqrt(gamma * R * T0)

# pursuer and evader surface areas (m^2)
A = 2.3

# pursuer and evader masses (kg)
m = 161.5
L = 3.65
radius = 0.178 / 2

# inertia matrix
Ixx = 0.5 * m * radius ** 2
Iyy = 1 / 12 * m * (3 * radius ** 2 + L ** 2)
Izz = Iyy
I = np.array([[Ixx, 0, 0],
            [0, Iyy, 0],
            [0, 0, Izz]])

# distance between center of pressure and center of mass
d_COP = 0.625

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
    # if t < 10:
    #     return 10000
    # elif t < 30:
    #     return 1800
    # else:
    #     return 0
    return 4000


# x = [p, R, v, omega], u = [tau_x, tau_y, tau_z]
def rocket_dynamics3d(x, u, t):
    p = x[:3]
    # print(x)
    R = x[3:12].reshape(3, 3)
    v = x[12:15]
    omega = x[15:]

    # air density and speed of sound at current altitude
    data = coesa76(p[2] / 1000)
    rho = data.rho
    T = data.T
    # P = data.P
    a = np.sqrt(specific_heat_ratio_air * gas_constant * T)

    # calculate drag coefficient by interpolating measured data
    speed = np.linalg.norm(v)
    Cd = pchip_interpolate(MiP, CdiP, speed / a)
    f_D = 0.5 * rho * speed ** 2 * Cd * A  # drag force
    # f_D_x = 
    f_T = SRM_thrust(t) # thrust force
    # f_D = 0
    # f_T = 0

    omega_hat = np.array([[0, -omega[2], omega[1]],
                            [omega[2], 0, -omega[0]],
                            [-omega[1], omega[0], 0]])

    p_dot = v
    R_dot = R @ omega_hat
    v_dot = f_T / m * R @ np.array([1, 0, 0]) - f_D / (m * speed) * v - np.array([0, 0, g])
    omega_dot = np.linalg.inv(I) @ (u - np.cross(omega, I @ omega)) # TODO: use solve to make inversion faster
    dx = np.concatenate((p_dot, R_dot.flatten(), v_dot, omega_dot))
    return dx
