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


# x = [p, R, v, omega], u = [tau_x, tau_y, tau_z, f_T]
def rocket_dynamics3d(x, u, t):
    p = x[:3]
    # print(x)
    R = x[3:12].reshape(3, 3)
    v = x[12:15]
    omega = x[15:]
    tau_control = u[:3]
    f_T = u[3]

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
    f_D_vector = -f_D * v / speed
    f_T_vector = f_T * R[:, 0]
    f_g_vector = np.array([0, 0, -m * g])  # gravity force
    f_net_vector = f_T_vector + f_D_vector + f_g_vector # net force acting on COM in world frame

    p = np.array([d_COP, 0, 0])  # vector from COP to COM in body frame
    tau_D = np.cross(-p, R.T @ f_D_vector)  # torque due to drag in body frame
    tau_net = tau_control + tau_D  # net torque acting on COM in body frame
    # TODO: track velocity and body frame instead of world frame, so we can use nicer adjoint formulation for wrenches

    omega_hat = np.array([[0, -omega[2], omega[1]],
                            [omega[2], 0, -omega[0]],
                            [-omega[1], omega[0], 0]])

    p_dot = v
    R_dot = R @ omega_hat
    v_dot = f_net_vector / m
    omega_dot = np.linalg.inv(I) @ (tau_net - np.cross(omega, I @ omega)) # TODO: use solve to make inversion faster
    dx = np.concatenate((p_dot, R_dot.flatten(), v_dot, omega_dot))
    return dx
