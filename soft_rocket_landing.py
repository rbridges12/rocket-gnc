import numpy as np
from typing import Tuple
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

from rocket_landing_dynamics import position_dynamics


def animate_trajectory(xs, us, ts):
    R = np.array([[0, 0, 1],
                  [0, -1, 0],
                  [1, 0, 0]])
    rs = R @ xs[:3, :]
    vs = R @ xs[3:6, :]
    ms = xs[6, :]
    Tcs = us

    x_margin = 1
    y_margin = 1
    z_margin = 1
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set(xlim3d=(np.min(rs[0]) - x_margin, np.max(rs[0]) + x_margin), xlabel="X Position (km)")
    ax.set(ylim3d=(np.min(rs[1]) - y_margin, np.max(rs[1]) + y_margin), ylabel="Y Position (km)")
    ax.set(zlim3d=(np.min(rs[2]) - z_margin, np.max(rs[2]) + z_margin), zlabel="Z Position (km)")
    ax.set_title("Intercept")
    ax.set_aspect("equal")
    ax.grid()
    rocket = ax.plot([], [], [], "bo")[0]
    rocket_path = ax.plot([], [], [], "b-")[0]
    thrust_vector = ax.plot([], [], [], "r-")[0]

    def animate(i):
        rocket.set_data_3d(rs[:, i].reshape(3, 1))
        rocket_path.set_data_3d(rs[:, :i])
        thrust_vector_data = np.array([rs[:, i], rs[:, i] + Tcs[:, i]]).T
        thrust_vector.set_data_3d(thrust_vector_data)
        return rocket, rocket_path, thrust_vector

    ani = animation.FuncAnimation(
        fig, animate, frames=ts.shape[0], interval=30, repeat=True
    )

    return ani

def plot_trajectory(xs, us, ts):
    R = np.array([[0, 0, 1],
                  [0, -1, 0],
                  [1, 0, 0]])
    rs = R @ xs[:3, :]
    vs = R @ xs[3:6, :]
    # rs = xs[:3, :]
    # vs = xs[3:6, :]
    ms = xs[6, :]
    Tcs = us

    fig, axs = plt.subplots(3, 3, sharex=True)
    axs[0, 0].plot(ts, rs[0], "r-")
    axs[0, 0].set_ylabel("X Position (m)")
    axs[0, 0].set_xlabel("Time (s)")

    axs[1, 0].plot(ts, rs[1], "r-")
    axs[1, 0].set_ylabel("Y Position (m)")
    axs[1, 0].set_xlabel("Time (s)")
    
    axs[2, 0].plot(ts, rs[2], "r-")
    axs[2, 0].set_ylabel("Z Position (m)")
    axs[2, 0].set_xlabel("Time (s)")
    
    axs[0, 1].plot(ts, vs[0], "r-")
    axs[0, 1].set_ylabel("X Velocity (m/s)")
    axs[0, 1].set_xlabel("Time (s)")
    
    axs[1, 1].plot(ts, vs[1], "r-")
    axs[1, 1].set_ylabel("Y Velocity (m/s)")
    axs[1, 1].set_xlabel("Time (s)")
    
    axs[2, 1].plot(ts, vs[2], "r-")
    axs[2, 1].set_ylabel("Z Velocity (m/s)")
    axs[2, 1].set_xlabel("Time (s)")
    
    axs[0, 2].plot(ts, Tcs[0], "r-")
    axs[0, 2].set_ylabel("X Angular Velocity (rad/s)")
    axs[0, 2].set_xlabel("Time (s)")
    
    axs[1, 2].plot(ts, Tcs[1], "r-")
    axs[1, 2].set_ylabel("Y Angular Velocity (rad/s)")
    axs[1, 2].set_xlabel("Time (s)")
    
    axs[2, 2].plot(ts, Tcs[2], "r-")
    axs[2, 2].set_ylabel("Z Angular Velocity (rad/s)")
    axs[2, 2].set_xlabel("Time (s)")

def simulate():
    Tmax = 24000 # maximum thrust in N
    rho1 = 0.2 * Tmax # minimum thrust
    rho2 = 0.8 * Tmax # maximum thrust
    alpha = 5e-4 # 
    g = np.array([-3.71, 0, 0]) # acceleration due to gravity on Mars
    omega = np.array([2.53e-5, 0, 6.62e-5]) # angular velocity of Mars
    m0 = 2000 # total initial mass in kg
    m_dry = 300 # dry mass in kg
    r0 = np.array([2400, 450, -330]) # initial position in m
    v0 = np.array([-10, -40, 10]) # initial velocity in m/s
    q = np.zeros(3) # position of the target in m

    x0 = np.concatenate([r0, v0, [m0]])

    def thrust_vector(t):
    #     if t < 10:
    #         return np.array([0, 0, 0])
    #     elif t < 20:
    #         return np.array([0, -800, 0])
    #     elif t < 30:
    #         return np.array([0, 0, 0])
    #     elif t < 40:
    #         return np.array([0, -1000, 750])
    #     else:
    #         return np.array([0, -1000, 0])
        return np.array([5000, 0, 0])

    max_duration = 300
    dt = 0.1
    n = 7 # number of states
    l = 3 # number of control inputs

    xs = np.zeros((n, int(max_duration / dt) + 1))
    ts = np.arange(0, max_duration + dt, dt)
    us = np.zeros((l, int(max_duration / dt) + 1))
    xs[:, 0] = x0

    for i in range(int(max_duration / dt)):
        # calculate next control input
        x_current = xs[:, i]
        
        # simulate dynamics for timestep
        t = i * dt
        next_t = (i + 1) * dt
        # u = lambda t: np.append(tau, SRM_thrust(t))
        # u = lambda t: np.append(control_torque(t), SRM_thrust(t))
        # odefun = lambda t, x: rocket_dynamics3d(x, u(t), t)
        # odefun = lambda t, x: rocket_dynamics3d(x, tau, t, SRM_thrust)
        odefun = lambda t, x: position_dynamics(x, thrust_vector(t), t)
        sol = solve_ivp(
            odefun,
            [t, next_t],
            x_current,
            method="RK45",
            t_eval=np.linspace(t, next_t, 40),
        )
        # print(f"solved iteration {i}")
        xsim = sol.y
        xs[:, i + 1] = xsim[:, -1]

        # check if rocket has hit the ground
        altitude = xs[0, i + 1]
        if altitude <= 0:
            ts = ts[:i + 1]
            xs = xs[:, :i + 1]
            us = us[:, :i + 1]
            print(f"Hit the ground at t = {t} s")
            break

    return xs, us, ts


def main():
    xs, us, ts = simulate()
    # xs, us, ts = test_trajectory_optimization()
    ani = animate_trajectory(xs, us, ts)
    plt.show()
    plot_trajectory(xs, us, ts)
    plt.show()

if __name__ == "__main__":
    main()