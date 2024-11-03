import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

from dynamics3d import rocket_dynamics3d

# plt.rcParams["text.usetex"] = True

g = 9.81

# def plot_intercept(xEs, xPs):
#     hPs = xPs[2, :]
#     dPs = xPs[3, :]
#     hEs = xEs[2, :]
#     dEs = xEs[3, :]
    
#     fig = plt.figure()
#     plt.plot(dPs / 1000, hPs / 1000, "b")
#     plt.plot(dPs[0] / 1000, hPs[0] / 1000, "bx")
#     plt.plot(dPs[-1] / 1000, hPs[-1] / 1000, "bo")
#     plt.plot(dEs / 1000, hEs / 1000, "r")
#     plt.plot(dEs[0] / 1000, hEs[0] / 1000, "rx")
#     plt.plot(dEs[-1] / 1000, hEs[-1] / 1000, "ro")
#     plt.grid()
#     ylimits = plt.ylim()
#     plt.ylim([0, ylimits[1]])
#     plt.xlabel(r"$d$ (km)")
#     plt.ylabel(r"$h$ (km)")
#     plt.legend(["Pursuer", "Evader"])


def animate_trajectory(xs, ts):
    ps = xs[:3, :] / 1000
    Rs = xs[3:12, :].reshape(3, 3, -1)
    vs = xs[12:15, :]
    omegas = xs[15:, :]

    x_margin = 1
    y_margin = 1
    z_margin = 1
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    # ax.set_xlim(0, np.max([np.max(dPs), np.max(dEs)]) + x_margin)
    # ax.set_ylim(0, np.max([np.max(hPs), np.max(hEs)]) + y_margin)
    # ax.set_xlabel(r"$d$ (km)")
    # ax.set_ylabel(r"$h$ (km)")
    ax.set(xlim3d=(np.min(ps[0]) - x_margin, np.max(ps[0]) + x_margin), xlabel="X Position (km)")
    ax.set(ylim3d=(np.min(ps[1]) - y_margin, np.max(ps[1]) + y_margin), ylabel="Y Position (km)")
    ax.set(zlim3d=(np.min(ps[2]) - z_margin, np.max(ps[2]) + z_margin), zlabel="Z Position (km)")
    ax.set_title("Intercept")
    ax.set_aspect("equal")
    # ax.grid()
    rocket = ax.plot([], [], [], "bo")[0]
    # (evader,) = ax.plot([], [], "ro")
    rocket_path = ax.plot([], [], [], "b-")[0]
    # (evader_path,) = ax.plot([], [], "r-")

    def animate(i):
        rocket.set_data_3d(ps[:, i].reshape(3, 1))
        # evader.set_data([dEs[i]], [hEs[i]])
        rocket_path.set_data_3d(ps[:, :i])
        # evader_path.set_data(dEs[:i], hEs[:i])
        return rocket, rocket_path

    ani = animation.FuncAnimation(
        fig, animate, frames=ts.shape[0], interval=30, repeat=True
    )

    return ani


def plot_trajectory(xs, ts):
    ps = xs[:3, :]
    Rs = xs[3:12, :].reshape(3, 3, -1)
    vs = xs[12:15, :]
    omegas = xs[15:, :]

    fig, axs = plt.subplots(3, 3, sharex=True)
    axs[0, 0].plot(ts, ps[0], "r-")
    axs[0, 0].set_ylabel("X Position (m)")
    axs[0, 0].set_xlabel("Time (s)")

    axs[1, 0].plot(ts, ps[1], "r-")
    axs[1, 0].set_ylabel("Y Position (m)")
    axs[1, 0].set_xlabel("Time (s)")
    
    axs[2, 0].plot(ts, ps[2], "r-")
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
    
    axs[0, 2].plot(ts, omegas[0], "r-")
    axs[0, 2].set_ylabel("X Angular Velocity (rad/s)")
    axs[0, 2].set_xlabel("Time (s)")
    
    axs[1, 2].plot(ts, omegas[1], "r-")
    axs[1, 2].set_ylabel("Y Angular Velocity (rad/s)")
    axs[1, 2].set_xlabel("Time (s)")
    
    axs[2, 2].plot(ts, omegas[2], "r-")
    axs[2, 2].set_ylabel("Z Angular Velocity (rad/s)")
    axs[2, 2].set_xlabel("Time (s)")


def simulate():
    p0 = np.array([0, 0, 10000])
    R0 = np.eye(3)
    v0 = np.array([450, 0, 0])
    omega0 = np.array([0, 0, 0])
    x0 = np.concatenate([p0, R0.flatten(), v0, omega0])
    # print(x0)

    def control_torque(t):
        if t < 10:
            return np.array([0, 0, 0])
        elif t < 15:
            return np.array([0, 100, 0])
        elif t < 30:
            return np.array([0, 0, 0])
        elif t < 35:
            return np.array([0, 0, 50])
        else:
            return np.array([0, 0, 0])
    
    duration = 70
    dt = 0.1
    n = 18 # number of states
    m = 3 # number of control inputs

    xs = np.zeros((n, int(duration / dt) + 1))
    ts = np.arange(0, duration + dt, dt)
    us = np.zeros((m, int(duration / dt) + 1))
    xs[:, 0] = x0
        
    for i in range(int(duration / dt)):
        # simulate dynamics for timestep
        t = i * dt
        next_t = (i + 1) * dt
        odefun = lambda t, x: rocket_dynamics3d(x, control_torque(t), t)
        sol = solve_ivp(
            odefun,
            [t, next_t],
            xs[:, i],
            method="RK45",
            t_eval=np.linspace(t, next_t, 40),
        )
        xsim = sol.y
        xs[:, i + 1] = xsim[:, -1]

        # check if missile has hit the ground
        if xs[2, i + 1] <= 0:
            ts = ts[:i + 1]
            xs = xs[:, :i + 1]
            us = us[:, :i + 1]
            print(f"Hit the ground at t = {t} s")
            break

    return xs, us, ts


def main():
    xs, us, ts = simulate()
    ani = animate_trajectory(xs, ts)
    plt.show()
    plot_trajectory(xs, ts)
    plt.show()

if __name__ == "__main__":
    main()