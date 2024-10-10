import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

from dynsim import dynsim

# plt.rcParams["text.usetex"] = True

g = 9.81
# x = [VP, gammaP, hP, dP, VE, gammaE, hE, dE]


def output(x):
    R = np.sqrt((x[6] - x[2]) ** 2 + (x[7] - x[3]) ** 2)
    Rdot = (
        (x[6] - x[2]) * (x[4] * np.sin(x[5]) - x[0] * np.sin(x[1]))
        + (x[7] - x[3]) * (x[4] * np.cos(x[5]) - x[0] * np.cos(x[1]))
    ) / R
    betadot = (
        (x[7] - x[3]) * (x[4] * np.sin(x[5]) - x[0] * np.sin(x[1]))
        - (x[6] - x[2]) * (x[4] * np.cos(x[5]) - x[0] * np.cos(x[1]))
    ) / ((x[6] - x[2]) ** 2 + (x[7] - x[3]) ** 2)
    return np.array([R, Rdot, betadot])


def fuze(xsim):
    R = np.sqrt((xsim[6, :] - xsim[2, :]) ** 2 + (xsim[7, :] - xsim[3, :]) ** 2)
    # print(np.min(R))
    temp1a = np.abs(R[1:]) < 10
    temp1b = R[1:] * R[:-1] < 0
    temp1c = R[1:] - R[:-1] > 0
    # print(temp1a.shape, temp1b.shape, temp1c.shape)
    # print(np.stack([temp1b, temp1c], axis=1).shape)
    temp1bc = np.any(np.stack([temp1b, temp1c], axis=1), axis=1)
    temp1 = np.all(np.stack([temp1a, temp1bc], axis=1), axis=1)
    # print(np.any(temp1a), np.any(temp1bc), np.any(temp1))
    temp2 = xsim[2, 1:] < 0
    temp3 = xsim[6, 1:] < 0
    temp4 = xsim[0, 1:] < 0
    temp5 = xsim[4, 1:] < 0
    # temp = np.logical_or(temp1, np.logical_or(temp2, np.logical_or(temp3, np.logical_or(temp4, temp5))))
    temp = np.any(np.stack([temp1, temp2, temp3, temp4, temp5], axis=1), axis=1)
    if np.any(temp, axis=0):
        fuze_index = 0
        for kk in range(len(temp)):
            if temp[kk] == 1:
                fuze_index = kk
                break
        missDistance = R[fuze_index]
        detonate = 1
    else:
        missDistance = np.nan
        detonate = 0
    return detonate, missDistance


def plot_intercept(x):
    fig = plt.figure()
    plt.plot(x[3, :] / 1000, x[2, :] / 1000, "b")
    plt.plot(x[3, 0] / 1000, x[2, 0] / 1000, "bx")
    plt.plot(x[3, -1] / 1000, x[2, -1] / 1000, "bo")
    plt.plot(x[7, :] / 1000, x[6, :] / 1000, "r")
    plt.plot(x[7, 0] / 1000, x[6, 0] / 1000, "rx")
    plt.plot(x[7, -1] / 1000, x[6, -1] / 1000, "ro")
    plt.grid()
    ylimits = plt.ylim()
    plt.ylim([0, ylimits[1]])
    plt.xlabel(r"$d$ (km)")
    plt.ylabel(r"$h$ (km)")
    plt.legend(["Pursuer", "Evader"])


def animate_intercept(xs):
    VPs = xs[0, :]
    gammaPs = xs[1, :]
    hPs = xs[2, :] / 1000
    dPs = xs[3, :] / 1000
    VEs = xs[4, :]
    gammaEs = xs[5, :]
    hEs = xs[6, :] / 1000
    dEs = xs[7, :] / 1000

    x_margin = 1
    y_margin = 1
    fig, ax = plt.subplots()
    ax.set_xlim(0, np.max([np.max(dPs), np.max(dEs)]) + x_margin)
    ax.set_ylim(0, np.max([np.max(hPs), np.max(hEs)]) + y_margin)
    ax.set_xlabel(r"$d$ (km)")
    ax.set_ylabel(r"$h$ (km)")
    ax.set_title("Intercept")
    ax.set_aspect("equal")
    ax.grid()
    (pursuer,) = ax.plot([], [], "bo")
    (evader,) = ax.plot([], [], "ro")
    (pursuer_path,) = ax.plot([], [], "b-")
    (evader_path,) = ax.plot([], [], "r-")

    def animate(i):
        pursuer.set_data([dPs[i]], [hPs[i]])
        evader.set_data([dEs[i]], [hEs[i]])
        pursuer_path.set_data(dPs[:i], hPs[:i])
        evader_path.set_data(dEs[:i], hEs[:i])
        return pursuer, evader, pursuer_path, evader_path

    ani = animation.FuncAnimation(
        fig, animate, frames=xs.shape[1], interval=30, repeat=True
    )
    return ani


def plot_trajectories(xs, ts):
    VPs = xs[0, :]
    gammaPs = xs[1, :]
    hPs = xs[2, :]
    dPs = xs[3, :]
    VEs = xs[4, :]
    gammaEs = xs[5, :]
    hEs = xs[6, :]
    dEs = xs[7, :]

    fig, axs = plt.subplots(2, 2, sharex=True)
    axs[0, 0].plot(ts, dPs, "r-", label="Pursuer")
    axs[0, 0].plot(ts, dEs, "b-", label="Evader")
    axs[0, 0].set_ylabel("Downrange (m)")
    axs[0, 0].set_xlabel("Time (s)")
    axs[0, 0].legend()

    axs[0, 1].plot(ts, hPs, "r-", label="Pursuer")
    axs[0, 1].plot(ts, hEs, "b-", label="Evader")
    axs[0, 1].set_ylabel("Altitude (m)")
    axs[0, 1].set_xlabel("Time (s)")

    axs[1, 0].plot(ts, VPs, "r-", label="Pursuer")
    axs[1, 0].plot(ts, VEs, "b-", label="Evader")
    axs[1, 0].set_ylabel("Speed (m/s)")
    axs[1, 0].set_xlabel("Time (s)")

    axs[1, 1].plot(ts, gammaPs, "r-", label="Pursuer")
    axs[1, 1].plot(ts, gammaEs, "b-", label="Evader")
    axs[1, 1].set_ylabel("Flight Path Angle (rad)")
    axs[1, 1].set_xlabel("Time (s)")


def simulate_engagement(x0):
    duration = 70
    dt = 0.1
    nzE = -g

    # loft conditions
    delta_h0 = x0[6] - x0[2]
    delta_d0 = np.abs(x0[7] - x0[3])
    loft = delta_h0 > -100 and delta_d0 > 2000
    # loft = False

    xs = np.zeros((8, int(duration / dt) + 1))
    ts = np.arange(0, duration + dt, dt)
    nzPs = np.zeros(int(duration / dt) + 1)
    xs[:, 0] = x0
    aPs = []
    rhoPs = []

    def odefun_debug(t, x):
        dx, aP, rhoP = dynsim(x, t, nzE, nzPs[int(t / dt)])
        aPs.append(aP)
        rhoPs.append(rhoP)
        # print(rhoPs[-1])
        # print(len(rhoPs))
        return dx
        
    for i in range(int(duration / dt)):
        # simulate dynamics for timestep
        t = i * dt
        next_t = (i + 1) * dt
        odefun = lambda t, x: dynsim(x, t, nzE, nzPs[i])
        sol = solve_ivp(
            odefun_debug,
            [t, next_t],
            xs[:, i],
            method="RK45",
            t_eval=np.linspace(t, next_t, 40),
        )
        xsim = sol.y
        xs[:, i + 1] = xsim[:, -1]
        y = output(xs[:, i + 1])

        # compute next guidance command
        nzP_PG = lambda k: -k * np.abs(y[1]) * y[2] - g * np.cos(xs[1, i + 1])
        if loft:
            loft_duration = delta_d0 / 2000.0 - 0.05
            if t < loft_duration:
                nzPs[i + 1] = nzP_PG(3) - 12 * g
            else:
                nzPs[i + 1] = nzP_PG(10)
        else:
            nzPs[i + 1] = nzP_PG(5)
            # nzPs[i + 1] = -g
        detonate, missDistance = fuze(xsim)
        # print(f"t = {t} s, miss distance = {missDistance} m")
        if detonate:
            ts = ts[: i + 1]
            xs = xs[:, : i + 1]
            nzPs = nzPs[: i + 1]
            print(f"Detonated at t = {t} s, miss distance = {missDistance} m")
            if loft:
                print("Loft initiated\n")
            break

    # loft_duration = delta_d0 / 2000.0 - 0.05
    # print(f"Loft duration = {loft_duration} s")
    return xs, ts, nzPs, aPs, rhoPs


# x = [VP, gammaP, hP, dP, VE, gammaE, hE, dE]
x0s = [
    np.array([450, 0, 10000, 0, 450, 0, 10000, 6500]),
    # np.array([450, 0, 10000, 0, -450, 0, 10000, 6500]),
    np.array([450, 0, 10000, 0, 350, 0.1, 10000, 9000]),
    np.array([450, 0, 10000, 0, 350, -0.1, 10000, 8000]),
    np.array([450, 0, 8000, 0, 350, -0.1, 10000, 6000]),
    np.array([450, 0, 4000, 0, 350, 0.2, 6000, 2000]),
    np.array([450, 0, 10000, 0, 350, 0.2, 6000, 2000]),
    np.array([450, 0, 10000, 0, 400, 0, 7000, 5000]),
    np.array([450, 0, 10000, 0, 400, 0.0, 12000, 6000]),
    np.array([450, 0, 9000, 0, 400, -0.1, 12000, 6000]),
    np.array([450, 0, 9000, 0, 320, -0.4, 12000, 7000]),
]

for x0 in x0s:
    xs, ts, nzPs, aPs, rhoPs = simulate_engagement(x0)
    # plot_trajectories(xs, ts)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(np.arange(len(aPs)), np.array(aPs), label="aP")
    ax2.plot(np.arange(len(rhoPs)), rhoPs, label="rhoP")
    plt.legend()
    plt.show()
    plot_intercept(xs)
    plt.show()
    ani = animate_intercept(xs)
    plt.show()
