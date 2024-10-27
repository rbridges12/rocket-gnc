import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

# from dynamics_combined import dynamics_combined
# from dynamics import rocket_dynamics, plane_dynamics
from dynsim_old import dynsim

# plt.rcParams["text.usetex"] = True

g = 9.81
# x = [VP, gammaP, hP, dP, VE, gammaE, hE, dE]
# x[0] = VP
# x[1] = gammaP
# x[2] = hP
# x[3] = dP
# x[4] = VE
# x[5] = gammaE
# x[6] = hE
# x[7] = dE


# def output_old(x):
#     R = np.sqrt((x[6] - x[2]) ** 2 + (x[7] - x[3]) ** 2)
#     Rdot = (
#         (x[6] - x[2]) * (x[4] * np.sin(x[5]) - x[0] * np.sin(x[1]))
#         + (x[7] - x[3]) * (x[4] * np.cos(x[5]) - x[0] * np.cos(x[1]))
#     ) / R
#     betadot = (
#         (x[7] - x[3]) * (x[4] * np.sin(x[5]) - x[0] * np.sin(x[1]))
#         - (x[6] - x[2]) * (x[4] * np.cos(x[5]) - x[0] * np.cos(x[1]))
#     ) / ((x[6] - x[2]) ** 2 + (x[7] - x[3]) ** 2)
#     return np.array([R, Rdot, betadot])

def output(xE, xP):
    VP = xP[0]
    gammaP = xP[1]
    hP = xP[2]
    dP = xP[3]
    VE = xE[0]
    gammaE = xE[1]
    hE = xE[2]
    dE = xE[3]
    
    # R = np.sqrt((x[6] - x[2]) ** 2 + (x[7] - x[3]) ** 2)
    R = np.sqrt((hE - hP) ** 2 + (dE - dP) ** 2)
    # Rdot = (
    #     (x[6] - x[2]) * (x[4] * np.sin(x[5]) - x[0] * np.sin(x[1]))
    #     + (x[7] - x[3]) * (x[4] * np.cos(x[5]) - x[0] * np.cos(x[1]))
    # ) / R
    Rdot = (
        (hE - hP) * (VE * np.sin(gammaE) - VP * np.sin(gammaP))
        + (dE - dP) * (VE * np.cos(gammaE) - VP * np.cos(gammaP))
    ) / R
    # betadot = (
    #     (x[7] - x[3]) * (x[4] * np.sin(x[5]) - x[0] * np.sin(x[1]))
    #     - (x[6] - x[2]) * (x[4] * np.cos(x[5]) - x[0] * np.cos(x[1]))
    # ) / ((x[6] - x[2]) ** 2 + (x[7] - x[3]) ** 2)
    betadot = (
        (dE - dP) * (VE * np.sin(gammaE) - VP * np.sin(gammaP))
        - (hE - hP) * (VE * np.cos(gammaE) - VP * np.cos(gammaP))
    ) / ((hE - hP) ** 2 + (dE - dP) ** 2)
    return np.array([R, Rdot, betadot])


def fuze(xEs, xPs):
    VPs = xPs[0, :]
    hPs = xPs[2, :]
    dPs = xPs[3, :]
    VEs = xEs[0, :]
    hEs = xEs[2, :]
    dEs = xEs[3, :]

    Rs = np.sqrt((hEs - hPs) ** 2 + (dEs - dPs) ** 2)
    d_less_than_range = np.abs(Rs[1:]) < 10
    d_crossed_zero_between_steps = Rs[1:] * Rs[:-1] < 0
    d_increasing = Rs[1:] - Rs[:-1] > 0
    moving_away_from_target = np.any(np.stack([d_crossed_zero_between_steps, d_increasing], axis=1), axis=1)
    fuzed = np.all(np.stack([d_less_than_range, moving_away_from_target], axis=1), axis=1)
    P_hit_ground = hPs[1:] < 0
    E_hit_ground = hEs[1:] < 0
    P_stopped = VPs[1:] < 0
    H_stopped = VEs[1:] < 0
    engagement_over = np.any(np.stack([fuzed, P_hit_ground, E_hit_ground, P_stopped, H_stopped], axis=1), axis=1)
    if np.any(engagement_over, axis=0):
        fuze_index = 0
        for kk in range(len(engagement_over)):
            if engagement_over[kk] == 1:
                fuze_index = kk
                break
        missDistance = Rs[fuze_index]
        detonate = 1
    else:
        missDistance = np.nan
        detonate = 0
    return detonate, missDistance


def plot_intercept(xEs, xPs):
    hPs = xPs[2, :]
    dPs = xPs[3, :]
    hEs = xEs[2, :]
    dEs = xEs[3, :]
    
    fig = plt.figure()
    plt.plot(dPs / 1000, hPs / 1000, "b")
    plt.plot(dPs[0] / 1000, hPs[0] / 1000, "bx")
    plt.plot(dPs[-1] / 1000, hPs[-1] / 1000, "bo")
    plt.plot(dEs / 1000, hEs / 1000, "r")
    plt.plot(dEs[0] / 1000, hEs[0] / 1000, "rx")
    plt.plot(dEs[-1] / 1000, hEs[-1] / 1000, "ro")
    plt.grid()
    ylimits = plt.ylim()
    plt.ylim([0, ylimits[1]])
    plt.xlabel(r"$d$ (km)")
    plt.ylabel(r"$h$ (km)")
    plt.legend(["Pursuer", "Evader"])


def animate_intercept(xEs, xPs):
    VPs = xPs[0, :]
    gammaPs = xPs[1, :]
    hPs = xPs[2, :] / 1000
    dPs = xPs[3, :] / 1000
    VEs = xEs[0, :]
    gammaEs = xEs[1, :]
    hEs = xEs[2, :] / 1000
    dEs = xEs[3, :] / 1000

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
        fig, animate, frames=xEs.shape[1], interval=30, repeat=True
    )
    return ani


def plot_trajectories(xEs, xPs, ts):
    VPs = xPs[0, :]
    gammaPs = xPs[1, :]
    hPs = xPs[2, :]
    dPs = xPs[3, :]
    VEs = xEs[0, :]
    gammaEs = xEs[1, :]
    hEs = xEs[2, :]
    dEs = xEs[3, :]

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


def simulate_engagement(xE0, xP0):
    duration = 70
    dt = 0.1
    nzE = -g

    # loft conditions
    delta_h0 = xE0[2] - xP0[2]
    delta_d0 = np.abs(xE0[3] - xP0[3])
    loft = delta_h0 > -100 and delta_d0 > 2000
    # loft = False

    xEs = np.zeros((4, int(duration / dt) + 1))
    xPs = np.zeros((4, int(duration / dt) + 1))
    ts = np.arange(0, duration + dt, dt)
    nzPs = np.zeros(int(duration / dt) + 1)
    xEs[:, 0] = xE0
    xPs[:, 0] = xP0
    # aPs = []
    # rhoPs = []

    # def odefun_debug(t, x):
    #     dx, aP, rhoP = dynsim(x, t, nzE, nzPs[int(t / dt)])
    #     aPs.append(aP)
    #     rhoPs.append(rhoP)
    #     # print(rhoPs[-1])
    #     # print(len(rhoPs))
    #     return dx
        
    for i in range(int(duration / dt)):
        # simulate dynamics for timestep
        t = i * dt
        next_t = (i + 1) * dt
        # odefun_E = lambda t, x: plane_dynamics(x, t, nzE)
        # odefun_P = lambda t, x: rocket_dynamics(x, t, nzPs[i])
        # odefun_combined = lambda t, x: dynamics_combined(x, t, nzE, nzPs[i])
        # odefun_combined = lambda t, x: dynsim(x, t, nzE, nzPs[int(t / dt)])
        print(f"next iteration, t0 = {t}")
        def odefun_combined(t, x):
            dx = dynsim(x, t, nzE, nzPs[i])
            print(f"i = {i}, nzP = {nzPs[i]}")
            return dx
        # solE = solve_ivp(
        #     odefun_E,
        #     [t, next_t],
        #     xEs[:, i],
        #     method="RK45",
        #     t_eval=np.linspace(t, next_t, 40),
        # )
        # xEsim = solE.y
        # xEs[:, i + 1] = xEsim[:, -1]

        # solP = solve_ivp(
        #     odefun_P,
        #     [t, next_t],
        #     xPs[:, i],
        #     method="RK45",
        #     t_eval=np.linspace(t, next_t, 40),
        # )
        # xPsim = solP.y
        # xPs[:, i + 1] = xPsim[:, -1]

        sol = solve_ivp(
            odefun_combined,
            [t, next_t],
            np.concatenate([xPs[:, i], xEs[:, i]]),
            method="RK45",
            t_eval=np.linspace(t, next_t, 40),
        )
        xPsim = sol.y[:4, :]
        xEsim = sol.y[4:, :]
        xEs[:, i + 1] = xEsim[:, -1]
        xPs[:, i + 1] = xPsim[:, -1]

        y = output(xEs[:, i + 1], xPs[:, i + 1])

        # compute next guidance command
        nzP_PG = lambda k: -k * np.abs(y[1]) * y[2] - g * np.cos(xPs[1, i + 1])
        if loft:
            loft_duration = delta_d0 / 2000.0 - 0.05
            if t < loft_duration:
                nzPs[i + 1] = nzP_PG(3) - 12 * g
            else:
                nzPs[i + 1] = nzP_PG(10)
        else:
            nzPs[i + 1] = nzP_PG(5)
        # nzPs[i + 1] = -g
        detonate, missDistance = fuze(xEsim, xPsim)
        # print(f"t = {t} s, miss distance = {missDistance} m")
        if detonate:
            ts = ts[: i + 1]
            xEs = xEs[:, : i + 1]
            xPs = xPs[:, : i + 1]
            nzPs = nzPs[: i + 1]
            print(f"Detonated at t = {t} s, miss distance = {missDistance} m")
            if loft:
                print("Loft initiated\n")
            break

    # loft_duration = delta_d0 / 2000.0 - 0.05
    # print(f"Loft duration = {loft_duration} s")
    return xEs, xPs, ts, nzPs


# (pursuer x0, evader x0) = ([VP, gammaP, hP, dP], [VE, gammaE, hE, dE])
initial_conditions = [
    (np.array([450, 0, 10000, 0]), np.array([450, 0, 10000, 6500])),
    #(np.array([450, 0, 10000, 0]), np.array([-450, 0, 10000, 6500])),
    (np.array([450, 0, 10000, 0]), np.array([350, 0.1, 10000, 9000])),
    (np.array([450, 0, 10000, 0]), np.array([350, -0.1, 10000, 8000])),
    (np.array([450, 0, 8000, 0]), np.array([350, -0.1, 10000, 6000])),
    (np.array([450, 0, 4000, 0]), np.array([350, 0.2, 6000, 2000])),
    (np.array([450, 0, 10000, 0]), np.array([350, 0.2, 6000, 2000])),
    (np.array([450, 0, 10000, 0]), np.array([400, 0, 7000, 5000])),
    (np.array([450, 0, 10000, 0]), np.array([400, 0.0, 12000, 6000])),
    (np.array([450, 0, 9000, 0]), np.array([400, -0.1, 12000, 6000])),
    (np.array([450, 0, 9000, 0]), np.array([320, -0.4, 12000, 7000])),
]

for xP0, xE0 in initial_conditions:
    xEs, xPs, ts, nzPs = simulate_engagement(xE0, xP0)
    # plot_trajectories(xs, ts)
    # fig, (ax1, ax2) = plt.subplots(2, 1)
    # ax1.plot(np.arange(len(aPs)), np.array(aPs), label="aP")
    # ax2.plot(np.arange(len(rhoPs)), rhoPs, label="rhoP")
    # plt.legend()
    plot_intercept(xEs, xPs)
    plt.show()
    ani = animate_intercept(xEs, xPs)
    plt.show()
