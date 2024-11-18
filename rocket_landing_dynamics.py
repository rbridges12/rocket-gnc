import numpy as np

def position_dynamics(y, u, t):
    r = y[:3]
    v = y[3:6]
    x = y[:6];
    m = y[6]
    Tc = u

    # constants
    Tmax = 24000 # maximum thrust in N
    rho1 = 0.2 * Tmax # minimum thrust
    rho2 = 0.8 * Tmax # maximum thrust
    alpha = 5e-4 # 
    g = np.array([-3.71, 0, 0]) # acceleration due to gravity on Mars
    # omega = np.array([2.53e-5, 0, 6.62e-5]) # angular velocity of Mars
    omega = np.array([0, 0, 0]) # angular velocity of Mars
    m0 = 2000 # total initial mass in kg
    m_dry = 300 # dry mass in kg
    r0 = np.array([2400, 450, -330]) # initial position in m
    v0 = np.array([-10, -40, 10]) # initial velocity in m/s
    q = np.zeros(3) # position of the target in m
    
    # "real world" constraints
    Gamma = np.linalg.norm(Tc)
    Tvec = Tc / Gamma
    Gamma = np.clip(Gamma, rho1, rho2)
    Tc = Gamma * Tvec
    if m <= m_dry:
        Tc = np.zeros(3)
    
    # dynamics
    S = np.array([[0, -omega[2], omega[1]],
                  [omega[2], 0, -omega[0]],
                  [-omega[1], omega[0], 0]])
    A = np.block([[np.zeros((3, 3)), np.eye(3)],
                  [-(S @ S), -2 * S]])
    B = np.block([[np.zeros((3, 3))],
                  [np.eye(3)]])

    xdot = A @ x + B @ (g + Tc / m)
    mdot = -alpha * np.linalg.norm(Tc)
    dy = np.concatenate([xdot, [mdot]])
    return dy