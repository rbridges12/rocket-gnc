import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

from dynsim import dynsim

# plt.rcParams["text.usetex"] = True

# x = [VP, gammaP, hP, dP, VE, gammaE, hE, dE]

def output(x):
    R = np.sqrt((x[6] - x[2])**2 + (x[7] - x[3])**2)
    Rdot = ((x[6] - x[2])*(x[4]*np.sin(x[5]) - x[0]*np.sin(x[1])) + (x[7] - x[3])*(x[4]*np.cos(x[5]) - x[0]*np.cos(x[1])))/R
    betadot = ((x[7] - x[3])*(x[4]*np.sin(x[5]) - x[0]*np.sin(x[1])) - (x[6] - x[2])*(x[4]*np.cos(x[5]) - x[0]*np.cos(x[1])))/( (x[6] - x[2])**2 + (x[7] - x[3])**2)
    return np.array([R, Rdot, betadot])

# function [detonate , missDistance ] = fuze( xx )
# % returns detonate = 1 and miss distance if fuzing conditions have been met.
# % returns detonate = 0 and miss distance = Nan if fuzing conditions have
# % NOT been met
# % Written by Syed Aseem Ul Islam (aseemisl@umich.edu)
# %
# R = sqrt( ( xx(:,7)-xx(:,3) ).^2 + ( xx(:,8)-xx(:,4) ).^2  );
# temp1a = ( abs(R(2:end,:)) < 10); %Is R<10 in intersample?
# temp1b = ( R(2:end,:).*R(1:end-1,:) < 0) ; % Is R crossing 0 in intersample?
# temp1c = ( R(2:end,:)-R(1:end-1,:) > 0) ; % Is R increasing in intersample?
# temp1bc = any( [temp1b temp1c] , 2 );
# temp1 = all( [temp1a temp1bc] , 2 ); %Is R<10 AND R crossing 0 or Rdot>0 in intersample?
# temp2 = xx(2:end,3) < 0; %Did pursuer hit the ground in intersample?
# temp3 = xx(2:end,7) < 0; %Did evader hit the ground in intersample?  
# temp4 = xx(2:end,1) < 0; %Did pursuer slow to 0 speed in intersample?
# temp5 = xx(2:end,5) < 0; %Did evader slow to 0 speed in intersample?
# temp = any( [temp1 temp2 temp3 temp4 temp5] , 2); %Is any fuzing condition satisfied in intersample?
# if any(temp,1)
#     fuze_index = 0;
#     for kk = 1:length(temp)
#         if temp(kk) == 1
#             fuze_index = kk;
#             break
#         end
#     end
#     missDistance = R(fuze_index,:);
#     detonate = 1;
# else
#     missDistance = NaN;
#     detonate = 0;
# end

# end

def fuze(xsim):
    R = np.sqrt((xsim[6, :] - xsim[2, :])**2 + (xsim[7, :] - xsim[3, :])**2)
    print(np.min(R))
    temp1a = np.abs(R[1:]) < 10
    temp1b = R[1:] * R[:-1] < 0
    temp1c = R[1:] - R[:-1] > 0
    # print(temp1a.shape, temp1b.shape, temp1c.shape)
    # print(np.stack([temp1b, temp1c], axis=1).shape)
    temp1bc = np.any(np.stack([temp1b, temp1c], axis=1), axis=1)
    temp1 = np.all(np.stack([temp1a, temp1bc], axis=1), axis=1)
    print(np.any(temp1a), np.any(temp1bc), np.any(temp1))
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

def plot_trajectories(x, i):
    plt.plot(x[3, :i]/1000, x[2, :i]/1000, 'b')
    plt.plot(x[3, 0]/1000, x[2, 0]/1000, 'bx')
    plt.plot(x[3, i]/1000, x[2, i]/1000, 'bo')
    plt.plot(x[7, :i]/1000, x[6, :i]/1000, 'r')
    plt.plot(x[7, 0]/1000, x[6, 0]/1000, 'rx')
    plt.plot(x[7, i]/1000, x[6, i]/1000, 'ro')
    plt.grid()
    ylimits = plt.ylim()
    plt.ylim([0, ylimits[1]])
    plt.xlabel(r"$d$ (km)")
    plt.ylabel(r"$h$ (km)")
    plt.legend(["Pursuer", "Evader"])
    plt.show()

def simulate_engagement(x0):
    duration = 50
    dt = 0.1
    g = 9.81
    umax = 40 * g
    nzE = -g
    
    # loft conditions
    delta_h0 = x0[6] - x0[2]
    delta_d0 = np.abs(x0[7] - x0[3])
    loft = delta_h0 > -100 and delta_d0 > 2000
    
    xs = np.zeros((8, int(duration/dt) + 1))
    nzPs = np.zeros(int(duration/dt) + 1)
    xs[:, 0] = x0
    
    for i in range(int(duration/dt)):
        # simulate dynamics for timestep
        t = i * dt
        next_t = (i+1) * dt
        odefun = lambda t, x: dynsim(x, t, nzE, nzPs[i])
        sol = solve_ivp(odefun, [t, next_t], xs[:, i], method='RK45', t_eval=np.linspace(t, next_t, 40))
        xsim = sol.y
        xs[:, i+1] = xsim[:, -1]
        
        # compute next guidance command
        y = output(xs[:, i+1])
        nzP_PG = lambda k: -k * np.abs(y[1]) * y[2] - g * np.cos(y[1])
        if loft:
            loft_duration = delta_d0 / 2000 - 0.05;
            if t < loft_duration:
                nzPs[i+1] = nzP_PG(3) - 12 * g
            else:
                nzPs[i+1] = nzP_PG(10)
        else:
            nzPs[i+1] = nzP_PG(5)
        # nzPs[i+1] = nzP_PG(5)
        # print(nzPs[i+1])
        # nzPs[i+1] = -1.1 * g
        # print(sol.t.shape)
        detonate, missDistance = fuze(xsim)
        print(f"t = {t} s, miss distance = {missDistance} m")
        if detonate:
            print(f"Detonated at t = {t} s, miss distance = {missDistance} m")
            if loft:
                print("Loft initiated\n")
            break
        
    return xs, nzPs, i

        
x0_dyn1 = np.array([450, 0, 10000, 0, 450, 0, 10000, 6500])
x0_dyn2 = np.array([450, 0, 10000, 0, 350, 0.1, 10000, 9000])
x0_dyn3 = np.array([450, 0, 10000, 0, 350, -0.1, 10000, 8000])
x0_dyn4 = np.array([450, 0, 8000, 0, 350, -0.1, 10000, 6000])
x0_dyn5 = np.array([450, 0, 4000, 0, 350, 0.2, 6000, 2000])
x0_dyn6 = np.array([450, 0, 10000, 0, 350, 0.2, 6000, 2000])
x0_dyn7 = np.array([450, 0, 10000, 0, 400, 0, 7000, 5000])
x0_dyn8 = np.array([450, 0, 10000, 0, 400, 0.0, 12000, 6000])
x0_dyn9 = np.array([450, 0, 9000, 0, 400, -0.1, 12000, 6000])
x0_dyn10 = np.array([450, 0, 9000, 0, 320, -0.4, 12000, 7000])
x0_dyns = [x0_dyn1, x0_dyn2, x0_dyn3, x0_dyn4, x0_dyn5, x0_dyn6, x0_dyn7, x0_dyn8, x0_dyn9, x0_dyn10]

    
for i in range(10):
    x, nzPs, ii = simulate_engagement(x0_dyns[i])
    plot_trajectories(x, ii)
    # R = np.sqrt((x[3, ii] - x[7, ii])**2 + (x[2, ii] - x[6, ii])**2)
    # plt.title(f'Initial Condition {i}, Miss Distance = {R} m')
    plt.show()
