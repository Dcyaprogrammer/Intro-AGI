import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from IPython.display import HTML


def inverse_scanner(num_rows, num_cols, x1, y1, theta, meas_phi1, meas_r1, rmax1, alpha1, beta1):
    mc = np.zeros((M, N))
    for i in range(num_rows):
        for j in range(num_cols):
            r = math.sqrt((i-x1)**2+(j-y1)**2)
            phi = (math.atan2(j-y1, i-x1)-theta+math.pi) % (2*math.pi)-math.pi
            k = np.argmin(np.abs(np.subtract(phi, meas_phi1)))
            if (r > min(rmax1, meas_r1[k]+alpha/2.0)) or (abs(phi-meas_phi1[k]) > beta1/2.0):
                mc[i, j] = 0.5
            elif (meas_r1[k] < rmax1) and (abs(r-meas_r1[k]) < alpha1/2.0):
                mc[i, j] = 0.7
            elif r < meas_r1[k]:
                mc[i, j] = 0.3
    return mc


def get_ranges(true_map1, x1, meas_phi1, rmax1):
    (M1, N1) = np.shape(true_map1)
    x2 = x1[0]
    y2 = x1[1]
    theta = x1[2]
    meas_r1 = rmax*np.ones(meas_phi1.shape)
    for i in range(len(meas_phi1)):
        for r in range(1, rmax1+1):
            xi = int(round(x2+r*math.cos(theta+meas_phi1[i])))
            yi = int(round(y2+r*math.cos(theta+meas_phi1[i])))
            if xi <= 0 or xi >= M1-1 or yi <= 0 or yi >= N1-1:
                meas_r1[i] = r
                break
            elif true_map1[int(round(xi)), int(round(yi))] == 1:
                meas_r1[i] = r
                break
    return meas_r1


def map_update(i):
    map_ax.clear()
    map_ax.set_xlim(0, N)
    map_ax.set_ylim(0, M)
    map_ax.imshow(np.subtract(1, true_map), cmap='gray', origin='lower', vmin=0.0, vmax=1.0)
    x_plot = x[1, :i + 1]
    y_plot = x[0, :i + 1]
    map_ax.plot(x_plot, y_plot, "bx-")


def invmod_update(i):
    invmod_ax.clear()
    invmod_ax.set_xlim(0, N)
    invmod_ax.set_ylim(0, M)
    invmod_ax.imshow(invmods[i], cmap='gray', origin='lower', vmin=0.0, vmax=1.0)
    for j in range(len(meas_rs[i])):
        invmod_ax.plot(x[1, i] + meas_rs[i][j] * math.sin(meas_phi[j] + x[2, i]),
                       x[0, i] + meas_rs[i][j] * math.cos(meas_phi[j] + x[2, i]), "ko")
    invmod_ax.plot(x[1, i], x[0, i], 'bx')


def belief_update(i):
    belief_ax.clear()
    belief_ax.set_xlim(0, N)
    belief_ax.set_ylim(0, M)
    belief_ax.imshow(ms[i], cmap='gray', origin='lower', vmin=0.0, vmax=1.0)
    belief_ax.plot(x[1, max(0, i - 10):i], x[0, max(0, i - 10):i], 'bx-')


T_MAX = 150
time_steps = np.arange(T_MAX)

x_0 = [30, 30, 0]

u = np.array([[3, 0, -3, 0], [0, 3, 0, -3]])
u_i = 1

w = np.multiply(0.3, np.ones(len(time_steps)))

M = 50
N = 60
true_map = np.zeros((M, N))
true_map[0:10, 0:10] = 1
true_map[30:35, 40:45] = 1

m = np.multiply(0.5, np.ones((M, N)))
L0 = np.log(np.divide(m, np.subtract(1, m)))
L = L0
meas_phi = np.arange(-0.4, 0.4, 0.05)
rmax = 30
alpha = 1
beta = 0.05
x = np.zeros((3, len(time_steps)))
x[:, 0] = x_0

map_fig = plt.figure()
map_ax = map_fig.add_subplot(111)
map_ax.set_xlim(0, N)
map_ax.set_xlim(0, M)
invmod_fig = plt.figure()
invmod_ax = invmod_fig.add_subplot(111)
invmod_ax.set_xlim(0, N)
invmod_ax.set_ylim(0, M)

belief_fig = plt.figure()
belief_ax = belief_fig.add_subplot(111)
belief_ax.set_xlim(0, N)
belief_ax.set_ylim(0, M)

meas_rs = []
meas_r = get_ranges(true_map, x[:, 0], meas_phi, rmax)
meas_rs.append(meas_r)

invmods = []
invmod = inverse_scanner(M, N, x[0, 0], x[1, 0], x[2, 0], meas_phi, meas_r, rmax, alpha, beta)
invmods.append(invmod)
ms = [m]
L_complete_array = []
Log_conversion = np.log(invmod)
L_complete_array.append(Log_conversion)


for t in range(1, len(time_steps)):
    move = np.add(x[0:2, t-1], u[:, u_i])
    if (move[0] >= M-1) or (move[1] >= N-1) or (move[0] <= 0) or (move[1] <= 0) \
            or true_map[int(round(move[0])), int(round(move[1]))] == 1:
        x[:, t] = x[:, t-1]
        u_i = (u_i+1) % 4
    else:
        x[0:2, t] = move
    x[2, t] = (x[2, t-1]+w[t]) % (2*math.pi)

    meas_r = get_ranges(true_map, x[:, 0], meas_phi, rmax)
    meas_rs.append(meas_r)

    invmod = inverse_scanner(M, N, x[0, 0], x[1, 0], x[2, 0], meas_phi, meas_r, rmax, alpha, beta)
    invmods.append(invmod)

    log_array = np.log(invmod/(1-invmod))
    L_complete_array.append(log_array)
    first_term = L_complete_array[t]
    previous_time = t-1
    if previous_time == 0:
        previous_time = 1
    second_term = L_complete_array[previous_time]
    third_term = L0
    Lt = first_term+second_term+third_term
    m = (np.exp(Lt))/(1+(np.exp(Lt)))
    ms.append(m)

# Output for grading. Do not modify this code!
m_f = ms[-1]
print("{}".format(m_f[40, 10]))
print("{}".format(m_f[30, 40]))
print("{}".format(m_f[35, 40]))
print("{}".format(m_f[0, 50]))
print("{}".format(m_f[10, 5]))
print("{}".format(m_f[20, 15]))
print("{}".format(m_f[25, 50]))


map_anim = anim.FuncAnimation(map_fig, map_update(50), frames=len(x[0, :]), repeat=False)
invmod_anim = anim.FuncAnimation(invmod_fig, invmod_update(50), frames=len(x[0, :]), repeat=False)
belief_anim = anim.FuncAnimation(belief_fig, belief_update(50), frames=len(x[0, :]), repeat=False)

plt.show()
