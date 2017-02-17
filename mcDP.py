#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import mountaincar
from MDPy import MDP

# set precision and value limits
pos_precision = 2
vel_precision = 4
pos_limits = [-1.2, 0.49]
vel_limits = [-0.07, 0.07]

# build state space
pos_delta = 10 ** -pos_precision
vel_delta = 10 ** -vel_precision
pos_range = np.round(np.linspace(pos_limits[0], pos_limits[1], np.round((pos_limits[1] - pos_limits[0]) / pos_delta + 1)), pos_precision)
vel_range = np.round(np.linspace(vel_limits[0], vel_limits[1], np.round((vel_limits[1] - vel_limits[0]) / vel_delta + 1)), vel_precision)

# mdp size
n_states = len(pos_range) * len(vel_range)
n_actions = 3

# state indexing
def state_id(state):
  if state == None:
    return n_states
  pos, vel = state
  if np.round(pos, pos_precision) > pos_limits[1]:
    return n_states
  pos = np.where(pos_range == np.round(pos, pos_precision))[0][0]
  vel = np.where(vel_range == np.round(vel, vel_precision))[0][0]
  return pos * len(vel_range) + vel

# add states and actions to mdp
mcar = MDP()
mcar.add_states(n_states + 1) # add terminal state at end
for i in range(n_states):
  mcar.add_actions(i, n_actions)

# wire up mdp
print('building mdp...')
for p in pos_range:
  for v in vel_range:
    s = (p, v)
    for a in range(3):
      R, sp = mountaincar.sample(s, a)
      mcar.add_transition(state_id(s), a, (state_id(sp), R, 1.0))

# compute values
print('solving mdp...')
V = mcar.value_iteration(1.0)

# map values for plotting
print('mapping function...')
x = pos_range
y = vel_range
plot_V = np.zeros([len(y), len(x)])
for i in range(len(x)):
  for j in range(len(y)):
    plot_V[j, i] = -V[state_id((x[i], y[j]))]

# plot
fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(x, y)
surf = ax.plot_surface(X, Y, plot_V, cmap=plt.get_cmap('hot'))
plt.show()