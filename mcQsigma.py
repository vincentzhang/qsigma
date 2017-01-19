#!/usr/bin/env python
from __future__ import print_function
import numpy as np
from random import randint, random, choice, seed

from tilecoding import tilecoder
import mountaincar

class QSigma:
  def __init__(self, steps=1, init_sigma=1.0, epsilon=0.1, step_size=0.1):
    # Qsigma parameters
    self._n = steps
    self._init_sigma = init_sigma
    # num actions
    self._n_actions = 3
    # tilecoder
    tilings = 20
    dims = [16, 16]
    lims = [(-1.2, 0.5), (-0.07, 0.07)]
    self._Q = [0] * self._n_actions
    for i in range(self._n_actions):
      self._Q[i] = tilecoder(dims, lims, tilings, step_size)
    # eps greedy
    self._eps = epsilon

  def episode(self, discount=1.0, max_steps=1e3):
    """ Run n-step Q(sigma) for one episode """
    self._s = mountaincar.init()
    self._r_sum = 0.0
    self._time = 0 # step counter
    self._T = float('inf')
    self._tau = 0
    action = self.pick_action(self._s)
    self._tr = [(self._s, self._r_sum)] * self._n
    self._delta = [0.0] * self._n
    self._Qt = [self._Q[action][self._s]] * (self._n + 1)
    self._pi = [0.0] * self._n
    self._sigma = [0.0] * self._n
    while (self._tau != (self._T - 1)) and (self._time < max_steps):
      action = self.act(action, discount)
    return self._r_sum

  def act(self, action, discount):
    """ do an action and update Q given the discount factor and step size """
    if self._time < self._T:
        (r, sp) = mountaincar.sample(self._s, action)
        self._r_sum += r
        self._tr[self._time % self._n] = (self._s, action)
        if sp == None: # if terminal
            self._T = self._time + 1
            self._delta[self._time%self._n] = r - self._Qt[self._time%(self._n+1)] # TD error
        else: # commit the next action
            action = self.pick_action(sp) # select arbitrarily and store an action as A_(t+1)
            self._Qt[(self._time + 1)%(self._n+1)] = self._Q[action][sp] # Store Q(St+1;At+1) as Qt+1
            self._sigma[(self._time+1)%self._n] = self._init_sigma
            self._delta[self._time%self._n] = r - self._Qt[self._time%(self._n+1)] + discount*((1-self._sigma[(self._time+1)%self._n]) * self.expected_Q(sp) + self._sigma[(self._time+1)%self._n] * self._Q[action][sp])
            self._pi[(self._time+1)%self._n] = self.get_action_probability(sp, action)
        self._s = sp # update agent state
    self._tau = self._time + 1 - self._n # time whose estimate is being updated
    if self._tau >= 0:
        E = 1.0
        G = self._Qt[self._tau%(self._n+1)]
        for k in range(self._tau, int(min(self._time, self._T-1))+1):
            G += E * self._delta[k%self._n]
            E *= discount * ((1-self._sigma[(k+1)%self._n]) * self._pi[(k+1)%self._n] + self._sigma[(k+1)%self._n])
        s, a = self._tr[self._tau%self._n]
        self._Q[a][s] = G
    self._time += 1
    return action # return the committed next action

  def get_action_probability(self, state, action):
    """ return the action probability at a state of a given policy P[s][a] """
    Q = [0.0] * self._n_actions
    for a in range(self._n_actions):
      Q[a] = self._Q[a][state]
    if Q[action] == max(Q):
        return self._eps/self._n_actions + (1.0 - self._eps)/Q.count(max(Q))
    else:
        return self._eps/self._n_actions

  def pick_action(self, state):
    """ return an action according to a given policy P[s][a] """
    if random() < self._eps:
      return randint(0, self._n_actions - 1)
    else:
      Q = [0.0] * self._n_actions
      for a in range(self._n_actions):
        Q[a] = self._Q[a][state]
      max_Q = max(Q)
      indices = [i for i, x in enumerate(Q) if x == max_Q]
      if len(indices) == 1:
          return indices[0]
      else: # break ties randomly to prevent the case where committing to one action leads to no reward
          return choice(indices)

  def expected_Q(self, state):
    """ get the expected Q under a target policy """
    Q = [0.0] * self._n_actions
    for a in range(self._n_actions):
      Q[a] = self._Q[a][state]
    Q_exp = (1.0 - self._eps) * max(Q)
    for a in range(self._n_actions):
      Q_exp += (self._eps / self._n_actions) * Q[a]
    return Q_exp

def example():
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D

  agent = QSigma(3, 1.0, 0.1, 0.5)
  for episode in range(100):
    R = agent.episode(1.0, 10000)
    print('episode ', episode + 1, 'reward ', R)
    #agent._init_sigma *= 0.9

  print('mapping function...')
  res = 100
  x = np.arange(-1.2, 0.5, (0.5 + 1.2) / res)
  y = np.arange(-0.07, 0.07, (0.07 + 0.07) / res)
  z = np.zeros([len(y), len(x)])
  for i in range(len(x)):
    for j in range(len(y)):
      Q_max = agent._Q[0][x[i], y[j]]
      for a in range(1, agent._n_actions):
        if agent._Q[a][x[i], y[j]] > Q_max:
          Q_max = agent._Q[a][x[i], y[j]]
      z[j, i] = -Q_max

  # plot
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  X, Y = np.meshgrid(x, y)
  surf = ax.plot_surface(X, Y, z, cmap=plt.get_cmap('hot'))
  plt.show()

if __name__ == '__main__':
  example()