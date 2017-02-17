#!/usr/bin/env python
from __future__ import print_function
import numpy as np
from random import randint, random, choice, seed

from multitilecoding import multitilecoder
import rwalk

class QSigma:
  def __init__(self, steps=1, init_sigma=1.0,step_size=0.1, beta=1.0):
    # Qsigma parameters
    self._n = steps
    self._alpha = step_size
    self._sig = init_sigma
    self._beta = beta
    # num actions
    self._n_actions = 2
    # Q
    self._Q = np.array([[0.0] * self._n_actions] * 19) # states 0-18

  def episode(self, discount=1.0, max_steps=1e3):
    """ Run n-step Q(sigma) for one episode """
    self._s = rwalk.init(19)
    self._r_sum = 0.0
    self._time = 0 # step counter
    self._T = float('inf')
    self._tau = 0
    action = self.pick_action(self._s)
    self._tr = [(self._s, self._r_sum)] * self._n
    self._delta = [0.0] * self._n

    self._Qt = [self._Q[self._s][action]] * (self._n + 1)
    self._pi = [0.0] * self._n
    self._sigma = [0.0] * self._n
    while (self._tau != (self._T - 1)) and (self._time < max_steps):
      action = self.act(action, discount)
    self._sig *= self._beta
    return self._r_sum

  def act(self, action, discount):
    """ do an action and update Q given the discount factor and step size """
    if self._time < self._T:
        (r, sp) = rwalk.sample(self._s, action, 19)
        self._r_sum += r
        self._tr[self._time % self._n] = (self._s, action)
        if sp == None: # if terminal
            self._T = self._time + 1
            self._delta[self._time%self._n] = r - self._Qt[self._time%(self._n+1)] # TD error
        else: # commit the next action
            action = self.pick_action(sp) # select arbitrarily and store an action as A_(t+1)
            self._Qt[(self._time + 1)%(self._n+1)] = self._Q[sp][action] # Store Q(St+1;At+1) as Qt+1
            self._sigma[(self._time+1)%self._n] = self._sig
            self._delta[self._time%self._n] = r - self._Qt[self._time%(self._n+1)] + \
              discount*((1-self._sigma[(self._time+1)%self._n]) * self.expected_Q(sp) + self._sigma[(self._time+1)%self._n] * self._Q[sp][action])
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
        self._Q[s][a] += self._alpha * (G - self._Q[s][a])
    self._time += 1
    return action # return the committed next action

  def get_action_probability(self, state, action):
    """ return the action probability at a state of a given policy P[s][a] """
    return 1.0 / self._n_actions

  def pick_action(self, state):
    """ return an action according to a given policy P[s][a] """
    return randint(0, self._n_actions - 1)

  def expected_Q(self, state):
    """ get the expected Q under a target policy """
    return np.mean(self._Q[state])

def example():
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D

  n_runs = 1
  n_eps = 500
  ep_R = 0.0

  for run in range(1, n_runs + 1):
    agent = QSigma(3, 0.0, 0.1, 1.0)
    for episode in range(1, n_eps + 1):
      R = agent.episode(1.0, 10000)
      ep_R += (1 / episode) * (R - ep_R)
      print('episode:', episode, 'reward:', R)
    print('run:', run, 'mean reward:', ep_R)

  print('mapping function...')
  print(agent._Q.round(2))

if __name__ == '__main__':
  example()