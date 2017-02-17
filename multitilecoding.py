#!/usr/bin/env python
from __future__ import print_function
from tilecoding import tilecoder

'''
creates a list of identical tile coders - convenient for Q[s, a] with discrete, independent actions
usage:
Q = multitilecoder(n_actions, dims, limits, tilings, step_size)
Q[s, a] = r + gamma * Q[sp, ap]
print(Q[s, a])
'''

class multitilecoder:
  def __init__(self, n_tilecoders, dims, limits, tilings, step_size=0.1, offset_vec=None):
    self._n = n_tilecoders
    self._tcs = [tilecoder(dims, limits, tilings, step_size, offset_vec) for i in range(self._n)]

  def set_step_size(self, index, step_size):
    self._tcs[index].set_step_size(step_size)

  def set_step_sizes(self, step_size):
    for tc in self._tcs:
      tc.set_step_size(step_size)

  # get value of tilecoder x[1] at location x[0]
  def __getitem__(self, x):
    return self._tcs[x[1]][x[0]]

  # set value of tilecoder x[1] at location x[0] to val
  def __setitem__(self, x, val):
    self._tcs[x[1]][x[0]] = val