import numpy as np
import pickle

from randomwalk import RandomWalk
from rwQsigma import QSigma

resume = False

n_runs = 100
n_eps = 100

ns = [1, 3, 5]
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#sigmas = [0.0, 0.25, 0.5, 0.75, 1.0, -0.95]
sigmas = [-0.9]

#  def __init__(self, steps=1, init_sigma=1.0,step_size=0.1, beta=1.0):
gamma = 0.9
max_steps = 10000

# WANT RMS ERROR
mdp = RandomWalk(19, -1)
Q_opt = mdp.Q_equiprobable(gamma); Q_opt[0] = [0, 0]; Q_opt[20] = [0, 0]
Q_opt = np.array(Q_opt)[1:-1]

if resume:
  R_final = pickle.load(open('rwQsig_R_final.p', 'rb'))
else:
  R_final = np.array([[[[[0.0] * n_eps] * n_runs] * len(sigmas)] * len(alphas)] * len(ns)) # R_final[steps, alpha, sigma, run, ep]

for n, steps in enumerate(ns):
  for a, alpha in enumerate(alphas):
    for s, sigma in enumerate(sigmas):
      for run in range(1, n_runs + 1):
        if resume and R_final[n, a, s, run - 1, 0] != 0.0:
          print('*', 'n', steps, 'a', alpha, 's', sigma, 'run', run, 'rmse', R_final[n, a, s, run - 1, :].mean())
          continue
        if sigma >= 0.0:
          agent = QSigma(steps, sigma, alpha, 1.0)
        else:
          agent = QSigma(steps, 1.0, alpha, -sigma)
        for ep in range(1, n_eps + 1):
          R = agent.episode(gamma, max_steps)
          rmse = 0.0
          for state in range(19):
            for action in range(2):
              rmse += (agent._Q[state][action] - Q_opt[state][action]) ** 2
          rmse = (rmse / 38) ** 0.5
          # compute agent rms error here
          R_final[n, a, s, run - 1, ep - 1] = rmse
          print('n', steps, 'a', alpha, 's', sigma, 'run', run, 'ep', ep, 'rmse', rmse)
        pickle.dump(R_final, open('rwQsig_R_final.p', 'wb'))

# R_final = pickle.load(open('Qsig_R_final.p', 'rb')); R_final.mean(4).mean(3)