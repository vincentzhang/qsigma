import numpy as np
import pickle

from mcQsigma import QSigma

resume = True

n_runs = 100
n_eps = 100

ns = [1, 3, 5]
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
sigmas = [0.0, 0.25, 0.5, 0.75, 1.0, -0.95]

epsilon = 0.1

gamma = 1.0
max_steps = 10000

if resume:
  R_final = pickle.load(open('Qsig_R_final.p', 'rb'))
else:
  R_final = np.array([[[[[0.0] * n_eps] * n_runs] * len(sigmas)] * len(alphas)] * len(ns)) # R_final[steps, alpha, sigma, run, ep]

for n, steps in enumerate(ns):
  for a, alpha in enumerate(alphas):
    for s, sigma in enumerate(sigmas):
      for run in range(1, n_runs + 1):
        if resume and R_final[n, a, s, run - 1, 0] != 0.0:
          print('*', 'n', steps, 'a', alpha, 's', sigma, 'run', run, 'R', R_final[n, a, s, run - 1, :].mean())
          continue
        if sigma >= 0.0:
          agent = QSigma(steps, sigma, epsilon, alpha, 1.0)
        else:
          agent = QSigma(steps, 1.0, epsilon, alpha, -sigma)
        R_ep = 0.0
        for ep in range(1, n_eps + 1):
          R = agent.episode(gamma, max_steps)
          R_final[n, a, s, run - 1, ep - 1] = R
          print('n', steps, 'a', alpha, 's', sigma, 'run', run, 'ep', ep, 'R', R)
        pickle.dump(R_final, open('Qsig_R_final.p', 'wb'))

# R_final = pickle.load(open('Qsig_R_final.p', 'rb')); R_final.mean(4).mean(3)