import numpy as np
import pickle

from mcQsigma import QSigma

n_runs = 50
n_eps = 100

ns = [1]
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
sigmas = [0.0, 0.25, 0.5, 0.75, 1.0, -0.95]

epsilon = 0.1
beta = 1.0

gamma = 1.0
max_steps = 10000

R_final = np.array([[[[0.0] * len(sigmas)] * len(alphas)] * len(ns)] * n_runs)

for n, steps in enumerate(ns):
  for a, alpha in enumerate(alphas):
    for s, sigma in enumerate(sigmas):
      for run in range(1, n_runs + 1):
        if sigma >= 0.0:
          agent = QSigma(steps, sigma, epsilon, alpha, 1.0)
        else:
          agent = QSigma(steps, 1.0, epsilon, alpha, -sigma)
        R_ep = 0.0
        for ep in range(1, n_eps + 1):
          R = agent.episode(gamma, max_steps)
          R_ep += (1 / ep) * (R - R_ep)
          print('n', steps, 'a', alpha, 's', sigma, 'run', run, 'ep', ep, 'R', R)
        print('n', steps, 'a', alpha, 's', sigma, 'run', run, 'R', R_ep)
        R_final[run - 1, n, a, s] = R_ep

pickle.dump(R_final, open('Qsig_R_final.p', 'wb'))
# R_final = pickle.load(open('Qsig_R_final.p', 'rb'))
