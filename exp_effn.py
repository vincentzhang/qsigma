import numpy as np
import pickle

from mcQsigma import QSigma

resume = False

n_runs = 100
n_eps = 100

epsilon = 0.1

gamma = 1.0
max_steps = 10000


alpha = 0.1;
agents = [(7, 0.5380, epsilon, alpha, 1.0), (9, 0.1275, epsilon, alpha, 1.0)];

if resume:
  R_final = pickle.load(open('Qsig_R_final.p', 'rb'))
else:
  R_final = np.array([[[0.0] * n_eps] * n_runs] * len(agents)) # R_final[agents, run, ep]

for a, agent_spec in enumerate(agents):
  for run in range(1, n_runs + 1):
    if resume and R_final[a, run - 1, 0] != 0.0:
      print('*', 'agent', a, 'run', run, 'R', R_final[a, run - 1, :].mean())
      continue
    agent = QSigma(agent_spec[0], agent_spec[1], agent_spec[2], agent_spec[3], agent_spec[4])
    R_ep = 0.0
    for ep in range(1, n_eps + 1):
      R = agent.episode(gamma, max_steps)
      R_final[a, run - 1, ep - 1] = R
      print('agent', a, 'run', run, 'ep', ep, 'R', R)
    pickle.dump(R_final, open('Qsig_R_final.p', 'wb'))

# R_final = pickle.load(open('Qsig_R_final.p', 'rb')); R_final.mean(4).mean(3)