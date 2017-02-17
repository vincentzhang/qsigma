import numpy as np
import pickle
import matplotlib.pyplot as plt

rmse = pickle.load(open('R_final_rw.p', 'rb'))
rmsed = pickle.load(open('R_final_rwdecay.p', 'rb'))

ns = [1, 3, 5]
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
sigmas = [0.0, 0.25, 0.5, 0.75, 1.0,]

#[steps, alpha, sigma, run, ep]
eps = np.arange(1, 101)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Episodes')
ax.set_ylabel('RMS Error')

ni = 2
ai = 3
for i in range(len(sigmas) - 1):
  ax.plot(eps, rmse.mean(3)[ni][ai][i], label='σ = ' + str(sigmas[i]))
ax.plot(eps, rmsed.mean(3)[ni][ai][0], label='β = 0.9')
ax.legend()
plt.show()
