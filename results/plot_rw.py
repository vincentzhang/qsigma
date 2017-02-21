import numpy as np
import pickle
import matplotlib.pyplot as plt

rmse = pickle.load(open('R_final_rw.p', 'rb'))
rmsed = pickle.load(open('R_final_rwdecay.p', 'rb'))

ns = [1, 3, 5]
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
sigmas = [0.0, 0.25, 0.5, 0.75, 1.0,]

#[steps, alpha, sigma, run, ep]
eps = np.arange(1, 51)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Episodes')
ax.set_ylabel('RMS Error')

ni = 1
ai = 3
ax.plot(eps, rmse.mean(3)[ni][ai][0][1:51], label='Tree-backup')
ax.plot(eps, rmse.mean(3)[ni][ai][1][1:51], label='Q(σ) (σ=0.25)')
ax.plot(eps, rmse.mean(3)[ni][ai][2][1:51], label='Q(σ) (σ=0.5)')
ax.plot(eps, rmse.mean(3)[ni][ai][3][1:51], label='Q(σ) (σ=0.75)')
ax.plot(eps, rmse.mean(3)[ni][ai][4][1:51], label='Sarsa')
ax.plot(eps, rmsed.mean(3)[ni][ai][0][1:51], label='Decaying σ')
ax.legend()
plt.show()