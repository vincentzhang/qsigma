import numpy as np
import pickle
import matplotlib.pyplot as plt

R = pickle.load(open('R_final_mcliff.p', 'rb'))

ns = [1, 3, 5]
alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
sigmas = [0.0, 0.25, 0.5, 0.75, 1.0, -0.95]

#[steps, alpha, sigma, run, ep]
eps = np.arange(1, 101)

ni = 2
ai = 4
for i in range(len(sigmas) - 1):
  plt.plot(eps, R.mean(3)[ni][ai][i], label=str(sigmas[i]))

plt.legend()
plt.show()