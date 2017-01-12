import sys; sys.path.append('../')
from randomwalk import RandomWalk
from Qsigma import QSigma
import pickle

import matplotlib.pyplot as plt

def run_agent(n=1, alpha=0.5, sigma=1.0, episodes=10):
    """ Run an agent for specified n-step Qsigma method """
    mdp = RandomWalk(19, -1)
    s = mdp.init()
    num_runs = 250
    num_episodes = episodes
    discount = 1.0
    step_size = alpha
    steps = n

    # Arrays for sum of rewards for each episodes
    Q_opt = mdp.Q_equiprobable(1.0)
    rms_err = 0.0

    # create n-step Qsigma agent
    agent = QSigma(mdp, sigma, s, steps)
    agent.set_policy_equiprobable()

    for run in range(num_runs):
        sqerr = 0.0
        for i in range(num_episodes):
            agent.episode(discount, step_size)
            agent.init()
        count = 0
        for s in range(mdp.num_states()):
          for a in range(mdp.num_actions(s)):
            count += 1
            sqerr += (1 / count) * ((agent.Q[s][a] - Q_opt[s][a]) ** 2 - sqerr)
        rms_err += sqerr ** 0.5
        # Reset Q after a run
        agent.reset_Q()

    rms_err /= num_runs

    return rms_err

def decay_agent(n=1, alpha=0.5, episodes=100, ep_start=30, decay=0.7):
    """ Run an agent for specified n-step Qsigma method with sigma decay"""
    mdp = RandomWalk(19, -1)
    s = mdp.init()
    num_runs = 250
    num_episodes = episodes
    discount = 1.0
    step_size = alpha
    steps = n

    # Arrays for sum of rewards for each episodes
    Q_opt = mdp.Q_equiprobable(1.0)
    rms_err = 0.0

    # create n-step Qsigma agent
    agent = QSigma(mdp, 1.0, s, steps)
    agent.set_policy_equiprobable()

    for run in range(num_runs):
        sqerr = 0.0
        agent._Psigma = 1.0
        for i in range(num_episodes):
            if i > ep_start:
              agent._Psigma *= decay
            agent.episode(discount, step_size)
            agent.init()
        count = 0
        for s in range(mdp.num_states()):
          for a in range(mdp.num_actions(s)):
            count += 1
            sqerr += (1 / count) * ((agent.Q[s][a] - Q_opt[s][a]) ** 2 - sqerr)
        rms_err += sqerr ** 0.5
        # Reset Q after a run
        agent.reset_Q()

    rms_err /= num_runs

    return rms_err

def plot_rms_err():
    """ Plot the sum of rewards of each episode, averaged over a few runs """
    #alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    episodes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    sigma = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    decays = [0.9, 0.8, 0.7, 0.6, 0.5]

    '''
    ep_rms_err = []
    for e in range(len(episodes)):
      sig_rms_err = []
      for s in range(len(sigma)):
        rms_err = run_agent(3, 0.5, sigma[s], episodes[e])
        sig_rms_err.append(rms_err)
        print("sig={} episodes={} rms_err={}".format(sigma[s], episodes[e], rms_err))
      ep_rms_err.append(sig_rms_err)
    pickle.dump(ep_rms_err, open('Qsig_test.p', 'wb'))
    '''

    ep_rms_err = []
    for e in range(len(episodes)):
      sig_rms_err = []
      for d in range(len(decays)):
        rms_err = decay_agent(3, 0.5, episodes[e], 10, decays[d])
        sig_rms_err.append(rms_err)
        print("dec={} episodes={} rms_err={}".format(decays[d], episodes[e], rms_err))
      ep_rms_err.append(sig_rms_err)
    pickle.dump(ep_rms_err, open('Qsig_decays.p', 'wb'))

    plt.figure(1)
    for e in range(len(episodes)):
        plt.plot(sigma, ep_rms_err[e])
    plt.ylabel("RMS Error")
    plt.xlabel("Sigma")

    plt.show()

def plot_rms_err2(load=False):
    """ Plot the sum of rewards of each episode, averaged over a few runs """
    #alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    episodes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    sigma = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    decays = [0.9] #[0.9, 0.8, 0.7, 0.6, 0.5]

    if not load:
        ep_rms_err = []
        for e in range(len(episodes)):
          sig_rms_err = []
          for s in range(len(sigma)):
            rms_err = run_agent(512, 1.0, sigma[s], episodes[e])
            sig_rms_err.append(rms_err)
            print("sig={} episodes={} rms_err={}".format(sigma[s], episodes[e], rms_err))
          ep_rms_err.append(sig_rms_err)
        pickle.dump(ep_rms_err, open('Qsig_n_512_alpha_1.p', 'wb'))
    else:
        ep_rms_err = pickle.load(open('Qsig_n_512_alpha_1.p', 'rb'))
        # plot the RMS error versus episodes
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        colors = ["r", "#81fca9", "b", "#777777", "#ffcafb", "#6afefe", "#b96cb1", "#ff00ff", "#ffab6a", "#dbcd95",
                  "#44140a"]
        for e in range(len(sigma)):
            print("plot for sigma = {}".format(sigma[e]))
            plt.plot(episodes, [x[e] for x in ep_rms_err], label=r"$P(\sigma=1)$ = {}".format(sigma[e]), color=colors[e])
        plt.ylabel("RMS Error")
        plt.legend(loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        plt.xlabel("Episodes")
        #plt.savefig('Qsig_n_512_alpha_1.pdf', bbox_inches='tight', format='pdf', dpi=600)
        plt.show()


if __name__ == '__main__':
    plot_rms_err(True)
