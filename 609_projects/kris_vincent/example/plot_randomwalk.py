# Add the parent folder path to the sys.path list
import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np
from random import seed
import pickle

from randomwalk import RandomWalk
from SARSA import Sarsa
from expSARSA import ExpSARSA
from treebackup import TreeBackup
from Qsigma import QSigma


def run_agent_RMS_Q(num_runs, num_episodes, discount, step_size, step=1):
    """ Run SARSA agent for num_episodes to get the Q values """
    mdp = RandomWalk(19)
    s = mdp.init()

    # ground truth for Q
    gt_Q = np.asarray(mdp.Q_equiprobable(discount)[1:-1])
    gt_Q_left = gt_Q[:, 0]
    gt_Q_right = gt_Q[:, 1]

    v = np.asarray([0.5] * mdp.num_states())
    v[0], v[-1] = 0.0, 0.0
    init_Q_left = np.asarray(mdp.value_to_Q(v, discount)[1:-1])[:, 0]
    init_Q_right = np.asarray(mdp.value_to_Q(v, discount)[1:-1])[:, 1]

    # Arrays for RMS error over all states
    rms_err_left = np.asarray([0.0] * (num_episodes + 1))  # Q[left]
    rms_err_right = np.asarray([0.0] * (num_episodes + 1))  # Q[right]
    sum_rms_err_left = np.asarray([0.0] * (num_episodes + 1))
    sum_rms_err_right = np.asarray([0.0] * (num_episodes + 1))
    rms_err_left[0] = np.sqrt(np.mean(np.square(init_Q_left - gt_Q_left)))
    rms_err_right[0] = np.sqrt(np.mean(np.square(init_Q_right - gt_Q_right)))

    # create n-step SARSA agent
    agent = Sarsa(mdp, s, step)
    for run in range(num_runs):
        for i in range(num_episodes):
            agent.episode(discount, step_size, 10000)
            agent.init()
            rms_err_left[i + 1] = np.sqrt(np.mean(np.square(np.asarray(agent.Q[1:-1])[:, 0] - gt_Q_left)))
            rms_err_right[i + 1] = np.sqrt(np.mean(np.square(np.asarray(agent.Q[1:-1])[:, 1] - gt_Q_right)))
        sum_rms_err_left += rms_err_left
        sum_rms_err_right += rms_err_right
        # Reset Q after a run
        agent.reset_Q()
    # averaged over num_runs
    return sum_rms_err_left / num_runs, sum_rms_err_right / num_runs


def plot_RMS_Q():
    """ Plot the RMS error of Q values over all states, averaged over a few runs
        Make two plots: one for Q[s, left], one for Q[s, right]
    """
    num_runs = 50
    num_episodes = 500
    discount = 1
    step_size = 0.1

    # Plot the RMS error over all states, averaged over a few runs
    steps = [1, 5, 10, 20]
    list_sum_rms_err_left = []
    list_sum_rms_err_right = []

    for step in steps:
        sum_rms_err_left, sum_rms_err_right = run_agent_RMS_Q(num_runs, num_episodes, discount, step_size, step)
        list_sum_rms_err_left.append(sum_rms_err_left)
        list_sum_rms_err_right.append(sum_rms_err_right)

    plt.figure(1)
    for i in range(len(list_sum_rms_err_left)):
        plt.plot(range(len(list_sum_rms_err_left[i])), list_sum_rms_err_left[i])
    plt.title("RMS error on Q[left]")
    plt.legend(("n=1", "n=5", "n=10", "n=20"), loc="upper right")
    plt.xlabel("Episodes")

    plt.figure(2)
    for i in range(len(list_sum_rms_err_right)):
        plt.plot(range(len(list_sum_rms_err_right[i])), list_sum_rms_err_right[i])
    plt.title("RMS error on Q[right]")
    plt.legend(("n=1", "n=5", "n=10", "n=20"), loc="upper right")
    plt.xlabel("Episodes")
    plt.show()


def run_agent_RMS_value(num_runs, num_episodes, discount, step_size, step=1):
    """ Run SARSA agent for num_episodes to get the state values """
    mdp = RandomWalk(19, -1)
    s = mdp.init()

    # ground truth for value
    gt_v = np.asarray(mdp.value_equiprobable(discount)[1:-1])
    # initial value
    init_v = np.asarray([0.5] * mdp.num_states())[1:-1]

    # Arrays for RMS error over all states
    rms_err = np.asarray([0.0] * (num_episodes + 1))
    sum_rms_err = np.asarray([0.0] * (num_episodes + 1))
    rms_err[0] = np.sqrt(np.mean(np.square(init_v - gt_v)))

    # create n-step SARSA agent
    agent = Sarsa(mdp, s, step)

    for run in range(num_runs):
        for i in range(num_episodes):
            agent.episode(discount, step_size, 10000)
            agent.init()
            rms_err[i + 1] = np.sqrt(np.mean(np.square(np.asarray(agent.Q_to_value()[1:-1]) - gt_v)))
        sum_rms_err += rms_err
        # Reset Q after a run
        agent.reset_Q()

    # averaged over num_runs
    return sum_rms_err / num_runs


def plot_RMS_value():
    """ Plot the RMS error of values over all states, averaged over a few runs """
    num_runs = 50
    num_episodes = 10
    discount = 1
    step_size = 0.1

    steps = [1, 5, 10, 20]
    list_avg_rms_err = []

    for step in steps:
        avg_rms_err = run_agent_RMS_value(num_runs, num_episodes, discount, step_size, step)
        print("RMS error at episode 0 for {}-step Sarsa: {}".format(step, avg_rms_err[0]))
        list_avg_rms_err.append(avg_rms_err)

    for i in range(len(list_avg_rms_err)):
        print("Plotting {}-step Sarsa".format(steps[i]))
        plt.plot(range(len(list_avg_rms_err[i])), list_avg_rms_err[i])
    plt.title("RMS error on values over all states, averaged over {} runs".format(num_runs))
    plt.legend(("n=1", "n=5", "n=10", "n=20"), loc="upper right")
    plt.xlabel("Episodes")
    plt.show()


def run_agent_value(num_episodes, discount, step_size, step=1):
    """ Run SARSA agent for num_episodes to get the state values"""
    mdp = RandomWalk(19)
    s = mdp.init()
    step = step

    # create n-step SARSA agent
    agent = Sarsa(mdp, s, step)
    for i in range(num_episodes):
        agent.episode(discount, step_size)
        agent.init()

    return agent.Q_to_value()


def plot_value():
    """ Plot the state values for random walk, found by Sarsa agent.
        Compare the effect on n-step.
    """
    num_episodes = 200
    discount = 1
    step_size = 0.1

    value_list = []
    for step in [2 ** x for x in range(5)]:
        value_list.append(run_agent_value(num_episodes, discount, step_size, step))

    mdp = RandomWalk(19)
    # ground truth for V
    gt_v = mdp.value_equiprobable(discount)
    # plot the value
    plt.plot(range(1, 20), gt_v[1:-1], 'ro-', label="True Value")
    colors = ["y", "b", "g", "m", "c"]
    for i, value in enumerate(value_list):
        plt.plot(range(1, 20), value[1:-1], 'o-', color=colors[i], label="{}-step SARSA".format(2 ** i))
    plt.legend(loc="upper left")
    plt.xlabel("State")
    plt.title("Value estimation of n-step Sarsa after {} episodes".format(num_episodes))
    plt.show()


def print_value():
    """ Print the RMS error on state values """
    mdp = RandomWalk(19, 1)
    # ground truth for value
    gt_v = np.asarray(mdp.value_equiprobable(1.0)[1:-1])
    # initial value
    init_v = np.asarray([0.5] * mdp.num_states())[1:-1]
    rms_err = np.sqrt(np.mean(np.square(init_v - gt_v)))
    print("RMS error is ", rms_err)


def plot_RMS_param(load=False):
    """ Plot the sum of RMS error of values over all states, averaged over a few runs
        load = True for loading from pickled file or False(default) for saving to pickled file.
    """
    num_runs = 100
    num_episodes = 10
    discount = 1
    agent_types = ["Sarsa", "expSarsa", "TreeBackup", "Qsigma"]
    agent_type = agent_types[0]

    if not load:
        steps = [2 ** x for x in range(10)]
        alphas = [x / 10 for x in range(1, 11)]
        print(steps)
        print(alphas)

        for agent_type in agent_types:
            # Run for each type of agent
            list_avg_rms_err = np.zeros((len(steps), len(alphas)))
            for i, step in enumerate(steps):
                for j, alpha in enumerate(alphas):
                    avg_rms_err = run_agent_RMS_param(num_runs, num_episodes, discount, alpha, step, agent_type)
                    print("RMS error for {} with n={}, alpha={}: {}".format(agent_type, step, alpha, avg_rms_err))
                    list_avg_rms_err[i, j] = avg_rms_err
            # Save the results to pickled file
            f = open('rms_nstep_{}.p'.format(agent_type), 'wb')
            pickle.dump(list_avg_rms_err, f)
            pickle.dump(steps, f)
            pickle.dump(alphas, f)
            f.close()
    else:
        f = open('rms_nstep_{}.p'.format(agent_type), 'rb')
        list_avg_rms_err = pickle.load(f)
        steps = pickle.load(f)
        alphas = pickle.load(f)

        # Uncomment this part to make the plot
        # plt.rc('text', usetex=True)
        # plt.rc('font', family='serif')
        # colors = ["#ffff00", "#b3b300", "#33cc33", "#ffa31a", "#cc3333", "#cc8033", "#cccc33", "#33a6cc", "#3333cc", "#a633cc"]
        # for i in range(len(steps)):
        #     print("Plotting {}-step {}".format(steps[i], agent_type))
        #     plt.plot(alphas, list_avg_rms_err[i], label="n={}".format(steps[i]), color=colors[i]) # RMS versus alpha
        # plt.title("Average RMS error on values averaged over 19 states, {} episodes and {} runs".format(num_episodes, num_runs))
        # plt.legend(loc="lower center", ncol=5, bbox_to_anchor=(0.6, .01))
        # plt.xlabel(r"$\alpha$")
        # axes = plt.gca()
        # axes.title.set_position((0.5, 1.0))
        # plt.ylim((0.2, 0.55))
        # plt.show()


def run_agent_RMS_param(num_runs, num_episodes, discount, step_size, step=1, agent_type="Sarsa"):
    """ Run the n-step Sarsa agent and return the avg RMS over all states, episodes and runs """
    mdp = RandomWalk(19, -1)
    s = mdp.init()

    # ground truth for value
    gt_v = np.asarray(mdp.value_equiprobable(discount)[1:-1])
    # initial value
    init_v = np.asarray([0.5] * mdp.num_states())[1:-1]

    # Arrays for RMS error over all states
    rms_err = np.asarray([0.0] * num_episodes)
    sum_rms_err = 0.0
    # rms_err[0] = np.sqrt(np.mean(np.square(init_v - gt_v)))

    # create n-step agent
    print("Starting agent {}-step {}".format(step, agent_type))
    if agent_type.lower() == "sarsa":
        agent = Sarsa(mdp, s, step)
    elif agent_type.lower() == "expsarsa":
        agent = ExpSARSA(mdp, s, step)
    elif agent_type.lower() == "treebackup":
        agent = TreeBackup(mdp, s, step)
    elif agent_type.lower() == "qsigma":
        agent = QSigma(mdp, 0.5, s, step)
    else:
        raise Exception("Wrong type of agent")

    for run in range(num_runs):
        for i in range(num_episodes):
            agent.episode(discount, step_size, 10000)
            agent.init()
            rms_err[i] = np.sqrt(np.mean(np.square(np.asarray(agent.Q_to_value()[1:-1]) - gt_v)))
        sum_rms_err += np.sum(rms_err)
        # Reset Q after a run
        agent.reset_Q()

    # averaged over num_runs and num_episodes
    return sum_rms_err / (num_runs * num_episodes)


def plot_Q_RMS_param(load=False):
    """ Plot the average RMS error of Q values as a function of alpha and n,
        load = True for loading from pickled file or False(default) for saving to pickled file.
    """
    num_runs = 100
    discount = 1
    agent_types = ["Sarsa", "expSarsa", "TreeBackup", "Qsigma"]
    agent_names = ["Sarsa", "Expected Sarsa", "Tree Backup", r"Q($\sigma$)"]

    for agent_idx in range(4):
        agent_type = agent_types[agent_idx]
        agent_name = agent_names[agent_idx]
        for num_episodes in [10,50]:
            if not load:
                steps = [2 ** x for x in range(10)]
                alphas = [x / 10 for x in range(1, 11)]
                print(steps)
                print(alphas)

                for agent_type in agent_types:
                    # Run for each type of agent
                    list_avg_rms_err = np.zeros((len(steps), len(alphas)))
                    for i, step in enumerate(steps):
                        for j, alpha in enumerate(alphas):
                            avg_rms_err = run_agent_Q_RMS_param(num_runs, num_episodes, discount, alpha, step, agent_type)
                            print("RMS error for {} with n={}, alpha={}: {}".format(agent_type, step, alpha, avg_rms_err))
                            list_avg_rms_err[i, j] = avg_rms_err
                    # Save the results to pickled file
                    f = open('rms_nstep_Q_{}_{}.p'.format(num_episodes, agent_type), 'wb')
                    pickle.dump(list_avg_rms_err, f)
                    pickle.dump(steps, f)
                    pickle.dump(alphas, f)
                    f.close()
            else:
                f = open('rms_nstep_Q_{}_{}.p'.format(num_episodes, agent_type), 'rb')
                list_avg_rms_err = pickle.load(f)
                steps = pickle.load(f)
                alphas = pickle.load(f)

                # Uncomment this part to make the plot
                plt.cla()
                plt.rc('text', usetex=True)
                plt.rc('font', family='serif')
                # colors in the textbook
                colors = ["r", "#81fca9", "b", "#777777", "#ffcafb", "#6afefe", "#b96cb1", "#ff00ff", "#ffab6a", "#dbcd95"]
                for i in range(len(steps)):
                    print("Plotting {}-step {}".format(steps[i], agent_type))
                    plt.plot(alphas, list_avg_rms_err[i], label="n={}".format(steps[i]), color=colors[i]) # RMS versus alpha
                plt.title("Average RMS error averaged over {} episodes and {} runs, {}".format(num_episodes, num_runs, agent_name))
                lgd = plt.legend(loc=2, ncol=1, bbox_to_anchor=(1.05, 1))
                plt.xlabel(r"$\alpha$")
                plt.ylabel("Average  RMS  Error")
                if num_episodes == 50:
                    plt.ylim((0.1, 0.55))
                else:
                    plt.ylim((0.2, 0.55))
                plt.savefig('{}_{}.pdf'.format(agent_type, num_episodes).format(agent_type), bbox_inches='tight', format='pdf', dpi=600)
                #plt.show()


def run_agent_Q_RMS_param(num_runs, num_episodes, discount, step_size, step=1, agent_type="Sarsa"):
    """ Run the n-step Sarsa agent and return the avg Q-value RMS over episodes and runs """
    mdp = RandomWalk(19, -1)
    s = mdp.init()

    # ground truth for value
    gt_v = np.asarray(mdp.Q_equiprobable(discount)[1:-1])
    # Arrays for RMS error over all states
    rms_err = np.asarray([0.0] * num_episodes)
    sum_rms_err = 0.0

    # create n-step agent
    print("Starting agent {}-step {}".format(step, agent_type))
    if agent_type.lower() == "sarsa":
        agent = Sarsa(mdp, s, step)
    elif agent_type.lower() == "expsarsa":
        agent = ExpSARSA(mdp, s, step)
    elif agent_type.lower() == "treebackup":
        agent = TreeBackup(mdp, s, step)
    elif agent_type.lower() == "qsigma":
        agent = QSigma(mdp, 0.5, s, step)
    else:
        raise Exception("Wrong type of agent")

    for run in range(num_runs):
        for i in range(num_episodes):
            agent.episode(discount, step_size, 10000)
            agent.init()
            rms_err[i] = np.sqrt(np.mean(np.square(np.asarray(agent.Q[1:-1]) - gt_v)))
        sum_rms_err += np.sum(rms_err)
        # Reset Q after a run
        agent.reset_Q()

    # averaged over num_runs and num_episodes
    return sum_rms_err / (num_runs * num_episodes)


if __name__ == '__main__':
    seed(0)
    # plot_RMS_Q()
    # plot_RMS_value()
    # plot_value()
    # print_value()
    # plot_RMS_param(True)
    plot_Q_RMS_param(True)
