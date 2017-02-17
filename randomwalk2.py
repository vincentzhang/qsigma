""" random walk MDP """
from MDPy import MDP


class RandomWalk(MDP):

    def __init__(self, num_states=19, left_reward=0):
        """ Initialize the MDP """
        super(RandomWalk, self).__init__()
        self.add_states(num_states+2) # 0 and num_states+1 are the two terminal states
        for s in range(1, num_states+1): # start from the left most non-terminal state
            self.add_actions(s, 1) # two actions, left(0) and right(1)
            # add transitions (s', r, P) for each state-action pair
            if s == 1: # for the left most state
                self.add_transition(s, 0, (s-1, left_reward, 0.5))
            else:
                self.add_transition(s, 0, (s-1, 0, 0.5))
            if s == num_states:
                # For the rightmost state
                self.add_transition(s, 0, (s+1, 1, 0.5))
            else:
                self.add_transition(s, 0, (s+1, 0, 0.5))

    def init(self):
        """ Return the initial state """
        # states are numbered from left to right
        # define the state in the middle as the starting state
        return int(self.num_states()/2)


def example():
    mdp = RandomWalk(19)
    s = mdp.init()
    print("The initial state index is: ".format(s))
    while not mdp.terminal(s):
        s, r = mdp.do_action(s, 1)
        print("The next reward, state pair is: ({},{})".format(r, s))
    print("Ended at the terminal state: {}".format(s))

    print("Q_equiprobable[s][a] ", mdp.Q_equiprobable(0.9))
    print("Q_eps_greedy[s][a]   ", mdp.Q_eps_greedy(0.1, 0.9))
    print("V_eps_greedy[s]      ", mdp.value_eps_greedy(0.1, 0.9))

if __name__ == '__main__':
    example()
