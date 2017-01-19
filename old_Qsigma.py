from MDPy import MDP
from randomwalk import RandomWalk
from random import randint, random, choice, seed
from agent import Agent


class QSigma(Agent):
  def __init__(self, mdp, Psigma=0.5, init_state=None, steps=1):
    super(QSigma, self).__init__(mdp, init_state)
    self._n = steps
    self._Psigma = Psigma # probability of (sigma=1)

  def episode(self, discount, step_size, max_steps=1e3):
    """ Run n-step Q(sigma) for one episode.
        Algorithm is adapted from treebackup,  adding the sigma term
    """
    self._T = float('inf')
    self._tau = 0
    action = self.pick_action(self._s)
    self._tr = [(self._s, 0)] * self._n # list of (state, action), no need for reward here
    self._delta = [0.0] * self._n
    # list of old Q values, this is different from self._Q because the TD error term `delta_t` needs Q at time (t-1)
    # note that the buffer size here should be (n+1)
    self._Qt = [self._Q[self._s][action]] * (self._n+1)
    self._pi = [0.0] * self._n
    self._sigma = [0.0] * self._n
    while (self._tau != (self._T-1)) and (self._time < max_steps):
      action = self.act(action, discount, step_size)
    return self.mean_reward

  def act(self, action, discount, step_size):
    """ do an action and update Q given the discount factor and step size """
    if self._time < self._T:
        # experience transition if not at terminal state
        (sp, r) = self.step(action)
        self._r_sum += r
        self._tr[self._time%self._n] = (self._s, action)
        if self._env.terminal(sp): # if terminal
            self._T = self._time + 1
            self._delta[self._time%self._n] = r - self._Qt[self._time%(self._n+1)] # TD error
        else: # commit the next action
            # Select arbitrarily and store an action as A_(t+1)
            action = self.pick_action(sp)
            # Store Q(St+1;At+1) as Qt+1
            self._Qt[(self._time + 1)%(self._n+1)] = self._Q[sp][action]
            # randomly pick sigma
            sigma = 1 if random() < self._Psigma else 0
            self._sigma[(self._time+1)%self._n] = sigma
            self._delta[self._time%self._n] = r - self._Qt[self._time%(self._n+1)] + \
                                              discount*((1-sigma) * self.expected_Q(sp) + sigma * self._Q[sp][action])
            # Store pi(At+1 | St+1) as pi(t+1),
            self._pi[(self._time+1)%self._n] = self.get_action_probability(sp, action)
        # update agent state
        self._s = sp
    # time whose estimate is being updated
    self._tau = self._time + 1 - self._n
    if self._tau >= 0:
        E = 1.0
        G = self._Qt[self._tau%(self._n+1)]
        for k in range(self._tau, int(min(self._time, self._T-1))+1):
            G += E * self._delta[k%self._n]
            E *= discount * ((1-self._sigma[(k+1)%self._n]) * self._pi[(k+1)%self._n] + self._sigma[(k+1)%self._n])
        s, a = self._tr[self._tau%self._n]
        self._Q[s][a] += step_size * (G - self._Q[s][a])
    self._time += 1
    return action # return the committed next action

  def get_action_probability(self, state, action):
    """ return the action probability at a state of a given policy P[s][a] """
    if self._policy == 'eps_greedy':
      # compute the maximum actions first since we need this to reason about the probability of an action
      max_Q = max(self._Q[state])
      indices = [i for i, x in enumerate(self._Q[state]) if x == max_Q]
      if action in indices:
          return self._eps/self._env.num_actions(state) + (1.0 - self._eps)/len(indices)
      else:
          return self._eps/self._env.num_actions(state)
    if self._policy == 'equiprobable':
      return 1.0/self._env.num_actions(state)
    if self._policy == 'custom':
      return self._P[state][action]

  def get_action_probabilities(self, state):
    """ return a list of probabilities for all actions under a state according to a given policy P[s][a]
        Return: [(action_1, prob_1),(action_2, prob_2), ...]
    """
    if self._policy == 'eps_greedy':
      # compute the maximum actions first since we need this to reason about the probability of an action
      max_Q = max(self._Q[state])
      indices = [i for i, x in enumerate(self._Q[state]) if x == max_Q]
      # [prob_action_1, prob_action_2, ... prob_action_N]
      prob_list = [self._eps/self._env.num_actions(state)] * self._env.num_actions(state)
      for a in indices:
          prob_list[a] += (1.0 - self._eps)/len(indices)
      return list(zip(range(self._env.num_actions(state)), prob_list))
    if self._policy == 'equiprobable':
      prob_list = [1.0/self._env.num_actions(state)] * self._env.num_actions(state)
      return list(zip(range(self._env.num_actions(state)), prob_list))
    if self._policy == 'custom':
      return list(zip(range(self._env.num_actions(state)), self._P[state]))

  def pick_action(self, state):
    """ return an action according to a given policy P[s][a] """
    if self._policy == 'eps_greedy':
      if random() < self._eps:
        return randint(0, self._env.num_actions(state) - 1)
      else:
        max_Q = max(self._Q[state])
        indices = [i for i, x in enumerate(self._Q[state]) if x == max_Q]
        if len(indices) == 1:
            return indices[0]
        else: # break ties randomly to prevent the case where committing to one action leads to no reward
            return choice(indices)
    if self._policy == 'equiprobable':
      return randint(0, self._env.num_actions(state) - 1)
    if self._policy == 'custom':
      sample = random()
      thresh = 0.0
      for i in range(len(self._P[state])):
        thresh += self._P[state][i]
        if sample < thresh:
          return i


def example_randomwalk():
  """ An example on random walk MDP """
  # create an MDP
  env = RandomWalk(19, -1)

  # create n-step QSigma agent
  agent = QSigma(env, 0.5, env.init(), 3) #Psigma=0.5, init_state=env.init(), steps=3
  agent2 = QSigma(env, 0.5, env.init(), 3)

  # act using equiprobable random policy with discount = 0.9 and step size = 0.1
  num_episode = 1000
  for iter in range(num_episode):
    agent.episode(0.9, 0.1)
    agent.init()

  agent2.set_policy_eps_greedy(0.1)
  for iter in range(num_episode):
   agent2.episode(0.9, 0.1)
   agent2.init()

  print('Q_DP[s][a]   ', env.Q_equiprobable(0.9))
  print('Q_eps_greedy[s][a]   ', env.Q_eps_greedy(0.1, 0.9))
  print('Equiprobable Q_Q(sigma)[s][a]', agent.Q)
  print('Eps greedy   Q_Q(sigma)[s][a]', agent2.Q)

def example():
  """ An example with a simple MDP """
  # create an MDP
  mdp = MDP()

  # add 2 states
  mdp.add_states(2)

  # add 2 actions to the first state and 1 action to the second state
  mdp.add_actions(0, 2)
  mdp.add_actions(1, 1)

  # add transitions (s', r, P) for each state-action pair
  mdp.add_transition(0, 0, (0, 0.5, 1.0))
  mdp.add_transition(0, 1, (0, -1.0, 0.3))
  mdp.add_transition(0, 1, (1, -1.0, 0.7))
  mdp.add_transition(1, 0, (0, 5.0, 0.6))
  mdp.add_transition(1, 0, (1, -1.0, 0.4))

  # create 3-step Qsigma agent
  agent = QSigma(mdp, 0.5, 0, 3) #Psigma=0.5, init_state=0, steps=3
  agent2 = QSigma(mdp, 0.5, 0, 3)

  # act using equiprobable random policy with discount = 0.9 and step size = 0.1
  num_episode = 1
  seed(0)
  for iter in range(num_episode):
    agent.episode(0.9, 0.1, 100000)
    agent.init(0)

  seed(0)
  agent2.set_policy_eps_greedy(0.1)
  for iter in range(num_episode):
    agent2.episode(0.9, 0.1, 100000)
    agent2.init(0)

  print('Q_DP[s][a]   ', mdp.Q_equiprobable(0.9))
  print('Q_eps_greedy[s][a]   ', mdp.Q_eps_greedy(0.1, 0.9))
  print('Equiprobable Q_Q(sigma)[s][a]', agent.Q)
  print('Eps greedy   Q_Q(sigma)[s][a]', agent2.Q)

if __name__ == '__main__':
  example()
  #example_randomwalk()
