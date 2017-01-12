class Agent(object):
  """ The base class for agent.
        The API somewhat follows the guidelines at
        https://webdocs.cs.ualberta.ca/~sutton/RLinterface/RLI-Cplusplus.html
  """
  def __init__(self, mdp, init_state=None):
    self._env = mdp # MDP
    self._s0 = init_state if init_state is not None else mdp.init() # the initial state
    self._s = self._s0 # the current state
    self._policy = 'equiprobable'
    self._Q = []
    for s in range(self._env.num_states()):
      self._Q.append([0.0] * self._env.num_actions(s))
    self._tr = [] # the state transitions that the agent has experienced
    self._r_sum = 0.0 # sum of rewards
    self._time = 0 # timer

  def init(self, init_state=None):
    """ Reset the agent back to the initial state, clears the rewards and timer """
    self._s = self._s0 if init_state is None else init_state
    self._r_sum = 0.0
    self._time = 0

  def step(self, action):
    """ Take one action and return the next state and reward """
    return self._env.do_action(self._s, action)

  def set_policy_eps_greedy(self, epsilon):
    """ use epsilon-greedy policy """
    self._policy = 'eps_greedy'
    self._eps = epsilon

  def set_policy_equiprobable(self):
    """ use equiprobable random policy """
    self._policy = 'equiprobable'

  def set_policy(self, policy):
    """ use a specific policy """
    self._policy = 'custom'
    self._P = policy

  @property
  def state(self):
    """ returns the current state of the agent """
    return self._s

  @state.setter
  def state(self, state):
    """ sets the agent's state """
    self._s = state

  @property
  def Q(self):
    """ gets the agent's learned action-value function """
    return self._Q

  @property
  def reward(self):
    """ returns the total reward the agent has received """
    return self._r_sum

  @property
  def time_steps(self):
    """ returns the number of time steps that have elapsed """
    return self._time

  @property
  def mean_reward(self):
    """ returns the mean reward the agent has received """
    return self._r_sum / self._time


  def reset_Q(self):
    """ clear the agent's learned action-value function """
    self._Q = []
    for s in range(self._env.num_states()):
      self._Q.append([0.0] * self._env.num_actions(s))

  def expected_Q(self, sp):
    """ get the expected Q under a target policy """
    if self._policy == 'eps_greedy':
      Q_exp = (1.0 - self._eps) * max(self._Q[sp])
      for a in range(self._env.num_actions(sp)):
        Q_exp += (self._eps / self._env.num_actions(sp)) * self._Q[sp][a]
      return Q_exp
    if self._policy == 'equiprobable':
      Q_exp = 0.0
      for a in range(self._env.num_actions(sp)):
        Q_exp += (1.0 / self._env.num_actions(sp)) * self._Q[sp][a]
      return Q_exp
    if self._policy == 'custom':
      Q_exp = 0.0
      for a in range(self._env.num_actions(sp)):
        Q_exp += self._P[sp][a] * self._Q[sp][a]
      return Q_exp

  def Q_to_value(self):
    """ Compute and return state values from Q values """
    V = []
    for s in range(self._env.num_states()):
      V.append(self.expected_Q(s))
    return V