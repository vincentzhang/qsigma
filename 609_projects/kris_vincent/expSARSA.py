from MDPy import MDP
from random import random
from random import randint, seed
from agent import Agent


class ExpSARSA(Agent):
  def __init__(self, mdp, init_state=None, steps=1):
    super(ExpSARSA, self).__init__(mdp, init_state)
    self._n = steps

  def act(self, action, discount, step_size):
    """ take an action """
    # experience transition
    (sp, r) = self._env.do_action(self._s, action)
    self._r_sum += r
    self._time += 1
    # save transition
    self._tr.append((self._s, action, r))
    if len(self._tr) > self._n:
      self._tr.pop(0)
    # n-step exp-SARSA update
    G = self.expected_Q(sp)
    for i in range(len(self._tr) - 1, -1, -1):
      (s, a, r) = self._tr[i]
      G = r + discount * G
      self._Q[s][a] += step_size * (G - self._Q[s][a])
    # update agent state
    self._s = sp

  def act_policy(self, discount, step_size):
    """ take an action according to the agent's policy """
    if self._policy == 'eps_greedy':
      if random() < self._eps:
        self.act(randint(0, self._env.num_actions(self._s) - 1), discount, step_size)
      else:
        self.act(self._Q[self._s].index(max(self._Q[self._s])), discount, step_size)
      return
    if self._policy == 'equiprobable':
      self.act(randint(0, self._env.num_actions(self._s) - 1), discount, step_size)
      return
    if self._policy == 'custom':
      sample = random()
      thresh = 0.0
      for i in range(len(self._P[self._s])):
        thresh += self._P[self._s][i]
        if sample < thresh:
          self.act(i, discount, step_size)
          return

  def episode(self, discount, step_size, max_steps=1e3):
    """ Run n-step expected-Sarsa for one episode. """
    self._tr = [] # reset the buffer
    while self._time < max_steps and not self._env.terminal(self._s):
      self.act_policy(discount, step_size)
    return self.mean_reward

def example():
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

  # create 3-step expected SARSA agent with an equiprobable random policy
  agent = ExpSARSA(mdp, 0, 3)
  agent2 = ExpSARSA(mdp, 0, 3)

  agent.set_policy_equiprobable()
  agent2.set_policy_eps_greedy(0.1)

  seed(0)
  # act under its policy with discount = 0.9, and step size = 0.1
  for iter in range(100000):
    agent.act_policy(0.9, 0.1)

  seed(0)
  # act under its policy with discount = 0.9, and step size = 0.1
  agent2.episode(0.9, 0.1, 100000)

  # output Q from DP and from the agent
  print('Q_DP[s][a]      ', mdp.Q_equiprobable(0.9))
  print('Q_eps_greedy[s][a]   ', mdp.Q_eps_greedy(0.1, 0.9))
  print('Equiprobable Q_expSARSA[s][a]', agent.Q)
  print('Eps greedy   Q_expSARSA[s][a]', agent2.Q)

if __name__ == '__main__':
  example()