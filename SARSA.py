from MDPy import MDP
from randomwalk import RandomWalk
from random import randint, random, choice, seed
from agent import Agent


class Sarsa(Agent):
  def __init__(self, mdp, init_state=None, steps=1):
    super(Sarsa, self).__init__(mdp, init_state)
    self._n = steps

  def act(self, action, discount, step_size):
    """ do an action and update Q given the discount factor and step size """
    if self._time < self._T:
        # experience transition
        (sp, r) = self.step(action)
        self._r_sum += r
        # save transition
        self._tr.append((self._s, action, r))
        if self._env.terminal(sp): # if terminal
            self._T = self._time + 1
        else: # commit the next action
            action = self.pick_action()
        # update agent state
        self._s = sp
    # time whose estimate is being updated
    self._tau = self._time + 1 - self._n
    if self._tau >= 0:
        G = 0.0
        for i in range(0, int(min(self._n-1, self._T-self._tau-1))+1):
            G += (discount ** i) * self._tr[i][-1]
        (s, a, r) = self._tr.pop(0) # delete the old entries, we only need at most n transitions
        if (self._time + 1) < self._T: # if the Q is not for terminal state
            G += (discount ** self._n) * self._Q[sp][action]
        self._Q[s][a] += step_size * (G - self._Q[s][a])
    self._time += 1
    return action # return the committed next action

  def pick_action(self):
    """ return an action according to a given policy P[s][a] """
    if self._policy == 'eps_greedy':
      if random() < self._eps:
        return randint(0, self._env.num_actions(self._s) - 1)
      else:
        max_Q = max(self._Q[self._s])
        indices = [i for i, x in enumerate(self._Q[self._s]) if x == max_Q]
        if len(indices) == 1:
            return indices[0]
        else: # break ties randomly to prevent the case where committing to one action leads to no reward
            return choice(indices)
    if self._policy == 'equiprobable':
      return randint(0, self._env.num_actions(self._s) - 1)
    if self._policy == 'custom':
      sample = random()
      thresh = 0.0
      for i in range(len(self._P[self._s])):
        thresh += self._P[self._s][i]
        if sample < thresh:
          return i

  def episode(self, discount, step_size, max_steps=1e3):
    """ Run n-step sarsa for one episode.
        Algorithm follow the pseudocode on page 155 of the RL textbook.
    """
    self._T = float('inf')
    self._tau = 0
    self._tr = []
    action = self.pick_action()
    while (self._tau != (self._T-1)) and (self._time < max_steps):
      action = self.act(action, discount, step_size)

  def episode_sarsa(self, discount, step_size, max_steps=1e3):
    """ Run 1-step Sarsa for one episode """
    action = self.pick_action()
    while self.state and (self._time < max_steps):
      action = self.act_sarsa(action, discount, step_size)

  def act_sarsa(self, action, discount, step_size):
    """ Run 1-step Sarsa for one step """
    (sp, r) = self.step(action)
    self._r_sum += r
    self._time += 1
    if not sp: # if terminal
      action_p = None
      G = r
    else:
      # next action
      action_p = randint(0,1)
      G = r + discount * self._Q[sp][action_p]
    # update Q value
    self._Q[self._s][action] += step_size * (G - self._Q[self._s][action])
    # update agent state
    self._s = sp
    return action_p



def example_randomwalk():
  # create an MDP
  env = RandomWalk(19)

  # create 1-step SARSA agent
  agent = Sarsa(env, env.init(), 1)
  agent2 = Sarsa(env, env.init(), 1)

  # act using equiprobable random policy with discount = 0.9 and step size = 0.1
  num_episode = 100
  for iter in range(num_episode):
    agent.episode(0.9, 0.1)
    agent.init()

  agent2.set_policy_eps_greedy(0.5)
  for iter in range(num_episode):
    agent2.episode(0.9, 0.1)
    agent2.init()

  print('Equiprobable Q_SARSA[s][a]', agent.Q)
  print('Eps greedy   Q_SARSA[s][a]', agent2.Q)

def example():
  # create an MDP
  mdp = MDP()

  # add 2 states + 1 terminal state
  mdp.add_states(3)

  # add 2 actions to each state
  mdp.add_actions(0, 2)
  mdp.add_actions(1, 2)
  mdp.add_actions(2, 2)

  # add transitions (s', r, P) for each state-action pair
  mdp.add_transition(1, 0, (1, 0.0, 1.0))
  mdp.add_transition(1, 1, (2, -1.0, 1.0))
  mdp.add_transition(2, 0, (2, 5.0, 1.0))
  mdp.add_transition(2, 1, (1, -1.0, 1.0))

  # create 1-step SARSA agent
  agent = Sarsa(mdp, 1, 3)
  agent2 = Sarsa(mdp, 1, 3)

  # act using equiprobable random policy with discount = 0.9 and step size = 0.1
  num_episode = 1
  seed(0)
  for iter in range(num_episode):
    agent.episode(0.9, 0.1, 100000)
    agent.init(1)

  seed(0)
  agent2.set_policy_eps_greedy(0.1)
  for iter in range(num_episode):
   agent2.episode(0.9, 0.1, 100000)
   agent2.init(1)

  print('Q_DP[s][a]   ', mdp.Q_equiprobable(0.9))
  print('Q_eps_greedy[s][a]   ', mdp.Q_eps_greedy(0.1, 0.9))
  #print('Q_iteration[s][a]   ', mdp.Q_iteration(0.9))
  print('Equiprobable Q_SARSA[s][a]', agent.Q)
  print('Eps greedy   Q_SARSA[s][a]', agent2.Q)

if __name__ == '__main__':
  example()
  #example_randomwalk()
