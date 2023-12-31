import gym
import numpy as np

class ValueIterationAgent:
    def __init__(self, gamma=1.0, theta=0.0001):
        self.gamma = gamma
        self.theta = theta
        self.state_values = np.zeros((32, 11, 2))  # 32 for player sum, 11 for dealer card, 2 for usable ace
        self.policy = np.zeros((32, 11, 2), dtype=int)  # Policy: 0 (stick) or 1 (hit)

    def value_iteration(self, env):
        while True:
            delta = 0
            for player_sum in range(1, 32):
                for dealer_showing in range(1, 11):
                    for usable_ace in range(2):
                        v_old = self.state_values[player_sum, dealer_showing, usable_ace]
                        v_new = self.evaluate_actions(env, player_sum, dealer_showing, usable_ace)
                        self.state_values[player_sum, dealer_showing, usable_ace] = v_new
                        delta = max(delta, abs(v_old - v_new))
            if delta < self.theta:
                break
        self.extract_policy(env)

    def evaluate_actions(self, env, player_sum, dealer_showing, usable_ace):
        actions_values = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            actions_values[action] = self.calculate_state_value(player_sum, dealer_showing, usable_ace, action)
        return np.max(actions_values)

    def calculate_state_value(self, player_sum, dealer_showing, usable_ace, action):
        if action == 0:  # stick
            return cmp(player_sum, dealer_showing)  # Compare player sum and dealer showing card
        else:  # hit
            return 0 if player_sum <= 21 else -1  # Reward for hitting

    def extract_policy(self, env):
        for player_sum in range(1, 32):
            for dealer_showing in range(1, 11):
                for usable_ace in range(2):
                    action_values = np.zeros(env.action_space.n)
                    for action in range(env.action_space.n):
                        action_values[action] = self.calculate_state_value(player_sum, dealer_showing, usable_ace, action)
                    self.policy[player_sum, dealer_showing, usable_ace] = np.argmax(action_values)

def cmp(a, b):
    return float(a > b) - float(a < b)

# Initialize the environment
env = gym.make('Blackjack-v1')

# Initialize the agent and perform value iteration
agent = ValueIterationAgent()
agent.value_iteration(env)

# Display the learned policy
print("Learned Policy:")
print(agent.policy)

np.save('learnedpolicy.npy', agent.policy)

