import numpy as np
import sys
sys.path.append('/Users/jamesrogers/.git/blackjack-agent')


from Environment.blackjack import cmp, score
#draw_card, draw_hand, usable_ace, sum_hand, is_bust, is_natural

class ValueIterationAgent:
    def __init__(self, gamma=1.0, theta=0.0001):
        self.gamma = gamma  # discount factor
        self.theta = theta  # threshold for convergence
        self.state_values = np.zeros((32, 11, 2))  # value for each state
        self.policy = np.zeros((32, 11, 2), dtype=int)  # policy: 0 for stick and 1 for hit

    def value_iteration(self, env):
        # Iterate until the value function converges
        while True:
            delta = 0
            for player_sum in range(1, 32):
                for dealer_showing in range(1, 11):
                    for usable_ace in range(2):
                        v_old = self.state_values[player_sum, dealer_showing, usable_ace]
                        v_new = self.evaluate_actions(env, player_sum, dealer_showing, usable_ace)
                        self.state_values[player_sum, dealer_showing, usable_ace] = v_new
                        delta = max(delta, abs(v_old - v_new))
            
            # Break out of the loop when the value function converges
            if delta < self.theta:
                break

        self.extract_policy(env)

    def evaluate_actions(self, env, player_sum, dealer_showing, usable_ace):
        # Evaluate the expected return of each action and return the maximum
        actions_values = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            # For simplicity, we assume equal probability for all possible next states
            # This is a simplification, the actual environment is more complex
            for next_player_sum in range(1, 32):
                for next_dealer_showing in range(1, 11):
                    for next_usable_ace in range(2):
                        next_state = (next_player_sum, next_dealer_showing, next_usable_ace)
                        reward, prob = self.get_transition_prob(env, player_sum, dealer_showing, usable_ace, action, next_state)
                        actions_values[action] += prob * (reward + self.gamma * self.state_values[next_state])
        return np.max(actions_values)

    def get_transition_prob(self, env, player_sum, dealer_showing, usable_ace, action, next_state):
        # This function should return the transition probability and reward for moving
        # from the current state to the next state given an action. However, for simplicity,
        # we will assume that each transition has an equal probability and a standard reward
        # structure: -1 for losing, 0 for draw, 1 for winning.
        # Note: This is a simplification. The actual environment has a more complex dynamics.
        prob = 1 / (31 * 10 * 2)  # Simplified uniform probability
        if action == 0:  # stick
            reward = cmp(score([player_sum]), score([dealer_showing]))  # Simplified reward
        else:  # hit
            reward = 0 if player_sum <= 21 else -1  # Simplified reward for hitting
        return reward, prob

    def extract_policy(self, env):
        # Extract policy from the value function
        for player_sum in range(1, 32):
            for dealer_showing in range(1, 11):
                for usable_ace in range(2):
                    action_values = np.zeros(env.action_space.n)
                    for action in range(env.action_space.n):
                        for next_player_sum in range(1, 32):
                            for next_dealer_showing in range(1, 11):
                                for next_usable_ace in range(2):
                                    next_state = (next_player_sum, next_dealer_showing, next_usable_ace)
                                    reward, prob = self.get_transition_prob(env, player_sum, dealer_showing, usable_ace, action, next_state)
                                    action_values[action] += prob * (reward + self.gamma * self.state_values[next_state])
                    best_action = np.argmax(action_values)
                    self.policy[player_sum, dealer_showing, usable_ace] = best_action

# Initialize the environment
env = BlackjackEnv()

# Initialize the agent and perform value iteration
agent = ValueIterationAgent()
agent.value_iteration(env)

# Display the learned policy
agent.policy[:,:,:]  # Display the policy for all states

