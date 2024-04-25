import gym
import numpy as np
import sys
sys.path.append('/Users/jamesrogers/.git/blackjack-agent')
from blackjack import BlackjackEnv

class ValueIterationAgent:
    def __init__(self, temperature_values, gamma=1, theta=0.0001):
        self.temperature_values = temperature_values
        self.gamma = gamma
        self.theta = theta
        self.state_values = np.zeros((32, 11, 2, 10), dtype=float)  # States with temperature (Your Hand)(Dealer showing)(usable ace)(temperature)
        self.policy = np.zeros((32, 11, 2, 10), dtype=int)  # Policy with temperature

    def value_iteration(self, env):
        count = 1
        while True:
            delta = 0
            for player_sum in range(1, 32):
                for dealer_showing in range(1, 11):
                    for usable_ace in range(2):
                        for temperature in range(10):
                            v_old = self.state_values[player_sum, dealer_showing, usable_ace, temperature]
                            v_new = self.evaluate_actions(env, player_sum, dealer_showing, usable_ace, temperature)
                            self.state_values[player_sum, dealer_showing, usable_ace, temperature] = v_new
                            delta = max(delta, abs(v_old - v_new))
            if delta < self.theta:
                break
            else:
                print(f'{count}: {delta}')
                count += 1
        self.extract_policy(env)

    def evaluate_actions(self, env, player_sum, dealer_showing, usable_ace, temperature):
        actions_values = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            actions_values[action] = self.calculate_state_value(player_sum, dealer_showing, usable_ace, action, temperature)
        return np.max(actions_values)
    
    def _calculate_dealer_odds(self, dealer_showing, temperature, dealer_odds, weight = 1.0):
        for card_value, prob in enumerate(self.temperature_values[temperature]):
                dealer_sum = dealer_showing + card_value + 1
                if dealer_sum in dealer_odds:
                    dealer_odds[dealer_sum] += weight * prob
                else:
                    if dealer_sum < 21:
                        ## BUGFIX ## was missing prob * weight
                        self._calculate_dealer_odds(dealer_sum, temperature, dealer_odds, weight = prob * weight)

    def calculate_state_value(self, player_sum, dealer_showing, usable_ace, action, temperature):
        if action == 0:  # stick

            # The expected_reward will be accumulated
            expected_reward = 0

            # Dictionary, will contain the dealer's odds of achieving these hands (sticking on a soft 17)
            # Could be memoized
            dealer_odds = {17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0}

            # Busting will be the probability of none of the above
            bust = 1.0

            # Should calculate the odds of the dealer achieving each of the above hands
            self._calculate_dealer_odds(dealer_showing, temperature, dealer_odds)
            for dealer_hand_value in dealer_odds:
                bust -= dealer_odds[dealer_hand_value]
            
            expected_reward = 0.0
            
            for dealer_hand_value in dealer_odds:
                if dealer_hand_value > player_sum:
                    expected_reward -= 1 * dealer_odds[dealer_hand_value]
                elif dealer_hand_value == player_sum:
                    expected_reward += 0
                elif dealer_hand_value < player_sum:
                    expected_reward += 1 * dealer_odds[dealer_hand_value]
            
            expected_reward += 1 * bust
            
            return expected_reward
        else:  # hit
            expected_reward = 0
            for card_value, prob in enumerate(self.temperature_values[temperature]):
                new_sum = player_sum + card_value + 1  # card_value is 0-indexed, card values start from 1
                if usable_ace and new_sum + 10 <= 21:  # Check for usable ace
                    new_sum += 10
                if new_sum > 21:
                    reward = -1  # bust
                else:
                    # Recursively calculate the value of the new state
                    reward = self.state_values[new_sum, dealer_showing, int(new_sum <= 21), temperature]

                expected_reward += prob * reward

            return expected_reward

    def extract_policy(self, env):
        for player_sum in range(1, 32):
            for dealer_showing in range(1, 11):
                for usable_ace in range(2):
                    for temperature in range(10):  # Loop over temperatures
                        action_values = np.zeros(env.action_space.n)
                        for action in range(env.action_space.n):
                            action_values[action] = self.calculate_state_value(player_sum, dealer_showing, usable_ace, action, temperature)
                        self.policy[player_sum, dealer_showing, usable_ace, temperature] = np.argmax(action_values)

def cmp(a, b):
    return float(a > b) - float(a < b)

# Initialize your custom environment
env = BlackjackEnv()

# Create an array of temperature values based on your deck temperatures
# Replace the following line with your actual temperature values
temperature_values = np.load('probabilities.npy')

# Initialize the agent and perform value iteration
agent = ValueIterationAgent(temperature_values)
agent.value_iteration(env)

# Display the learned policy
print("Learned Policy:")
print(agent.policy)

# Save the learned policy as 'learnedpolicy.npy'
np.save('learnedpolicy2.npy', agent.policy)
