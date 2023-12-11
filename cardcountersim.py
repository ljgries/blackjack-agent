import gym
import numpy as np
from blackjack import BlackjackEnv

class GameState:
    def __init__(self, observation, env, done=False, reward=0):
        self.observation = observation
        self.env = env
        self.done = done
        self.reward = reward

    def take_action(self, action):
        observation, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        return GameState(observation, self.env, done, reward)

    def is_terminal(self):
        return self.done

def play_game_with_policy(env, policy):
    observation = env.reset()
    total_reward = 0
    done = False

    while not done:
        # Check if the observation is a tuple containing another tuple and a dictionary
        if isinstance(observation, tuple) and len(observation) == 2 and isinstance(observation[0], tuple):
            game_state = observation[0]
        else:
            game_state = observation

        player_sum, dealer_card, usable_ace, temperature = game_state  # Unpack including temperature
        usable_ace = int(usable_ace)  # Convert boolean usable_ace to integer for indexing
        action = policy[player_sum - 1][dealer_card - 1][usable_ace][temperature]
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    return total_reward


def simulate_games(env, policy, num_games=1000):
    results = {'wins': 0, 'losses': 0, 'draws': 0}

    for _ in range(num_games):
        reward = play_game_with_policy(env, policy)
        if reward > 0:
            results['wins'] += 1
        elif reward < 0:
            results['losses'] += 1
        else:
            results['draws'] += 1

    return results

# Load the policy array
best_actions = np.load('learnedpolicy2.npy')

# Setup the environment
env = BlackjackEnv()

# Simulate the games
num_games = 10000000  # Number of games you want to simulate
results = simulate_games(env, best_actions, num_games)
print(f"Simulation results: {results}")

