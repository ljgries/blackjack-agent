import gym
import numpy as np

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
        # Handle the observation based on whether it's nested or not
        if isinstance(observation, tuple) and isinstance(observation[0], tuple):
            game_state, _ = observation  # Unpack nested tuple
        else:
            game_state = observation   # Directly use the observation as the game state

        player_sum, dealer_card, usable_ace = game_state

        # Convert boolean usable_ace to integer for indexing
        usable_ace = int(usable_ace)

        # Get the action from the policy
        action = policy[player_sum - 1][dealer_card - 1][usable_ace]

        # Step the environment
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        # Use either terminated or truncated to determine if the game is done
        done = terminated or truncated

    return total_reward







def simulate_games(env, policy, num_games=100000):
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
best_actions = np.load('learnedpolicy.npy')

# Setup the environment
env = gym.make('Blackjack-v1')

# Simulate the games
results = simulate_games(env, best_actions, 100000)
print(f"Simulation results: {results}")
