import gym
import numpy as np
import random
from math import sqrt, log

class MCTSAgent:
    def __init__(self, num_iterations=1000, exploration_constant=1.41):
        self.num_iterations = num_iterations
        self.exploration_constant = exploration_constant

    def select_action(self, env):
        initial_observation = env.reset()
        root = Node(state=GameState(initial_observation, env), parent=None, untried_actions=list(range(env.action_space.n)))
        for _ in range(self.num_iterations):
            node = self.select_node(root)
            reward = self.rollout(env, node.state)
            self.backpropagate(node, reward)
        return self.best_action(root)

    def select_node(self, node):
        while not node.is_terminal():
            if node.is_fully_expanded():
                node = node.best_child(self.exploration_constant)
            else:
                return node.expand()
        return node

    def rollout(self, env, state):
        while not state.is_terminal():
            action = random.choice(state.available_actions())
            state = state.take_action(action)
        return state.get_reward()

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def best_action(self, node):
        return max(node.children, key=lambda n: n.value / n.visits if n.visits > 0 else 0).action

class Node:
    def __init__(self, state, parent, untried_actions, action=None):
        self.state = state
        self.parent = parent
        self.untried_actions = untried_actions
        self.action = action
        self.children = []
        self.value = 0
        self.visits = 0

    def expand(self):
        action = self.untried_actions.pop()
        next_state = self.state.take_action(action)
        child_node = Node(state=next_state, parent=self, untried_actions=next_state.available_actions(), action=action)
        self.children.append(child_node)
        return child_node

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, exploration_constant):
        best_score = float('-inf')
        best_children = []
        for child in self.children:
            exploit = child.value / child.visits
            explore = sqrt(2 * log(self.visits) / child.visits)
            score = exploit + exploration_constant * explore
            if score == best_score:
                best_children.append(child)
            elif score > best_score:
                best_children = [child]
                best_score = score
        return random.choice(best_children)

    def is_terminal(self):
        return self.state.is_terminal()

class GameState:
    def __init__(self, observation, env, done=False, reward=0):
        self.observation = observation
        self.env = env
        self.done = done
        self.reward = reward

    def take_action(self, action):
        results = self.env.step(action)
        observation, reward, done = results[:3]
        return GameState(observation, self.env, done, reward)

    def available_actions(self):
        return [0, 1] if not self.done else []

    def is_terminal(self):
        return self.done

    def get_reward(self):
        return self.reward

def run_mcts_on_state(env, agent, state):
    env.reset()
    env.s = state
    return agent.select_action(env)

def main():
    env = gym.make('Blackjack-v1')
    agent = MCTSAgent()

    best_actions = np.zeros((32, 11, 2), dtype=int)

    for player_sum in range(1, 32):
        for dealer_card in range(1, 11):
            for usable_ace in (0, 1):
                state = (player_sum, dealer_card, usable_ace)
                
                best_action = run_mcts_on_state(env, agent, state)
                best_actions[player_sum - 1][dealer_card - 1][usable_ace] = best_action

    np.save('best_actions.npy', best_actions)
    print("Best Actions Array:")
    print(best_actions)

if __name__ == "__main__":
    main()
