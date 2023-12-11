import numpy as np

# Load the policy array
best_actions = np.load('learnedpolicy2.npy')
print(best_actions[20, 7, 1])