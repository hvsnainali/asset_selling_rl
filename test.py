import numpy as np
from environments.CryptoEnvRL import CryptoEnvRL

# Example price series
price_series = np.linspace(100, 200, 50)  # Replace with real price data
env = CryptoEnvRL(price_series)

# Test environment
state = env.reset()
print("Initial State:", state)

done = False
while not done:
    action = env.action_space.sample()  # Random action
    next_state, reward, done, _ = env.step(action)
    print(f"Action: {action}, Reward: {reward}, Next State: {next_state}")
