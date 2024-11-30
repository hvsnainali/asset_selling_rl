import os
import numpy as np
import pandas as pd
from environments.CryptoEnvRL import CryptoEnvRL
from models.RLAgent import DQNAgent

def load_data(folder="data", cryptocurrencies=["BTC-USD"]):
    """
    Load cryptocurrency data from CSV files.

    Args:
        folder (str): Directory containing the CSV files.
        cryptocurrencies (list): List of symbols to load.

    Returns:
        dict: Dictionary with cryptocurrency symbols as keys and price series as values.
    """
    data = {}
    for symbol in cryptocurrencies:
        file_path = os.path.join(folder, f"{symbol.replace('-', '_')}.csv")
        if os.path.exists(file_path):
            print(f"Loading data for {symbol}...")
            df = pd.read_csv(file_path, index_col="Date", parse_dates=["Date"])
            data[symbol] = df["Close"].values  # Use closing prices for simplicity
        else:
            print(f"Data for {symbol} not found in {folder}.")
    return data

if __name__ == "__main__":
    # Load cryptocurrency data
    cryptocurrencies = ["BTC-USD"]  # Modify this list to include other symbols
    data = load_data(cryptocurrencies=cryptocurrencies)

    # Training parameters
    episodes = 50
    batch_size = 32
    update_target_every = 10  # Update target network every 10 episodes

    # Train the RL agent for each cryptocurrency
    for symbol, price_series in data.items():
        print(f"\n--- Training RL agent on {symbol} ---")

        # Initialize environment and agent
        env = CryptoEnvRL(price_series)
        agent = DQNAgent(
            state_size=env.state_size,  # State dimension
            action_size=env.action_space.n,  # Number of actions
            epsilon=1.0,  # Initial exploration rate
            epsilon_min=0.1,  # Minimum exploration rate
            epsilon_decay=0.995,  # Decay factor for exploration rate
            learning_rate=0.001,  # Learning rate for the optimizer
            memory_size=2000  # Replay memory size
        )

        # Training loop
        for e in range(episodes):
            print(f"\nEpisode {e + 1}/{episodes}")
            state = env.reset()
            total_reward = 0
            done = False
            step = 0
            max_steps = len(price_series)  # Limit steps to the length of the series

            while not done and step < max_steps:
                # Select action using the agent's policy
                action = agent.act(state)

                # Step through the environment
                next_state, reward, done, _ = env.step(action)

                # Store the experience in replay memory
                agent.remember(state, action, reward, next_state, done)

                # Train the agent if there is enough data in replay memory
                if len(agent.memory) > batch_size:
                    loss = agent.replay(batch_size)
                    print(f"Step {step + 1}, Training Loss: {loss}")

                # Update state and accumulate reward
                state = next_state
                total_reward += reward
                step += 1

                print(f"Step {step}, Action: {action}, Reward: {reward}, Done: {done}")

            # Update the target network periodically
            if e % update_target_every == 0:
                agent.update_target_network()
                print(f"Target network updated at episode {e + 1}.")

            # Decay epsilon after each episode
            agent.decay_epsilon()

            # Log the episode result
            print(f"Episode {e + 1}/{episodes}, Total Reward: {total_reward}")

        print(f"Training complete for {symbol}!")

