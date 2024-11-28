from environments.CryptoEnvRL import CryptoEnvRL
from models.RLAgent import DQNAgent
import pandas as pd
import os

def load_data(folder="data", cryptocurrencies=["BTC-USD", "ETH-USD", "XRP-USD", "SOL-USD", "ADA-USD"]):
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
            # Ensure proper column parsing
            df = pd.read_csv(file_path, index_col="Date", parse_dates=["Date"])
            data[symbol] = df["Close"].values
        else:
            print(f"Data for {symbol} not found.")
    return data


if __name__ == "__main__":
    # Load cryptocurrency data
    cryptocurrencies = ["BTC-USD", "ETH-USD", "XRP-USD", "ADA-USD"]
    data = load_data(cryptocurrencies=cryptocurrencies)

    # Train RL agent on each cryptocurrency
    for symbol, price_series in data.items():
        print(f"Training RL agent on {symbol}...")

        # Initialize environment and agent
        env = CryptoEnvRL(price_series)
        agent = DQNAgent(state_size=7, action_size=3)  # 7 features, 3 actions (Buy, Hold, Sell)

        # Training parameters
        episodes = 500
        batch_size = 32
        epsilon = 1.0
        epsilon_decay = 0.995
        epsilon_min = 0.1

        # Training loop
        for e in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                action = agent.act(state, epsilon)
                next_state, reward, done, _ = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

            agent.replay(batch_size)
            agent.update_target_network()

            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward}")


for e in range(episodes):
    print(f"Episode {e + 1}/{episodes}")
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        # Select action
        action = agent.act(state)

        # Step through environment
        next_state, reward, done, _ = env.step(action)

        # Store experience
        agent.remember(state, action, reward, next_state, done)

        # Train the agent
        if len(agent.memory) > batch_size:
            loss = agent.replay(batch_size)
            print(f"Training Loss: {loss}")

        # Update state and total reward
        state = next_state
        total_reward += reward

    # Decay epsilon
    agent.decay_epsilon()

    # Log total reward for the episode
    print(f"Episode {e + 1}/{episodes}, Total Reward: {total_reward}")
