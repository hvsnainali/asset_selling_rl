import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
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
            # Ensure proper column parsing
            df = pd.read_csv(file_path, index_col="Date", parse_dates=["Date"])
            data[symbol] = df["Close"].values
        else:
            print(f"Data for {symbol} not found.")
    return data

if __name__ == "__main__":
    # Load cryptocurrency data
    cryptocurrencies = ["BTC-USD"]
    data = load_data(cryptocurrencies=cryptocurrencies)

    # Training parameters
    episodes = 300
    batch_size = 128

    # Initialize lists for tracking metrics
    episode_rewards = []  # To store total rewards per episode
    training_losses = []  # To store average loss per episode
    action_counts = Counter()  # To track the frequency of actions taken

    

    for symbol, price_series in data.items():
        print(f"\n--- Training RL agent on {symbol} ---")

        # Initialize environment and agent
        env = CryptoEnvRL(price_series)
        agent = DQNAgent(
            state_size=7,  # 7 features in state
            action_size=3,  # 3 actions: Buy, Hold, Sell
            epsilon=1.0,  # Initial exploration rate
            epsilon_min=0.1,  # Minimum exploration rate
            epsilon_decay=0.995  # Decay factor for exploration rate
        )

        # Training loop
        for e in range(episodes):
            print(f"Episode {e + 1}/{episodes}")
            state = env.reset()
            total_reward = 0
            
            cumulative_profit = 0
            successful_trades = 0
            unsuccessful_trades = 0

            done = False
            step = 0
            max_steps = len(price_series)  # Limit to prevent infinite steps
            episode_loss = []  # Store losses for this episode

            while not done and step < max_steps:
                # Select action
                action = agent.act(state)
                action_counts[action] += 1  # Count the selected action

                # Step through environment
                next_state, reward, done, _ = env.step(action)

                # Accumulate rewards
                total_reward += reward
                if action == 2:  # Sell
                      if reward > 0:
                          successful_trades += 1  # Sell with profit
                      elif reward < 0:
                          unsuccessful_trades += 1  # Sell at a loss

                        # Track cumulative profit
                cumulative_profit += reward

                # Store experience in memory
                agent.remember(state, action, reward, next_state, done)

                # Train agent if enough memory
                if len(agent.memory) > batch_size:
                    loss = agent.replay(batch_size)  # Replay returns the loss
                    episode_loss.append(loss)  # Track step-level loss

                # Update state
                state = next_state
                step += 1

            # Average loss for the episode
            avg_loss = np.mean(episode_loss) if episode_loss else 0
            training_losses.append(avg_loss)  # Track average loss per episode

            # Decay epsilon after each episode
            agent.decay_epsilon()

            # Log episode result
            episode_rewards.append(total_reward)  # Track total reward per episode
            print(f"Episode {e + 1}/{episodes}, Total Reward: {total_reward}, Average Loss: {avg_loss},Profit: {cumulative_profit}, Successful Trades: {successful_trades}, " f"Unsuccessful Trades: {unsuccessful_trades}")

        print(f"Training complete for {symbol}!")

    # Action Distribution
    print("\nAction Distribution:")
    for action, count in action_counts.items():
        action_name = ["Buy", "Hold", "Sell"][action]
        print(f"{action_name}: {count} times")

    # Plot rewards per episode
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, episodes + 1), episode_rewards, label="Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Rewards per Episode")
    plt.legend()

    # Plot training loss per episode
    plt.subplot(1, 2, 2)
    plt.plot(range(1, episodes + 1), training_losses, label="Loss", color="red")
    plt.xlabel("Episode")
    plt.ylabel("Average Training Loss")
    plt.title("Training Loss per Episode")
    plt.legend()

    plt.tight_layout()
    plt.show()

