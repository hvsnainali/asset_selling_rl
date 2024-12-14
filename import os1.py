import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from environments.CryptoEnvRL import CryptoEnvRL
from models.RLAgent import DQNAgent
from utils import evaluate_agent


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
    
            data[symbol] = df[["Close", "Volume"]]
        else:
            print(f"Data for {symbol} not found.")
    return data

if __name__ == "__main__":
    # Load cryptocurrency data
    cryptocurrencies = ["BTC-USD"]
    data = load_data(cryptocurrencies=cryptocurrencies)

    # Training parameters
    episodes = 425
    batch_size = 256

    # Initialize lists for tracking metrics
    episode_rewards = []  # To store total rewards per episode
    training_losses = []  # To store average loss per episode
    action_counts = Counter()  # To track the frequency of actions taken

    

    for symbol, df in data.items():
        print(f"\n--- Training RL agent on {symbol} ---")

        price_series = df["Close"].values  # Close prices
        volume_series = df["Volume"].values  # Volume data

        # Initialize environment and agent
        env = CryptoEnvRL(price_series, volume_series)
        agent = DQNAgent(
            state_size=9,  # 9 features in state
            action_size=3,  # 3 actions: Buy, Hold, Sell
            epsilon=1.0,  # Initial exploration rate
            epsilon_min=0.1,  # Minimum exploration rate
            epsilon_decay=0.998  # Decay factor for exploration rate
        )

        # Training loop
        for e in range(episodes):
            print(f"Episode {e + 1}/{episodes}")
            state = env.reset()
            total_reward = 0
            
            cumulative_profit = 0
            successful_trades = 0
            unsuccessful_trades = 0
            shares_owned = 0

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

                # Track the share owned...
                if env.stock_owned:
                    shares_owned += 1

                # Accumulate rewards...
                total_reward += reward
                if action == 2 :  # Sell
                      if reward > 0:
                          successful_trades += 1  # Sell with profit
                      elif reward < 0:
                          unsuccessful_trades += 1  # Sell at a loss

                if action == 2 and env.stock_owned == 0:  # After a sell action
                   profit = env.price_series[env.t] - env.buy_price
                   cumulative_profit += profit

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
            print(f"Episode {e + 1}/{episodes}, Epsilon: {agent.epsilon:.4f}, Total Reward: {total_reward}, "
                    f"Average Loss: {avg_loss}, Profit: {cumulative_profit:.4f},"
                    f"Shares Owned: {shares_owned}, "
                    f"Successful Trades: {successful_trades}, "
                    f"Unsuccessful Trades: {unsuccessful_trades}")

        print(f"Training complete for {symbol}!")

        plr = successful_trades / (successful_trades + unsuccessful_trades)
        print(f"Episode {e}/{episodes}, Profit-Loss Ratio (PLR): {plr:.2f}")

    portfolio_value = (env.stock_owned * env.price_series[env.t]) + cumulative_profit
    print(f"Portfolio Value at Episode End: {portfolio_value}")

    # Action Distribution
    print("\nAction Distribution:")
    for action, count in action_counts.items():
        action_name = ["Buy", "Hold", "Sell"][action]
        print(f"{action_name}: {count} times")

    action_history = []
    while not done:
        action = agent.act(state)
        action_history.append(action)
        ...
    print(f"Action History: {action_history}")

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

