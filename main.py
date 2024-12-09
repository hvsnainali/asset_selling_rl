import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from environments.CryptoEnvRL import CryptoEnvRL
from models.RLAgent import DQNAgent
from utils import evaluate_agent


def load_data(folder="data", cryptocurrencies=["BTC-USD", "ETH-USD"]):
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
            df = pd.read_csv(file_path, index_col="Date", parse_dates=["Date"])
    
            #data[symbol] = df[["Close", "Volume"]]
            train_data = df.loc["2017-01-01":"2019-12-31"]
            test_data = df.loc["2020-01-01":"2024-01-01"]
            data[symbol] = (train_data, test_data)
        else:
            print(f"Data for {symbol} not found.")
    return data

if __name__ == "__main__":
    # Load cryptocurrency data
    cryptocurrencies = ["BTC-USD", "ETH-USD"]
    data = load_data(cryptocurrencies=cryptocurrencies)

    # Training parameters
    episodes = 300
    batch_size = 512

    # Initialize lists for tracking metrics
    episode_rewards = []  # To store total rewards per episode
    training_losses = []  # To store average loss per episode
    action_counts = Counter()  # To track the frequency of actions taken

    

    for symbol, (train_data, test_data)  in data.items():
        print(f"\n--- Training RL agent on {symbol} ---")

        #price_series = df["Close"].values  # Close prices
        #volume_series = df["Volume"].values  # Volume data

        # Split into train and test sets
        price_series_train = train_data["Close"].values
        volume_series_train = train_data["Volume"].values
        price_series_test = test_data["Close"].values
        volume_series_test = test_data["Volume"].values

        if symbol == "ETH-USD":
           
            #price_series_train = (price_series_train - np.mean(price_series_train)) / np.std(price_series_train)
            volume_series_train = (volume_series_train - np.mean(volume_series_train)) / np.std(volume_series_train)

            #price_series_test = (price_series_test - np.mean(price_series_test)) / np.std(price_series_test)
            volume_series_test = (volume_series_test - np.mean(volume_series_test)) / np.std(volume_series_test)
            print("Normalisation for ETH-USD...")

        # Initialize environment and agent
        env = CryptoEnvRL(price_series_train, volume_series_train)
        agent = DQNAgent(
            state_size=10,  # 9 features in state
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
            shares_owned = 0

            done = False
            step = 0
            max_steps = len(price_series_train)  # Limit to prevent infinite steps
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
                if action == 2:  # Sell
                      if reward > 0:
                          successful_trades += 1  # Sell with profit
                      elif reward < 0:
                          unsuccessful_trades += 1  # Sell at a loss

                        # Track cumulative profit
                cumulative_profit += reward

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
                    f"Average Loss: {avg_loss}, Profit: {cumulative_profit:.2f},"
                    f"Shares Owned: {shares_owned}, "
                    f"Successful Trades: {successful_trades}, "
                    f"Unsuccessful Trades: {unsuccessful_trades}")

        print(f"Training complete for {symbol}!")

        plr = successful_trades / (successful_trades + unsuccessful_trades)
        print(f"Episode {e}/{episodes}, Profit-Loss Ratio (PLR): {plr:.2f}")

          # Evaluation on test data
        env_test = CryptoEnvRL(price_series_test, volume_series_test)
        print(f"\n--- Evaluating RL agent on test data for {symbol} ---")
       # evaluate_agent(env_test, agent, episodes=10,export_csv=True)

    portfolio_value = (env.stock_owned * env.price_series[env.t]) + cumulative_profit
    print(f"Portfolio Value at Episode End: {portfolio_value}")

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
