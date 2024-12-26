import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from environments.CryptoEnvRL import CryptoEnvRL
from models.RLAgent import DQNAgent

from test import evaluate_agent_deterministic

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


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
            train_data = df.loc["2017-01-01":"2022-10-31"]
            test_data = df.loc["2022-11-01":"2024-01-01"]
            data[symbol] = (train_data, test_data)
        else:
            print(f"Data for {symbol} not found.")
    return data


if __name__ == "__main__":
    # Load cryptocurrency data
    cryptocurrencies = ["BTC-USD", "ETH-USD"]
    data = load_data(cryptocurrencies=cryptocurrencies)

    # Training parameters
    episodes = 4
    batch_size = 128

    # Initialize lists for tracking metrics
    episode_rewards = []  # To store total rewards per episode
    training_losses = []  # To store average loss per episode
    action_counts = Counter()  # To track the frequency of actions taken

    

    for symbol, (train_data, test_data)  in data.items():
        print(f"\n--- Training RL agent on {symbol} ---")

        # Split into train and test sets
        # if symbol == "ETH-USD":
        # # Normalize only the ETH data
        #     price_series_train = (train_data["Close"].values - np.mean(train_data["Close"].values)) / (np.std(train_data["Close"].values) + 1e-8)
        #     volume_series_train = (train_data["Volume"].values - np.mean(train_data["Volume"].values)) / (np.std(train_data["Volume"].values) + 1e-8)
        #     price_series_test = (test_data["Close"].values - np.mean(test_data["Close"].values)) / (np.std(test_data["Close"].values) + 1e-9)
        #     volume_series_test = (test_data["Volume"].values - np.mean(test_data["Volume"].values)) / (np.std(test_data["Volume"].values) + 1e-8)
        # else:
        # # Do not normalize other data

        price_series_train = train_data["Close"].values
        volume_series_train = train_data["Volume"].values
        price_series_test = test_data["Close"].values
        volume_series_test = test_data["Volume"].values


        # Initialize environment and agent
        env = CryptoEnvRL(price_series_train, volume_series_train)
        agent = DQNAgent(
            state_size=10,  # 10 features in state
            action_size=3,  # 3 actions: Buy, Hold, Sell
            epsilon=1.0,  # Initial exploration rate
            epsilon_min=0.1,  # Minimum exploration rate
            epsilon_decay=0.998  # Decay factor for exploration rate
        )

        test_env = CryptoEnvRL(price_series_test,volume_series_test)
        trained_agent = DQNAgent(
            state_size=10,
            action_size=3
        ) 

        # Training loop
        for e in range(episodes):
            print(f"Episode {e + 1}/{episodes}")
            state = env.reset()
            agent.model.train()
            total_reward = 0
            
            cumulative_profit = 0
            successful_trades = 0
            unsuccessful_trades = 0
            shares_owned = 0

            done = False
            step = 0
            max_steps = len(price_series_train)  
            episode_loss = []  

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
    #     env_test = CryptoEnvRL(price_series_test, volume_series_test)
    #     print(f"\n--- Evaluating RL agent on test data for {symbol} ---")
    #    # evaluate_agent(env_test, agent, episodes=10,export_csv=True)

    portfolio_value = (env.stock_owned * env.price_series[env.t]) + cumulative_profit
    print(f"Portfolio Value at Episode End: {portfolio_value}")

    # Action Distribution
    print("\nAction Distribution:")
    for action, count in action_counts.items():
        action_name = ["Buy", "Hold", "Sell"][action]
        print(f"{action_name}: {count} times")


    agent.model.eval()    
    total_reward, actions_taken = evaluate_agent_deterministic(test_env, trained_agent)


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    def update(frame):
        ax1.clear()
        ax2.clear()

        # Plotting the rewards
        ax1.plot(episode_rewards[:frame + 1], label='Total Rewards per Episode')
        ax1.set_title('Rewards per Episode')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.legend(loc='upper left')

        # Plotting the losses
        ax2.plot(training_losses[:frame + 1], label='Average Loss per Episode', color='red')       
        ax2.set_title('Training Loss per Episode')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Average Loss')
        ax2.legend(loc='upper left')

        plt.tight_layout()

    ani = FuncAnimation(fig, update, frames=episodes, interval=250, repeat = False)
    plt.show()

    

    # actions = ['Buy', 'Hold', 'Sell']

    # # Simulated data collection
    # action_counts = [Counter() for _ in range(episodes)]

    # # Simulate action counting per episode
    # for episode in range(episodes):
    #     action_counts[episode].update({0: np.random.randint(0, 10), 1: np.random.randint(0, 10), 2: np.random.randint(0, 10)})

    # # Plot setup
    # fig, ax = plt.subplots()
    # x = np.arange(len(actions))  # the label locations
    # bars = ax.bar(x, [0, 0, 0], color=['green', 'blue', 'red'])

    # ax.set_xticks(x)
    # ax.set_xticklabels(actions)
    # ax.set_ylim(0, 50)  # You may need to adjust this based on expected counts
    # ax.set_title('Action Distribution Over Episodes')
    # ax.set_ylabel('Counts')
    # ax.set_xlabel('Actions')

    # def update(frame):
    #     for i, bar in enumerate(bars):
    #         bar.set_height(action_counts[frame][i])
    #     ax.set_title(f'Episode {frame + 1}')
    #     return bars

    # ani = FuncAnimation(fig, update, frames=episodes, repeat=False)
    # plt.show()
 
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.plot(range(1, episodes + 1), episode_rewards, label="Rewards")
    # plt.xlabel("Episode")
    # plt.ylabel("Total Reward")
    # plt.title("Rewards per Episode")
    # plt.legend()


    # # Plot training loss per episode
    # plt.subplot(1, 2, 2)
    # plt.plot(range(1, episodes + 1), training_losses, label="Loss", color="red")
    # plt.xlabel("Episode")
    # plt.ylabel("Average Training Loss")
    # plt.title("Training Loss per Episode")
    # plt.legend()

    # plt.tight_layout()
    # plt.show()
