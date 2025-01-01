import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from matplotlib.animation import FuncAnimation
from sklearn.model_selection import TimeSeriesSplit
from environments.CryptoEnvRL import CryptoEnvRL
from models.RLAgent import DQNAgent
from collections import Counter
from test import run_tests
from test import evaluate_agent

from test import (
                analyze_rolling_std,
                evaluate_agent_cumulative, 
                evaluate_heuristic_sdp_cumulative,
                evaluate_simple_policy_cumulative,
                plot_three_strategies,
                run_tests_with_separate_plots
            )

from environments.CryptoSDPModel import CryptoSDPModel, advanced_heuristic_policy_sdp

import warnings
warnings.filterwarnings("ignore")
import torch

# def load_data(folder="data", cryptocurrencies=["BTC-USD", "ETH-USD"]):
#     """
#     Load cryptocurrency data from CSV files.

#     Args:
#         folder (str): Directory containing the CSV files.
#         cryptocurrencies (list): List of symbols to load.

#     Returns:
#         dict: Dictionary with cryptocurrency symbols as keys and price series as values.
#     """
#     data = {}
#     for symbol in cryptocurrencies:
#         file_path = os.path.join(folder, f"{symbol.replace('-', '_')}.csv")
#         if os.path.exists(file_path):
#             df = pd.read_csv(file_path, index_col="Date", parse_dates=["Date"])
    
#             #data[symbol] = df[["Close", "Volume"]]
#             train_data = df.loc["2017-01-01":"2022-10-31"]
#             test_data = df.loc["2022-11-01":"2024-01-01"]
#             data[symbol] = (train_data, test_data)
#         else:
#             print(f"Data for {symbol} not found.")
#     return data


# if __name__ == "__main__":
#     # Load cryptocurrency data
#     cryptocurrencies = ["BTC-USD", "ETH-USD"]
#     data = load_data(cryptocurrencies=cryptocurrencies)

def load_data(folder, cryptocurrencies):
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
            train_data = df.loc["2017-01-01":"2022-12-31"]
            test_data = df.loc["2023-01-01":"2024-01-01"]
            data[symbol] = (train_data, test_data)
        else:
            print(f"Data for {symbol} not found.")
    return data



if __name__ == "__main__":
    # Load cryptocurrency data
    folder = "data"
    cryptocurrencies = ["BTC-USD", "ETH-USD"]
    data = load_data(folder, cryptocurrencies)

    # Training parameters
    episodes = 10
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
        date_series_train = train_data.index


        price_series_test = test_data["Close"].values
        volume_series_test = test_data["Volume"].values
        date_series_test = test_data.index


        # Initialize environment and agent
        print(f"Training data: {len(price_series_train)} entries")
        print(f"Test data: {len(price_series_test)} entries from {date_series_test[0]} to {date_series_test[-1]}")


        # ------------------------------------------------
        # Initialize environment and DQN agent
        # ------------------------------------------------
        env = CryptoEnvRL(price_series_train, volume_series_train)
        agent = DQNAgent(
            state_size=10,   # 10 features in state
            action_size=3,   # Buy, Hold, Sell
            epsilon=1.0,     # Starting exploration rate
            epsilon_min=0.1, # Minimum exploration rate
            epsilon_decay=0.998
        )

        # A second agent to load final weights for testing
        test_env = CryptoEnvRL(price_series_test, volume_series_test)
        trained_agent = DQNAgent(
            state_size=10,
            action_size=3
        ) 

        returns = []
        print("Model weights before training:")
        print(agent.model.state_dict())

        # ------------------------------------------------
        # Training loop
        # ------------------------------------------------
        for e in range(episodes):
            print(f"Episode {e + 1}/{episodes}")
            agent.model.train()
            state = env.reset()
            total_reward = 0
            
            # Some metrics
            cumulative_profit = 0
            successful_trades = 0
            unsuccessful_trades = 0
            shares_owned = 0

            done = False
            step = 0
            max_steps = len(price_series_train)
            episode_loss = []

            # Debug Q-values in the first episode for the first 20 steps
            if e == 0:
                print("Debugging Q-values during training episode 0 for the first 20 steps.")
                for dbg_step in range(20):
                    with torch.no_grad():
                        q_vals = agent.model(torch.FloatTensor(state).unsqueeze(0).to(agent.device))
                    dbg_action = np.argmax(q_vals.cpu().numpy())
                    print(f"Step {dbg_step}, Q-vals={q_vals.cpu().numpy()}, Action={dbg_action}")

            # Step through environment
            while not done and step < max_steps:
                action = agent.act(state)   # Epsilon-greedy
                action_counts[action] += 1

                next_state, reward, done, _ = env.step(action)
                total_reward += reward

                # Track trades
                if env.stock_owned:
                    shares_owned += 1

                if action == 2:  # Sell
                    if reward > 0:
                        successful_trades += 1
                    elif reward < 0:
                        unsuccessful_trades += 1


                
                # Store experience and train
                agent.remember(state, action, reward, next_state, done)
                if len(agent.memory) > batch_size:
                    loss = agent.replay(batch_size)
                    episode_loss.append(loss)

                state = next_state
                step += 1

            # After each episode
            avg_loss = np.mean(episode_loss) if episode_loss else 0
            training_losses.append(avg_loss)

            agent.decay_epsilon()
            episode_rewards.append(total_reward)

            print(f"Episode {e + 1}/{episodes}, Epsilon: {agent.epsilon:.4f}, Total Reward: {total_reward:.0f}, "
                  f"Average Loss: {avg_loss:.0f}, Profit: {cumulative_profit:.2f}, "
                  f"Shares Owned: {shares_owned}, "
                  f"Successful Trades: {successful_trades}, "
                  f"Unsuccessful Trades: {unsuccessful_trades}")
            
        # sharpe_ratio = sharpe_ratio(returns)
        # print(f"Sharpe Ratio for {symbol}: {sharpe_ratio:.4f}")
            # Save trained model and load into 'trained_agent'
            agent.save("trained_agent.pth")
            trained_agent.load("trained_agent.pth")
            trained_agent.epsilon = 0.0  # pure exploitation at test

        print(f"Training complete for {symbol}!")
        print("Model weights after training:")
        print(agent.model.state_dict())

        # # Save trained model and load into 'trained_agent'
        # agent.save("trained_agent.pth")
        # trained_agent.load("trained_agent.pth")
        # trained_agent.epsilon = 0.0  # pure exploitation at test

        # ------------------------------------------------
        # Evaluate DQN on test set
        # ------------------------------------------------
        # from test import evaluate_agent
        # dqn_total_reward, dqn_actions = evaluate_agent(test_env, trained_agent, print_qvals=True)
        # print("DQN Total Reward on Test Environment:", dqn_total_reward)
        # print("DQN Actions Distribution:", Counter(dqn_actions))

        # Profit-Loss Ratio on the final training environment (optional)
        plr = successful_trades / (successful_trades + unsuccessful_trades) if (successful_trades+unsuccessful_trades)>0 else 0
        print(f"Final PLR (Train) => {plr:.2f}")

        portfolio_value = (env.stock_owned * env.price_series[env.t]) + cumulative_profit
        print(f"Portfolio Value at Episode End: {portfolio_value:.2f}")

        print("\nAction Distribution (Train Loop Overall):")
        for action, count in action_counts.items():
            action_name = ["Buy", "Hold", "Sell"][action]
            print(f"{action_name}: {count} times")

        # Optionally, run your test harness
        from test import run_tests
        run_tests(test_env, trained_agent, date_series_test)
        
        analyze_rolling_std(price_series_test)
        # Example: Test thresholds

        # ------------------------------------------------
        # Evaluate the SDP Heuristic on the Same Test Data
        # ------------------------------------------------
        # print("\n--- Running SDP Heuristic on Test Data ---")
        # sdp_model = CryptoSDPModel(price_series_test, volume_series_test)
        # sdp_total_reward, sdp_actions, sdp_rewards = advanced_heuristic_policy_sdp(sdp_model)
        # print(f"[SDP] Heuristic Total Reward on Test: {sdp_total_reward:.2f}")
        # print("SDP Actions Distribution:", Counter(sdp_actions))

        # ------------------------------------------------
        # Compare DQN vs. SDP actions visually
        # ------------------------------------------------
        # def compare_actions(price_data, actions_dqn, label_dqn, actions_sdp, label_sdp):
        #     plt.figure(figsize=(12, 6))
        #     plt.plot(price_data, color='gray', label='Price')

        #     buy_dqn = [i for i, a in enumerate(actions_dqn) if a == 0]
        #     sell_dqn = [i for i, a in enumerate(actions_dqn) if a == 2]
        #     buy_sdp = [i for i, a in enumerate(actions_sdp) if a == 0]
        #     sell_sdp = [i for i, a in enumerate(actions_sdp) if a == 2]

        #     # DQN
        #     plt.scatter(buy_dqn, [price_data[i] for i in buy_dqn],
        #                 color='green', marker='^', label=f'{label_dqn} Buy')
        #     plt.scatter(sell_dqn, [price_data[i] for i in sell_dqn],
        #                 color='red', marker='v', label=f'{label_dqn} Sell')

        #     # SDP
        #     plt.scatter(buy_sdp, [price_data[i] for i in buy_sdp],
        #                 color='lime', marker='^', facecolors='none', label=f'{label_sdp} Buy')
        #     plt.scatter(sell_sdp, [price_data[i] for i in sell_sdp],
        #                 color='orange', marker='v', facecolors='none', label=f'{label_sdp} Sell')

        #     plt.title(f"Comparison: {label_dqn} vs {label_sdp} (Test Data)")
        #     plt.legend()
        #     plt.show()

        # # If you want to see the difference visually:
        # compare_actions(price_series_test, dqn_actions, "DQN", sdp_actions, "SDP Heuristic")

         # ------------------------------------------------
            # If you want step-by-step date-based cumulative lines:
            # (Only if you use the same dates from date_series_test)
            # ------------------------------------------------
            
            # Evaluate DQN
        dqn_dates, dqn_cum_profits, dqn_actions = evaluate_agent_cumulative(test_env, trained_agent, date_series_test)
            
            # Evaluate SDP
        # sdp_env = CryptoEnvRL(price_series_test, volume_series_test)
        # sdp_dates, sdp_cumprofits, sdp_step_actions = evaluate_sdp_cumulative(sdp_env, date_series_test)

        sdp_model = CryptoSDPModel(price_series_test, volume_series_test)
        sdp_dates, sdp_cum_profits, sdp_actions = evaluate_heuristic_sdp_cumulative(sdp_model, date_series_test)
            
            # Evaluate Simple
        simp_env = CryptoEnvRL(price_series_test, volume_series_test)
        simp_dates, simp_cumprofits, simp_step_actions = evaluate_simple_policy_cumulative(test_env, date_series_test)
            
            # Plot
        plot_three_strategies(dqn_dates, dqn_cum_profits, sdp_cum_profits, simp_cumprofits)

        def compare_actions(price_data, actions_dqn, label_dqn, actions_sdp, label_sdp):
            plt.figure(figsize=(12, 6))
            plt.plot(price_data, color='gray', label='Price')

            buy_dqn = [i for i, a in enumerate(actions_dqn) if a == 0]
            sell_dqn = [i for i, a in enumerate(actions_dqn) if a == 2]
            buy_sdp = [i for i, a in enumerate(actions_sdp) if a == 0]
            sell_sdp = [i for i, a in enumerate(actions_sdp) if a == 2]

            # DQN
            plt.scatter(buy_dqn, [price_data[i] for i in buy_dqn],
                        color='green', marker='^', label=f'{label_dqn} Buy')
            plt.scatter(sell_dqn, [price_data[i] for i in sell_dqn],
                        color='red', marker='v', label=f'{label_dqn} Sell')

            # SDP
            plt.scatter(buy_sdp, [price_data[i] for i in buy_sdp],
                        color='lime', marker='^', facecolors='none', label=f'{label_sdp} Buy')
            plt.scatter(sell_sdp, [price_data[i] for i in sell_sdp],
                        color='orange', marker='v', facecolors='none', label=f'{label_sdp} Sell')

            plt.title(f"Comparison: {label_dqn} vs {label_sdp} (Test Data)")
            plt.legend()
            plt.show()

        # If you want to see the difference visually:
        compare_actions(price_series_test, dqn_actions, "DQN", sdp_actions, "SDP Heuristic")
    
    run_tests_with_separate_plots(test_env, trained_agent, date_series_test)

    # ------------------------------------------------
    # Plot the training rewards/losses as an animation
    # ------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    def update(frame):
        ax1.clear()
        ax2.clear()

        # Rewards
        ax1.plot(episode_rewards[:frame + 1], label='Total Rewards per Episode')
        ax1.set_title('Rewards per Episode')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.legend(loc='upper left')

        # Losses
        ax2.plot(training_losses[:frame + 1], label='Average Loss per Episode', color='red')
        ax2.set_title('Training Loss per Episode')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        ax2.legend(loc='upper left')

        plt.tight_layout()

    ani = FuncAnimation(fig, update, frames=episodes, interval=250, repeat=False)
    plt.show()