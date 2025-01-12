import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from matplotlib.animation import FuncAnimation
from sklearn.model_selection import TimeSeriesSplit
from environments.CryptoEnvRL import CryptoEnvRL
from models.RLAgent import DQNAgent
from collections import Counter, deque

from test import (
                run_tests,
                calculate_sharpe_ratio,
                calculate_sortino_ratio,
                calculate_calmar_ratio,
                calculate_max_drawdown, 
                calculate_trade_stats,
                plot_equity_vs_buyhold,
                plot_drawdown,
                plot_return_distribution,
                analyze_rolling_std,
                evaluate_agent_cumulative, 
                evaluate_heuristic_sdp_cumulative,
                evaluate_simple_policy_cumulative,
                plot_three_strategies,
                plot_trade_distribution,
                plot_portfolio_value,
                evaluate_portfolio_on_training_data,
                evaluate_agent_cumulative_with_portfolio,
                stability_of_returns
            )

from environments.CryptoSDPModel import CryptoSDPModel, advanced_heuristic_policy_sdp

import warnings
warnings.filterwarnings("ignore")
import torch

def load_data(folder, cryptocurrencies):
    """
    Load cryptocurrency data from CSV files.

    Args:
        folder (str): Directory containing the CSV files. 
        cryptocurrencies (list): List of symbols to load.

    Returns:
        Data
    """
    data = {}
    for symbol in cryptocurrencies:
        file_path = os.path.join(folder, f"{symbol.replace('-', '_')}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col="Date", parse_dates=["Date"])
            df.sort_index(inplace=True)

            train_data = df.loc["2017-01-01":"2022-06-23"]
            test_data = df.loc["2022-06-24":"2024-01-01"]
            data[symbol] = (train_data, test_data)
        else:
            print(f"Data for {symbol} not found in {folder}.")
    return data

if __name__ == "__main__":
    folder = "data"
    cryptocurrencies = ["BTC-USD","ETH-USD"]
    data = load_data(folder, cryptocurrencies)

    episodes = 5
    batch_size = 64

    episode_rewards = []    # total rewards per episode
    training_losses = []    # average training loss per episode
    action_counts = Counter()

    # Loop over each symbol's train and test sets
    for symbol, (train_data, test_data) in data.items():
        print(f"\n--- Training RL agent on {symbol} ---")

        price_series_train = train_data["Close"].values
        volume_series_train = train_data["Volume"].values
        date_series_train = train_data.index

        price_series_test = test_data["Close"].values
        volume_series_test = test_data["Volume"].values
        date_series_test = test_data.index

        print(f"Training data: {len(price_series_train)} entries")
        print(f"Test data: {len(price_series_test)} entries from {date_series_test[0]} to {date_series_test[-1]}")

        # ------------------------------------------------
        # Initialize environment (11-feature state) and DQN agent
        # ------------------------------------------------
        env = CryptoEnvRL(
            price_series_train, 
            volume_series_train,  
            initial_cash=30000.0  
        )

        agent = DQNAgent(
            state_size=11,   # 11 features in the new environment
            action_size=3,   # Buy, Hold, Sell
            epsilon=1.0,     # Start exploration
            epsilon_min=0.1,
            epsilon_decay=0.998
        )

        # For plotting or final training check
        train_env_for_plot = env

        # Prepare a test environment
        test_env = CryptoEnvRL(
            price_series_test, 
            volume_series_test,
            initial_cash=30000.0
        )
        trained_agent = DQNAgent(
            state_size=11,
            action_size=3
        ) 
        
        returns = []
        reward_window = deque(maxlen=5)

        print("Model weights before training:")
        print(agent.model.state_dict())

        # ------------------------------------------------
        # Training loop
        # ------------------------------------------------
        for e in range(episodes):
            print(f"Episode {e+1}/{episodes}")
            agent.model.train()

            state = env.reset()
            total_reward = 0.0

            done = False
            step = 0
            max_steps = len(price_series_train)
            episode_loss = []
            episode_portfolio_values = []

            successful_trades = 0
            unsuccessful_trades = 0

            # Debug Q-values in the first 20 Steps
            if e == 0:
                print("Debugging Q-values for first 20 steps:")
                for dbg_step in range(20):
                    with torch.no_grad():
                        q_vals = agent.model(
                            torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                        )
                    dbg_action = np.argmax(q_vals.cpu().numpy())
                    print(f" Step {dbg_step}, Q-vals={q_vals.cpu().numpy()}, Action={dbg_action}")

            while not done and step < max_steps:
                action = agent.act(state)  # Epsilon-greedy
                action_counts[action] += 1

                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                returns.append(reward)


                if action == 2:  # Sell
                    if reward > 0:
                        successful_trades += 1
                    elif reward < 0:
                        unsuccessful_trades += 1

                # Track step-level portfolio value (from env) if you like
                episode_portfolio_values.append(env.portfolio_value)

                # Memory + replay
                agent.remember(state, action, reward, next_state, done)
                if len(agent.memory) > batch_size:
                    loss_val = agent.replay(batch_size)
                    if loss_val is not None:
                        episode_loss.append(loss_val)

                state = next_state
                step += 1

            # End of episode stats
            avg_loss = np.mean(episode_loss) if episode_loss else 0
            training_losses.append(avg_loss)
            agent.decay_epsilon()

            episode_rewards.append(total_reward)
            reward_window.append(total_reward)
            learning_metric = np.mean(reward_window)

            final_portfolio_value = episode_portfolio_values[-1] if episode_portfolio_values else 0
            print(
                f"Episode {e+1}/{episodes}, "
                f"Epsilon: {agent.epsilon:.4f}, "
                f"Total Reward: {total_reward:.2f}, "
                f"Average Loss: {avg_loss:.2f}, "
                f"Final Portfolio: {final_portfolio_value:.2f}, "
                f"Learning Metric: {learning_metric:.2f}, "
                f"Successful Trades: {successful_trades}, "
                f"Unsuccessful Trades: {unsuccessful_trades}"
            )

        # Save and load agent
        agent.save("trained_agent.pth")
        trained_agent.load("trained_agent.pth")
        trained_agent.epsilon = 0.0  # pure exploitation

        print(f"Training complete for {symbol}!")
        print("Model weights after training:")
        print(agent.model.state_dict())

        plr = successful_trades / (successful_trades + unsuccessful_trades) if (successful_trades+unsuccessful_trades)>0 else 0
        print(f"Final PLR (Train) => {plr:.2f}")

        # Sharpe ratio
        sharpe_ratio = calculate_sharpe_ratio(returns)
        print(f"Sharpe Ratio for {symbol}: {sharpe_ratio:.4f}")

        solino_ratio = calculate_sortino_ratio(returns)
        print(f"Solino Ratio for {symbol}: {solino_ratio:.4f}")

        vol = np.std(episode_portfolio_values)

        print("\nAction Distribution (Train Loop Overall):")
        for act, count in action_counts.items():
            name = ["Buy", "Hold", "Sell"][act]
            print(f"{name}: {count} times")

        # ------------------------------------------------
        # Evaluate the trained DQN on the test set
        # ------------------------------------------------
      
        run_tests(test_env, trained_agent, date_series_test)

        analyze_rolling_std(price_series_test)

        # Evaluate DQN step by step
        dqn_dates, dqn_cum_profits, dqn_actions, dqn_plr,dqn_returns = evaluate_agent_cumulative(
            test_env, trained_agent, date_series_test
        )
        dqn_final = dqn_cum_profits[-1] if dqn_cum_profits else 0

        #print(f"[DEBUG] DQN Returns: {dqn_returns}")

        dqn_sharpe = calculate_sharpe_ratio(dqn_returns)
        dqn_solino = calculate_sortino_ratio(dqn_returns)
        dqn_vol = np.std(dqn_returns)
        dqn_mdd = calculate_max_drawdown(dqn_cum_profits)
        dqn_annualised = calculate_calmar_ratio(dqn_cum_profits) 

        # Calmar Ratio
        if  dqn_mdd > 0:
            dqn_calmar = dqn_annualised / dqn_mdd
        else:
            dqn_calmar = float('inf')
        
        stability = stability_of_returns(dqn_returns, threshold=0.02)
        dqn_win_rate, dqn_avg_return = calculate_trade_stats(dqn_actions, dqn_returns)
        print("DQN steps:", len(dqn_dates), len(dqn_cum_profits))

        print(f"[DEBUG] PLR: {dqn_plr}, Win Rate: {dqn_win_rate}, Avg Trade Return: {dqn_avg_return}")
        print(f"[DEBUG] DQN stability: {stability}")
        


        # Evaluate SDP heuristic
        sdp_model = CryptoSDPModel(price_series_test, volume_series_test, initial_cash=30000.0)
        sdp_dates, sdp_cum_profits, sdp_actions, sdp_plr, sdp_returns = evaluate_heuristic_sdp_cumulative(
            sdp_model, date_series_test
        )
        sdp_final = sdp_cum_profits[-1] if sdp_cum_profits else 0
        sdp_sharpe = calculate_sharpe_ratio(sdp_returns)
        sdp_solino = calculate_sortino_ratio(sdp_returns)
        sdp_vol = np.std(sdp_returns)
        sdp_mdd = calculate_max_drawdown(sdp_cum_profits)
        sdp_annualised = calculate_calmar_ratio(sdp_cum_profits) 

        if  sdp_mdd > 0:
            sdp_calmar = sdp_annualised / sdp_mdd
        else:
            sdp_calmar = float('inf')

        sdp_win_rate, sdp_avg_return = calculate_trade_stats(sdp_actions, sdp_returns)

        print(f"[DEBUG] SDP PLR: {sdp_plr}, SDP Win Rate: {sdp_win_rate}, SDP Avg Trade Return: {sdp_avg_return}")
    

        # Evaluate simple
        simp_env = CryptoEnvRL(
            price_series_test, 
            volume_series_test,
            initial_cash=30000.0
        )
        simp_dates, simp_cumprofits, simp_step_actions, simple_plr, simp_returns = evaluate_simple_policy_cumulative(
            simp_env, date_series_test
        )
        simple_final = simp_cumprofits[-1] if simp_cumprofits else 0
        simple_sharpe = calculate_sharpe_ratio(simp_returns)
        simple_solino = calculate_sortino_ratio(simp_returns)
        simple_vol = np.std(simp_returns)
        simple_mdd = calculate_max_drawdown(simp_cumprofits)
        simple_annualised = calculate_calmar_ratio(simp_cumprofits) 

        if  simple_mdd > 0:
            simple_calmar = simple_annualised / simple_mdd
        else:
            simple_calmar = float('inf')
            
        simple_win_rate, simple_avg_return = calculate_trade_stats(simp_step_actions, simp_returns)

        print(f"[DEBUG] Simple PLR: {simple_plr},  Win Rate: {simple_win_rate}, simple Avg Trade Return: {simple_avg_return}")

        # Plot three strategies
        plot_three_strategies(
            dqn_dates, dqn_cum_profits, 
            sdp_cum_profits, 
            simp_cumprofits
        )

        results_df = pd.DataFrame({
            "Algorithm": ["DQN", "SDP_HER", "Simple"],
            "FinalProfit": [dqn_final, sdp_final, simple_final],
            "SharpeRatio": [dqn_sharpe, sdp_sharpe, simple_sharpe],
            "SolinoRatio": [dqn_solino, sdp_solino, simple_solino],
            "MaxDrawdown": [dqn_mdd , sdp_mdd, simple_mdd],
            "CalmarRatio": [dqn_calmar, sdp_calmar, simple_calmar],
            "WinRate": [dqn_win_rate * 100, sdp_win_rate * 100, simple_win_rate * 100],
            "AvgTradeReturn": [dqn_avg_return, sdp_avg_return, simple_avg_return],
            "Volatility": [dqn_vol, sdp_vol, simple_vol],
            "PLR": [dqn_plr, sdp_plr, simple_plr],
            })
        results_df = results_df[["Algorithm", "FinalProfit", "SharpeRatio", "SolinoRatio","MaxDrawdown", "CalmarRatio", "WinRate", "AvgTradeReturn", "Volatility", "PLR"]]

        print("\nComparison of DQN, SDP-Her, and Simple on Test Data:")
        print(results_df)

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


        # Compare actions visually
        compare_actions(price_series_test, dqn_actions, "DQN", sdp_actions, "SDP Heuristic")

        train_time_steps, train_portfolio_vals = evaluate_portfolio_on_training_data(env, agent)

        dqn_dates, dqn_cum_rewards, dqn_portfolio_values, dqn_actions = evaluate_agent_cumulative_with_portfolio(
            test_env, 
            trained_agent, 
            date_series_test
        )

        plot_portfolio_value(dqn_dates, dqn_portfolio_values, title="Agent's Portfolio Value on Test Data")

        plot_portfolio_value(train_time_steps, train_portfolio_vals, title="Agent's Portfolio Value on Training Data")


        results_d = pd.DataFrame({
        "Algorithm": ["DQN", "SDP_HER", "Simple"],
        "FinalProfit": [dqn_final, sdp_final, simple_final],
        "SharpeRatio": [dqn_sharpe, sdp_sharpe, simple_sharpe],
        "SolinoRatio": [dqn_solino, sdp_solino, simple_solino],
        "WinRate": [dqn_win_rate, sdp_win_rate, simple_win_rate],
        "AvgTradeReturn": [dqn_avg_return, sdp_avg_return, simple_avg_return],
        "PLR": [dqn_plr, sdp_plr, simple_plr],
        })
        results_d = results_d[["Algorithm", "FinalProfit", "SharpeRatio", "SolinoRatio", "WinRate", "AvgTradeReturn", "PLR"]]

        print("\nComparison of DQN, SDP-Her, and Simple :")
        print(results_d)
   
    plot_equity_vs_buyhold(dqn_cum_profits, price_series_test, initial_cash=35000.0)
    plot_drawdown(dqn_cum_profits)
    plot_return_distribution(dqn_returns)
    plot_trade_distribution(dqn_actions, dqn_returns)


    # ------------------------------------------------
    # Plot training rewards/losses as an animation
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

    ani = FuncAnimation(fig, update, frames=episodes, interval=200, repeat=False)
    plt.show()