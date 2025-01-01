import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import random
from collections import Counter


from environments.CryptoEnvRL import CryptoEnvRL  
from models.RLAgent import DQNAgent
from environments.CryptoSDPModel import CryptoSDPModel



def evaluate_agent(env, agent, dates, print_qvals=False):
    """
    Evaluate the agent on the given environment in a deterministic (greedy) way.

    Args:
        env (gym.Env): Instance of the CryptoEnvRL or similar environment.
        agent (DQNAgent): The trained agent with a .model that produces Q-values.
        print_qvals (bool): If True, will print Q-values at each step for debugging.

    Returns:
        total_reward (float): Sum of all rewards obtained by the agent.
        actions_taken (list): List of actions (0=Buy, 1=Hold, 2=Sell) taken at each step.
        rewards (list): List of step-by-step rewards obtained by the agent.
    """
    # Reset the environment to start
    state = env.reset()
    
    # Force the agent to exploit (set epsilon=0), just to be safe
    # (Alternatively, rely on your agent.act(..., train=False) to bypass exploration.)
    old_epsilon = agent.epsilon  # Store the current epsilon
    agent.epsilon = 0.0          # Turn  exploration
    dates_list = []
    done = False
    total_reward = 0.0
    actions_taken = []

    step = 0
    
    while not done:
        # Optionally debug: Check Q-values
        if print_qvals:
            with torch.no_grad():
                q_vals = agent.model(torch.FloatTensor(state).unsqueeze(0).to(agent.device))
            # Argmax is the chosen action
            action = np.argmax(q_vals.cpu().numpy())
            print(f"Step={step:4d}, Q-vals={q_vals.cpu().numpy()}, Action={action}")
        else:
            # Evaluate policy in a purely greedy way
            action = agent.act(state, train=False)
        
        # Step the environment
        next_state, reward, done, _ = env.step(action)
        
        total_reward += reward
        actions_taken.append(action)
        dates_list.append(dates[step])

        # Move to the next state
        state = next_state
        step += 1
    
    # Restore agent's original epsilon
    agent.epsilon = old_epsilon
    
    return  dates_list, total_reward, actions_taken

def evaluate_agent_cumulative(env, agent, dates):
    """
    Step-by-step evaluation: (dates_list, cum_profits, actions).
    """
    state = env.reset()
    done = False
    step = 0
    cum_profit = 0.0

    old_epsilon = agent.epsilon  # Store the current epsilon
    agent.epsilon = 0.1 

    dates_list = []
    cum_profits = []
    actions_list = []

    while not done:
        action = agent.act(state, train=False)
        next_state, reward, done, _ = env.step(action)
        cum_profit += reward

        dates_list.append(dates[step])
        cum_profits.append(cum_profit)
        actions_list.append(action)

        step += 1
        state = next_state
    
    # Restore agent's original epsilon
    agent.epsilon = old_epsilon

    return dates_list, cum_profits, actions_list

def evaluate_heuristic_sdp_cumulative(sdp_model, dates):
    """
    Evaluate the CryptoSDPModel with the advanced heuristic policy, 
    tracking cumulative profits and actions over time.

    Args:
        sdp_model (CryptoSDPModel): The SDP model instance.
        dates (pd.DatetimeIndex): Corresponding dates for the test series.

    Returns:
        tuple: (dates_list, cum_profits, actions_list)
    """
    sdp_model.reset()
    done = False
    step = 0
    cum_profit = 0.0

    dates_list = []
    cum_profits = []
    actions_list = []

    while not done:
        info = sdp_model.exog_info_fn()
        rsi = info["rsi"]
        current_price = sdp_model.price_series[sdp_model.t]

        if rsi < 40 and current_price < info["sma"]:
            action = 0  # Buy
        elif rsi > 60 or current_price > info["sma"] * 1.05:
            action = 2  # Sell
        else:
            action = 1  # Hold

        _, reward, done = sdp_model.transition_fn(action)
        cum_profit += reward

        dates_list.append(dates[step])
        cum_profits.append(cum_profit)
        actions_list.append(action)

        step += 1

    return dates_list, cum_profits, actions_list


def evaluate_simple_policy_cumulative(env, dates):
    """
    Simple policy: buy if price < mean, else sell if price>mean, else hold.
    Step-by-step to gather (date, cum_profit).
    """
    price_arr = env.price_series
    mean_price = np.mean(price_arr)

    env.reset()
    done = False
    step = 0
    cum_profit = 0.0

    dates_list = []
    cum_profits = []
    actions_list = []

    while not done:
        current_price = price_arr[env.t]
        if current_price < mean_price:
            action = 0  # buy
        elif current_price > mean_price:
            action = 2  # sell
        else:
            action = 1  # hold

        _, reward, done, _ = env.step(action)
        cum_profit += reward

        dates_list.append(dates[step])
        cum_profits.append(cum_profit)
        actions_list.append(action)

        step += 1

    return dates_list, cum_profits, actions_list

def plot_three_strategies(dqn_dates, dqn_cum, sdp_cum, simple_cum):
    """
    Plot 3 lines: DQN, SDP, Simple on the same date-based x-axis.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dqn_dates, dqn_cum, label='DQN', color='blue')
    plt.plot(dqn_dates, sdp_cum, label='HER', color='orange')
    plt.plot(dqn_dates, simple_cum, label='Simple', color='green')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Profit')
    plt.title(f"Compare: DQN vs. HER vs. Simple ")
    plt.legend()
    plt.show()


def simple_policy(price_series):
    """Simple trading policy based on mean price."""
    # Ensure the input is valid
    assert len(price_series) > 0, "Price series is empty!"
    assert not np.isnan(price_series).any(), "Price series contains NaN values!"

    mean_price = np.mean(price_series)
    print(f"Simple Policy Mean Price: {mean_price}")

    actions = [0 if price < mean_price else 2 if price > mean_price else 1 for price in price_series]
    print(f"Simple Policy Actions Distribution: {Counter(actions)}")
    return actions


def plot_results(price_series, actions_trained, actions_simple, title1="Trained Agent Actions", title2="Simple Policy Actions"):
    """Plot the results of the trained agent and the simple policy."""
    plt.figure(figsize=(20, 10))

    # Plot for trained agent
    plt.subplot(1, 2, 1)
    plt.plot(price_series, label='Price', color='gray')
    buy_signals_trained = [i for i, a in enumerate(actions_trained) if int(a) == 0]
    sell_signals_trained = [i for i, a in enumerate(actions_trained) if int(a) == 2]
    plt.scatter(buy_signals_trained, [price_series[i] for i in buy_signals_trained], color='green', marker='^', label='Buy')
    plt.scatter(sell_signals_trained, [price_series[i] for i in sell_signals_trained], color='red', marker='v', label='Sell')
    plt.title(title1)
    plt.legend()

    # Plot for simple policy
    plt.subplot(1, 2, 2)
    plt.plot(price_series, label='Price', color='gray')
    buy_signals_simple = [i for i, a in enumerate(actions_simple) if a == 0]
    sell_signals_simple = [i for i, a in enumerate(actions_simple) if a == 2]
    plt.scatter(buy_signals_simple, [price_series[i] for i in buy_signals_simple], color='green', marker='^', label='Buy')
    plt.scatter(sell_signals_simple, [price_series[i] for i in sell_signals_simple], color='red', marker='v', label='Sell')
    plt.title(title2)
    plt.legend()

    plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def analyze_rolling_std(price_series, window=14):
    """
    Analyze rolling standard deviation (volatility) of price series.
    
    Args:
        price_series (np.ndarray): Array of price values.
        window (int): Rolling window size for calculating standard deviation.

    Returns:
        None (Displays histogram and line plot of rolling std)
    """
    # Convert to pandas Series for convenience
    price_series = pd.Series(price_series)
    
    # Calculate rolling standard deviation
    rolling_std = price_series.rolling(window=window).std()
    
    # Plot rolling standard deviation over time
    plt.figure(figsize=(12, 6))
    plt.plot(rolling_std, label='Rolling Std (Volatility)', color='blue')
    plt.title(f'Rolling Standard Deviation of Prices (Window = {window})')
    plt.xlabel('Time')
    plt.ylabel('Standard Deviation')
    plt.legend()
    plt.grid()
    plt.show()
    
    # Plot histogram of rolling standard deviation
    plt.figure(figsize=(8, 6))
    plt.hist(rolling_std.dropna(), bins=30, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of Rolling Std (Window = {window})')
    plt.xlabel('Rolling Std')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()
    
    # Print percentiles to guide threshold selection
    percentiles = [10, 25, 50, 75, 90]
    for p in percentiles:
        print(f"{p}th Percentile: {np.percentile(rolling_std.dropna(), p):.4f}")
   
def evaluate_agent_cumulative_with_portfolio(env, agent, dates):
    """
    Evaluate the agent step-by-step, returning cumulative profits and portfolio values.

    Args:
        env (CryptoEnvRL): Trading environment.
        agent (DQNAgent): Trained DQN agent.
        dates (pd.DatetimeIndex): Date series for visualization.

    Returns:
        tuple: (dates_list, cum_profits, portfolio_values, actions_list)
    """
    state = env.reset()
    done = False
    step = 0
    cum_profit = 0.0
    portfolio_value = 0.0

    old_epsilon = agent.epsilon  # Store the current epsilon
    agent.epsilon = 0.0  # Pure exploitation during testing

    dates_list = []
    cum_profits = []
    portfolio_values = []
    actions_list = []

    while not done:
        action = agent.act(state, train=False)
        next_state, reward, done, _ = env.step(action)
        cum_profit += reward
        portfolio_value = (env.stock_owned * env.price_series[env.t]) + cum_profit

        dates_list.append(dates[step])
        cum_profits.append(cum_profit)
        portfolio_values.append(portfolio_value)
        actions_list.append(action)

        step += 1
        state = next_state

    # Restore agent's original epsilon
    agent.epsilon = old_epsilon

    return dates_list, cum_profits, portfolio_values, actions_list


def plot_cumulative_profit(dates, total_reward, title="Cumulative Profit Over Time"):
    """
    Plot cumulative profits over time.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, total_reward, label='Cumulative Profit', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Profit')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


def plot_portfolio_value(dates, portfolio_values, title="Portfolio Value Over Time"):
    """
    Plot portfolio value over time.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, portfolio_values, label='Portfolio Value', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


def run_tests_with_separate_plots(test_env, trained_agent, date_series):
    """
    Run tests on the trained agent and display separate plots for cumulative profit and portfolio value.
    """
    # Evaluate trained agent with cumulative profits and portfolio values
    dqn_dates, dqn_cum_profits, dqn_portfolio_values, dqn_actions = evaluate_agent_cumulative_with_portfolio(
        test_env, trained_agent, date_series
    )
    print(f"DQN Total Reward: {dqn_cum_profits[-1]:.2f}")
    print(f"DQN Actions Distribution: {Counter(dqn_actions)}")

    # Plot cumulative profits
    plot_cumulative_profit(dqn_dates, dqn_cum_profits, title="DQN: Cumulative Profit Over Time")

    # Plot portfolio value
    plot_portfolio_value(dqn_dates, dqn_portfolio_values, title="DQN: Portfolio Value Over Time")


def calculate_metrics(rewards, actions):
    cumulative_return = np.sum(rewards)
    returns = np.array(rewards)
    if np.std(returns) != 0:
        sharpe_ratio = np.mean(returns) / np.std(returns)
    else:
        sharpe_ratio = 0
    return cumulative_return, sharpe_ratio

def run_tests(test_env, trained_agent, date_series):
    """Run tests on the trained agent and compare with a simple policy."""
    # Evaluate trained agent
    dqn_dates, dqn_cum_profits, dqn_actions = evaluate_agent_cumulative(test_env, trained_agent, date_series)
    print(f"DQN Total Reward: {dqn_cum_profits[-1]:.2f}")
    print(f"DQN Actions Distribution: {Counter(dqn_actions)}")

    # Evaluate SDP Heuristic
    sdp_model = CryptoSDPModel(test_env.price_series, test_env.volume_series)
    sdp_dates, sdp_cum_profits, sdp_actions = evaluate_heuristic_sdp_cumulative(sdp_model, date_series)
    print(f"SDP Total Reward: {sdp_cum_profits[-1]:.2f}")
    print(f"SDP Actions Distribution: {Counter(sdp_actions)}")

    # Evaluate Simple Policy
    simp_dates, simp_cumprofits, simp_step_actions = evaluate_simple_policy_cumulative(test_env, date_series)
    print(f"Simple Policy Total Reward: {simp_cumprofits[-1]:.2f}")
    print(f"Simple Policy Actions Distribution: {Counter(simp_step_actions)}")