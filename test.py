# test.py
import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from environments.CryptoEnvRL import CryptoEnvRL  # or your updated env file
from models.RLAgent import DQNAgent
from environments.CryptoSDPModel import CryptoSDPModel


def evaluate_agent(env, agent, dates, print_qvals=False):
    """
    Evaluate the agent on the given environment in a deterministic (epsilon=0) way.
    Args:
        env (CryptoEnvRL): The environment instance (already loaded with test data).
        agent (DQNAgent): The trained DQN agent.
        dates (pd.DatetimeIndex): For plotting or reference.
        print_qvals (bool): If True, will print Q-values each step for debugging.
    Returns:
        total_reward (float): Sum of all rewards obtained.
        actions_taken (list): The actions at each step (0=Buy,1=Hold,2=Sell).
        rewards (list): The step-by-step rewards.
    """
    state = env.reset()
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Force greedy (no exploration)

    done = False
    step = 0
    total_reward = 0.0
    actions_taken = []
    rewards_list = []
    dates_list = []

    while not done:
        if print_qvals:
            with torch.no_grad():
                q_vals = agent.model(
                    torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                )
            action = np.argmax(q_vals.cpu().numpy())
            print(f"Step={step}, Q-vals={q_vals.cpu().numpy()}, Action={action}")
        else:
            action = agent.act(state, train=False)

        next_state, reward, done, _ = env.step(action)

        total_reward += reward
        actions_taken.append(action)
        rewards_list.append(reward)
        if step < len(dates):
            dates_list.append(dates[step])

        state = next_state
        step += 1

    agent.epsilon = old_epsilon  # restore
    return total_reward, actions_taken, rewards_list


def evaluate_agent_cumulative(env, agent, dates):
    """
    Similar to the above, but we store cumulative profits step by step.
    For an environment that accumulates reward for each trade,
    we can define 'cum_profit' as sum of step rewards.
    Returns:
        (dates_list, cum_profits, actions_list)
    """
    state = env.reset()
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    done = False
    step = 0
    cum_profit = 0.0

    successful_trades = 0
    unsuccessful_trades = 0

    dates_list = []
    cum_profits = []
    actions_list = []
    step_rewards = []

    while not done:
        action = agent.act(state, train=False)
        next_state, reward, done, _ = env.step(action)
        cum_profit += reward
        step_rewards.append(reward)

        if action == 2: 
            if reward > 0:
                successful_trades += 1
            elif reward < 0:
                unsuccessful_trades += 1

        if step < len(dates):
            dates_list.append(dates[step])
        cum_profits.append(cum_profit)
        actions_list.append(action)

        step += 1
        state = next_state

    agent.epsilon = old_epsilon

    plr = successful_trades / (successful_trades + unsuccessful_trades) if (successful_trades + unsuccessful_trades) > 0 else 0

    print(f"[DEBUG] Step: {step}, Action: {action}, Reward: {reward}, Cum Profit: {cum_profit}")
    print(f"[DEBUG] Successful Trades: {successful_trades}, Unsuccessful Trades: {unsuccessful_trades}")
    
    
    if successful_trades + unsuccessful_trades == 0:
        print("[DEBUG] No trades executed during evaluation.")


    return dates_list, cum_profits, actions_list, plr, step_rewards


def evaluate_agent_cumulative_with_portfolio(env, agent, dates):
    """
    Evaluate the agent step-by-step, returning both 'cum_profits' and 'portfolio_values'.
    Since your environment has env.portfolio_value, we can track it directly.
    Returns:
        (dates_list, cum_rewards, portfolio_values, actions_list)
    """
    state = env.reset()
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    done = False
    step = 0
    cum_reward = 0.0


    dates_list = []
    cum_rewards = []
    portfolio_values = []
    actions_list = []

    while not done:
        action = agent.act(state, train=False)
        next_state, reward, done, _ = env.step(action)
        cum_reward += reward
        current_portfolio = env.portfolio_value  # see environment's property

        if step < len(dates):
            dates_list.append(dates[step])
        cum_rewards.append(cum_reward)
        portfolio_values.append(current_portfolio)
        actions_list.append(action)

        step += 1
        state = next_state

    agent.epsilon = old_epsilon
    return dates_list, cum_rewards, portfolio_values, actions_list

def evaluate_heuristic_sdp_cumulative(sdp_model, dates):
    """
    Evaluate the CryptoSDPModel with advanced_heuristic_policy_sdp (or your custom policy).
    Returns (dates_list, cum_profits, actions_list).
    We won't track 'portfolio_value' here unless you add a 'cash' logic to your SDP model.
    """
    sdp_model.reset()
    done = False
    step = 0
    cum_profit = 0.0

    successful_trades = 0
    unsuccessful_trades = 0

    dates_list = []
    cum_profits = []
    actions_list = []
    step_rewards = []

    while not done:
        info = sdp_model.exog_info_fn()
        current_price = sdp_model.price_series[sdp_model.t]

        # Basic heuristic
        if (info["rsi"] < 40) and (current_price < info["sma"]):
            action = 0  # Buy
        elif (info["rsi"] > 60) or (current_price > info["sma"] * 1.05):
            action = 2  # Sell
        else:
            action = 1  # Hold

        next_state, reward, done = sdp_model.transition_fn(action)
        cum_profit += reward
        step_rewards.append(reward)

        if action == 2: 
            if reward > 0:
                successful_trades += 1
            elif reward < 0:
                unsuccessful_trades += 1

        if step < len(dates):
            dates_list.append(dates[step])
       
        cum_profits.append(cum_profit)
        actions_list.append(action)

        step += 1
    
    plr = successful_trades / (successful_trades + unsuccessful_trades) if (successful_trades + unsuccessful_trades) > 0 else 0

    print(f"[DEBUG] SDP Successful Trades: {successful_trades}, SDP Unsuccessful Trades: {unsuccessful_trades}")

    if successful_trades + unsuccessful_trades == 0:
        print("[DEBUG] No trades executed during evaluation for SDP.")

    return dates_list, cum_profits, actions_list, plr,step_rewards


def evaluate_simple_policy_cumulative(env, dates):
    """
    A naive policy: buy if price<mean, else sell if price>mean, else hold.
    Tracks cumulative profit the same way as evaluate_agent_cumulative does.
    """
    price_arr = env.price_series
    mean_price = np.mean(price_arr)

    env.reset()
    done = False
    step = 0
    cum_profit = 0.0

    successful_trades = 0
    unsuccessful_trades = 0

    dates_list = []
    cum_profits = []
    actions_list = []
    step_rewards = []

    while not done:
        current_price = price_arr[env.t]
        if current_price < mean_price:
            action = 0  # Buy
        elif current_price > mean_price:
            action = 2  # Sell
        else:
            action = 1  # Hold

        next_state, reward, done, _ = env.step(action)
        cum_profit += reward
        step_rewards.append(reward)

        if action == 2: 
            if reward > 0:
                successful_trades += 1
            elif reward < 0:
                unsuccessful_trades += 1

        if step < len(dates):
            dates_list.append(dates[step])

        cum_profits.append(cum_profit)
        actions_list.append(action)

        step += 1
    
    plr = successful_trades / (successful_trades + unsuccessful_trades) if (successful_trades + unsuccessful_trades) > 0 else 0

    print(f"[DEBUG] SIMPLE Successful Trades: {successful_trades}, SIMPLE Unsuccessful Trades: {unsuccessful_trades}")

    if successful_trades + unsuccessful_trades == 0:
        print("[DEBUG] No trades executed during evaluation for Simple.")

    return dates_list, cum_profits, actions_list, plr, step_rewards


def plot_three_strategies(dqn_dates, dqn_cum, sdp_cum, simple_cum):
    """
    Plot 3 lines: DQN, SDP_Heuristic, Simple. Each is a cumulative profit curve vs. time.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dqn_dates, dqn_cum, label='DQN', color='blue')
    plt.plot(dqn_dates, sdp_cum, label='SDP', color='orange')
    plt.plot(dqn_dates, simple_cum, label='Simple', color='green')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Profit')
    plt.title("Comparison: DQN vs SDP vs Simple")
    plt.legend()
    plt.show()


def simple_policy(price_series):
    mean_price = np.mean(price_series)
    actions = []
    for price in price_series:
        if price < mean_price:
            actions.append(0)  # Buy
        elif price > mean_price:
            actions.append(2)  # Sell
        else:
            actions.append(1)  # Hold
    return actions


def plot_portfolio_value(time_steps, portfolio_values, title="Portfolio Value Over Time"):
    """
    Plot a single line of portfolio values vs time steps.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(time_steps, portfolio_values, label='Portfolio Value', color='blue')
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.show()


def analyze_rolling_std(price_series, window=14):
    """
    Plot the rolling std for price_series and a histogram, to see volatility distribution.
    """
    price_series = pd.Series(price_series)
    rolling_std = price_series.rolling(window=window).std()

    plt.figure(figsize=(12, 6))
    plt.plot(rolling_std, label='Rolling Std (Volatility)', color='blue')
    plt.title(f'Rolling Standard Deviation (Window={window})')
    plt.xlabel('Time')
    plt.ylabel('Std')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.hist(rolling_std.dropna(), bins=30, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of Rolling Std (Window={window})')
    plt.xlabel('Rolling Std')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()

    # Print some percentiles
    for p in [10, 25, 50, 75, 90]:
        val = np.percentile(rolling_std.dropna(), p)
        print(f"{p}th Percentile = {val:.4f}")


def calculate_sharpe_ratio(returns, risk_free_rate=0.01):
    """
    Sharpe Ratio = (mean(returns - r_f)) / std(returns).
    If there's no variability or too few data points, returns 0.0
    """
    returns_arr = np.array(returns)
    if len(returns_arr) < 2:
        return 0.0
    excess = returns_arr - risk_free_rate
    if excess.std() == 0:
        return 0.0
    return excess.mean() / excess.std()

def calculate_sortino_ratio(returns, risk_free_rate=0.01):
    """
    Sortino Ratio = (mean(returns - r_f)) / std(negative excess returns)

    returns (list or np.array): Step or daily returns.
    risk_free_rate (float): risk-free rate.
    """
    arr = np.array(returns)
    if len(arr) < 2:
        return 0.0

    excess = arr - risk_free_rate
    # Only negative returns matter for the denominator
    downside = excess[excess < 0]

    mean_excess = excess.mean()
    if len(downside) < 1:
        # If no negative returns, the ratio is effectively infinite
        return float('inf')

    # Downside volatility
    downside_std = downside.std()
    if downside_std == 0:
        # If no negative deviation, ratio is infinite
        return float('inf')

    sortino = mean_excess / downside_std
    return sortino

def calculate_max_drawdown(equity_curve):
    """
    Calculates the maximum drawdown for a given equity curve (list or array).
    equity_curve: array-like of portfolio or cumulative profit over time

    Returns: float (max drawdown as a positive fraction), e.g. 0.30 means 30% drawdown
    """
    arr = np.array(equity_curve)
    if len(arr) < 2:
        return 0.0

    peak = arr[0]
    max_dd = 0.0
    for x in arr:
        peak = max(peak, x)
        dd = (peak - x) / peak
        max_dd = max(max_dd, dd)
    return max_dd  

def calculate_calmar_ratio(equity_curve, periods_per_year=365):
    """
    Calmar Ratio = (annualized return) / (max drawdown)

    equity_curve: array-like of portfolio values
    periods_per_year: e.g. 365 if daily, 252 if trading days in a year, etc.
    """
    max_dd = calculate_max_drawdown(equity_curve)
    if max_dd == 0:
        return float('inf')

    start_val = equity_curve[0]
    end_val = equity_curve[-1]
    n = len(equity_curve)
    # approximate # of years
    years = n / float(periods_per_year)
    # if years < 1.0 => we just annualize anyway
    if start_val <= 0:
        return 0.0

    total_return = (end_val / start_val) - 1.0
    # annualized return (geometric)
    annual_return = (1 + total_return)**(1/years) - 1 if years > 0 else 0

    return annual_return / max_dd

def calculate_trade_stats(actions, rewards):
    """
    actions: list of actions (0=Buy,1=Hold,2=Sell)
    rewards: the step-level reward at each step
    returns: (win_rate, avg_trade_return)
    """
    trade_profits = []
    for a, r in zip(actions, rewards):
        if a == 2:  # Sell
            trade_profits.append(r)  # the environment's reward on Sell is the trade's P/L

    if len(trade_profits) == 0:
        return 0.0, 0.0  # no sells

    wins = sum(1 for p in trade_profits if p > 0)
    total = len(trade_profits)
    win_rate = wins / total
    avg_return = np.mean(trade_profits)
    return win_rate, avg_return

def stability_of_returns(returns, threshold=0.02):
    """
    Returns the fraction of steps that have a 'large' absolute return 
    bigger than the threshold (e.g. 2%).
    The lower this fraction, the more 'stable' the returns.
    """
    arr = np.array(returns)
    if len(arr) < 1:
        return 1.0
    big_moves = sum(1 for r in arr if abs(r) > threshold)
    fraction_big_moves = big_moves / len(arr)
    # If you want "stability index," maybe 1 - fraction_big_moves
    return 1 - fraction_big_moves

def plot_equity_vs_buyhold(portfolio_vals, price_series, initial_cash=30000.0):
    """
    Compare your RL portfolio to a naive buy-and-hold strategy on the same asset.
    portfolio_vals: array of your strategy's portfolio values
    price_series: array of the asset's price
    """
    # Buy-hold: how many shares if you spent all cash at t=0
    first_price = price_series[0]
    shares_bought = initial_cash / first_price
    bh_curve = shares_bought * price_series

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,6))
    plt.plot(portfolio_vals, label='RL Strategy', color='blue')
    plt.plot(bh_curve, label='Buy-and-Hold', color='orange')
    plt.title("Equity Curve vs. Buy & Hold")
    plt.xlabel("Time Steps")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid()
    plt.show()


def plot_drawdown(equity_vals):
    """
    Plot drawdown over time. equity_vals is array-like of the portfolio or equity curve.
    """
    arr = np.array(equity_vals)
    peak = arr[0]
    drawdowns = []
    for x in arr:
        peak = max(peak, x)
        dd = (peak - x) / peak
        drawdowns.append(dd)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,6))
    plt.plot(drawdowns, color='red', label='Drawdown')
    plt.title("Drawdown Over Time")
    plt.xlabel("Time Steps")
    plt.ylabel("Drawdown (fraction)")
    plt.grid()
    plt.legend()
    plt.show()


def plot_return_distribution(returns):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8,6))
    plt.hist(returns, bins=30, color='skyblue', edgecolor='black')
    plt.title("Distribution of Step (or Daily) Returns")
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


def plot_trade_distribution(actions, rewards):
    """
    For each Sell, reward is the trade P/L. 
    Plot distribution of trade P/L and show the number of trades.
    """
    trade_pl = []
    for a, r in zip(actions, rewards):
        if a == 2:  # Sell
            trade_pl.append(r)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,6))
    plt.hist(trade_pl, bins=30, color='lightgreen', edgecolor='black')
    plt.title(f"Distribution of Trade P/L (Total Trades={len(trade_pl)})")
    plt.xlabel("Trade Profit/Loss")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


def evaluate_portfolio_on_training_data(env, agent):
    """
    After training, run a purely greedy pass on the SAME training environment
    from the start, collecting the portfolio value at each step.
    Returns (time_steps, portfolio_values).
    """
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    obs = env.reset()
    done = False
    step_count = 0

    time_steps = []
    portfolio_values = []

    while not done:
        action = agent.act(obs, train=False)
        next_obs, reward, done, _ = env.step(action)

        portfolio_values.append(env.portfolio_value)
        time_steps.append(step_count)

        step_count += 1
        obs = next_obs

    agent.epsilon = old_epsilon
    return time_steps, portfolio_values


def run_tests(test_env, trained_agent, date_series):
    """
    Example test harness that runs the agent, prints total reward, 
    plus distributions of actions, then maybe calls other policies, etc.
    """
   
    dqn_dates, dqn_cum_profits, dqn_actions, dqn_plr, dqn_returns = evaluate_agent_cumulative(test_env, trained_agent, date_series)
    total_reward = dqn_cum_profits[-1] if len(dqn_cum_profits) > 0 else 0
    print(f"DQN Total Reward (final cum profit) = {total_reward:.2f}")
    print(f"DQN Action Distribution = {Counter(dqn_actions)}")

    sdp_model = CryptoSDPModel(test_env.price_series, test_env.volume_series)
    sdp_dates, sdp_cum_profits, sdp_actions,sdp_plr, sdp_returns = evaluate_heuristic_sdp_cumulative(sdp_model, date_series)
    print(f"SDP Total Reward: {sdp_cum_profits[-1]:.2f}")
    print(f"SDP Actions Distribution: {Counter(sdp_actions)}")
    print(f"SDP returns SDP: {Counter(sdp_returns)}")


    # Evaluate Simple Policy
    simp_dates, simp_cumprofits, simp_step_actions, simple_plr, simp_returns = evaluate_simple_policy_cumulative(test_env, date_series)
    print(f"Simple Policy Total Reward: {simp_cumprofits[-1]:.2f}")
    print(f"Simple Policy Actions Distribution: {Counter(simp_step_actions)}")
    print(f"SDP returns simple: {Counter(simp_returns)}")
