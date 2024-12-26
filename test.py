import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import random

from environments.CryptoEnvRL import CryptoEnvRL  
from models.RLAgent import DQNAgent


def evaluate_agent_deterministic(env, agent):
    state = env.reset()  # Reset environment to the beginning of the test dataset
    done = False
    total_reward = 0
    action_log = []

    while not done:
        action = agent.act(state, train=False)  # Make sure to use the evaluation mode of your agent
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        action_log.append(action)
        state = next_state

    print(f"Total reward from deterministic test environment: {total_reward}")
    return total_reward, action_log

# cumulative_rewards = np.cumsum([reward for _, reward, _, _ in test_env.step_log])
# buy_and_hold_rewards = [price / price_series_test[0] * 100 for price in price_series_test]

# plt.figure(figsize=(12, 6))
# plt.plot(cumulative_rewards, label='Agent Cumulative Rewards')
# plt.plot(buy_and_hold_rewards, label='Buy and Hold Cumulative Return', linestyle='--')
# plt.title('Comparison of Trading Agent vs. Buy and Hold Strategy')
# plt.xlabel('Trading Steps')
# plt.ylabel('Cumulative Returns (%)')
# plt.legend()
# plt.show()

# def buy_and_hold_return(prices):
#     """Calculate the return of a simple buy and hold strategy."""
#     return ((prices[-1] - prices[0]) / prices[0]) * 100

# def sharpe_ratio(returns, risk_free_rate=0):
#     """Calculate the Sharpe Ratio to measure risk-adjusted returns."""
#     mean_return = np.mean(returns)
#     std_return = np.std(returns)
#     return (mean_return - risk_free_rate) / std_return

# def validate_performance(test_env, agent):
#     total_rewards = []
#     state = test_env.reset()
#     done = False
#     while not done:
#         action = agent.act(state, train=False)  # Assume exploration is off for testing
#         next_state, reward, done, _ = test_env.step(action)
#         total_rewards.append(reward)
#         print(f"State: {state}, Action: {action}, Reward: {reward}")
#         state = next_state
#     return total_rewards

# def plot_performance(price_series, test_rewards, buy_and_hold_start):
#     # Calculate the cumulative return for buy and hold from start price
#     buy_and_hold_returns = ((price_series - buy_and_hold_start) / buy_and_hold_start) * 100

#     # Calculate cumulative rewards for the agent
#     agent_cumulative_rewards = np.cumsum(test_rewards)

#     # Setting up the plot
#     plt.figure(figsize=(14, 7))
#     plt.plot(buy_and_hold_returns, label='Buy and Hold Cumulative Return', color='blue')
#     plt.plot(agent_cumulative_rewards, label='Agent Cumulative Rewards', color='green')

#     # Adding labels and title
#     plt.title('Comparison of Trading Agent vs. Buy and Hold Strategy')
#     plt.xlabel('Trading Steps')
#     plt.ylabel('Cumulative Returns (%)')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# def calculate_buy_and_hold_returns(price_series):
#     start_price = price_series[0]
#     end_price = price_series[-1]
#     return_percentage = ((end_price - start_price) / start_price) * 100
#     return return_percentage
    

# def run_tests(test_env, agent):
#     """Convenience function to manage test execution and logging."""
#     print("Testing started...")
#     test_rewards = validate_performance(test_env, agent)
#     total_test_reward = sum(test_rewards)
#     print(f"Total reward from test environment: {total_test_reward}")

#     # Calculating Buy and Hold Return for comparison
#     buy_and_hold_perf = buy_and_hold_return(test_env.price_series)
#     print(f"Buy and Hold Return: {buy_and_hold_perf}%")

#     # Calculating the Sharpe Ratio
#     if len(test_rewards) > 1:  # Ensure there are enough points to perform np.diff
#         print(f"Length of test_rewards: {len(test_rewards)}")
#         print(f"Length of price_series: {len(test_env.price_series)}")
#         daily_returns = np.diff(test_rewards) / test_env.price_series[1:len(test_rewards)]  # Ensure matching lengths
#         sharpe_ratio_value = sharpe_ratio(daily_returns)
#         print(f"Sharpe Ratio: {sharpe_ratio_value}")
#     else:
#         print("Not enough data points to calculate the Sharpe ratio.")

#     # Counting successful and unsuccessful trades
#     successful_trades = sum(1 for reward in test_rewards if reward > 0)
#     unsuccessful_trades = sum(1 for reward in test_rewards if reward <= 0)
#     print(f"Successful Trades: {successful_trades}, Unsuccessful Trades: {unsuccessful_trades}")
    

#     # Visualizing the results
#     plt.figure(figsize=(14, 7))
#     plt.plot(np.cumsum(test_rewards), label='Cumulative Reward')
#     plt.title('Cumulative Reward Over Time')
#     plt.xlabel('Time Steps')
#     plt.ylabel('Cumulative Reward')
#     plt.legend()
#     plt.show()

#     buy_and_hold_start = test_env.price_series[0]  # Starting price for buy and hold
#     plot_performance(test_env.price_series, np.array(test_rewards), buy_and_hold_start)

#     buy_and_hold_return_percentage = calculate_buy_and_hold_returns(test_env.price_series)
#     print(f"Buy and Hold Return: {buy_and_hold_return_percentage:.2f}%")

