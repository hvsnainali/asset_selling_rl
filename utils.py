import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def evaluate_agent(env, agent, episodes=10, export_csv=False):
    """
    Evaluate the agent on the given environment and visualize its actions.

    Args:
        env: The trading environment.
        agent: The trained DQN agent.
        episodes (int): Number of episodes to evaluate.
        export_csv (bool): If True, save evaluation metrics to a CSV file.
    """
    results = []  # Store metrics for all episodes
    action_counts = {"Buy": 0, "Sell": 0, "Hold": 0}  # Track action distribution

    for episode in range(episodes):
        print(f"Running evaluation for Episode {episode + 1}/{episodes}")
        state = env.reset()
        total_profit = 0
        done = False
        step = 0

        prices = []
        portfolio_values = []
        actions = []
        buying_points = []
        selling_points = []
        macd_signals = []

        while not done:
            prices.append(env.price_series[env.t])
            portfolio_values.append((env.stock_owned * env.price_series[env.t]) + total_profit)
            macd_signals.append(env.macd[env.t])
            action = agent.act(state, train=False)
            actions.append(action)

            if action == 0:
                action_counts["Buy"] += 1
                buying_points.append((env.t, env.price_series[env.t]))
            elif action == 1:
                action_counts["Hold"] += 1
            elif action == 2:
                action_counts["Sell"] += 1
                selling_points.append((env.t, env.price_series[env.t]))

            next_state, reward, done, _ = env.step(action)
            total_profit += reward
            state = next_state
            step += 1

        # Calculate metrics for the episode
        successful_trades = sum(1 for i in range(len(actions)) if actions[i] == 2 and reward > 0)
        unsuccessful_trades = sum(1 for i in range(len(actions)) if actions[i] == 2 and reward < 0)
        plr = successful_trades / max(1, successful_trades + unsuccessful_trades)

        results.append({
            "Episode": episode + 1,
            "Total Profit": total_profit,
            "Successful Trades": successful_trades,
            "Unsuccessful Trades": unsuccessful_trades,
            "Profit-Loss Ratio (PLR)": plr,
            "Portfolio Value": portfolio_values[-1] if portfolio_values else 0
        })

        print(f"📈 Evaluation - Episode {episode + 1}")
        print(f"✅ Total Profit: ${total_profit:.2f}")
        print(f"💵 Successful Trades: {successful_trades}")
        print(f"❌ Unsuccessful Trades: {unsuccessful_trades}")
        print(f"📊 Profit-Loss Ratio (PLR): {plr:.2f}")
        print("-----------------------------------")

        # Visualize portfolio value over time for this episode
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(prices)), prices, label="Price", color="blue")
        plt.plot(range(len(portfolio_values)), portfolio_values, label="Portfolio Value", color="purple")
        plt.plot(range(len(macd_signals)), macd_signals, label="MACD Signal", color="orange", linestyle="--")
        plt.scatter(*zip(*buying_points), marker="^", color="green", label="Buy Signal")
        plt.scatter(*zip(*selling_points), marker="v", color="red", label="Sell Signal")
        plt.title(f"Episode {episode + 1} - Total Profit: ${total_profit:.2f}")
        plt.xlabel("Time Steps")
        plt.ylabel("Value")
        plt.legend()
        plt.show()

    # Save metrics to CSV
    if export_csv:
        results_df = pd.DataFrame(results)
        results_df.to_csv("evaluation_results.csv", index=False)
        print("Evaluation results saved to evaluation_results.csv.")

    # Plot overall metrics
    metrics_df = pd.DataFrame(results)

    plt.figure(figsize=(12, 6))

    # Profit curve over episodes
    plt.subplot(1, 3, 1)
    plt.plot(metrics_df["Episode"], metrics_df["Total Profit"], label="Total Profit", color="blue")
    plt.xlabel("Episode")
    plt.ylabel("Profit ($)")
    plt.title("Profit per Episode")
    plt.legend()

    # PLR per episode
    plt.subplot(1, 3, 2)
    plt.plot(metrics_df["Episode"], metrics_df["Profit-Loss Ratio (PLR)"], label="PLR", color="green")
    plt.xlabel("Episode")
    plt.ylabel("Profit-Loss Ratio")
    plt.title("PLR per Episode")
    plt.legend()

    # Action distribution
    plt.subplot(1, 3, 3)
    plt.bar(action_counts.keys(), action_counts.values(), color=["green", "blue", "red"])
    plt.xlabel("Action")
    plt.ylabel("Count")
    plt.title("Action Distribution")

    plt.tight_layout()
    plt.show()