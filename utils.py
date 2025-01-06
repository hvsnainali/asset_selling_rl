import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def evaluate_agent(env, agent, episodes=30, export_csv=False):
    """
    Evaluate the agent on the given environment and visualize the aggregated results.

    Args:
        env: The trading environment (CryptoEnvRL).
        agent: The trained DQN agent.
        episodes (int): Number of episodes to evaluate.
        export_csv (bool): If True, save evaluation metrics to a CSV file.
    """
    # Store metrics for all episodes
    results = []
    # Keep track of how many times we see each action across *all* episodes
    action_counts = {"Buy": 0, "Sell": 0, "Hold": 0}

    for episode in range(episodes):
        # Reset environment and agent state
        state = env.reset()
        done = False
        step = 0

        # Accumulate episode metrics
        rewards = []
        actions = []
        total_profit = 0.0

        while not done:
            # Agent picks an action
            action = agent.act(state, train=False)
            actions.append(action)

            if action == 0:
                action_counts["Buy"] += 1
            elif action == 1:
                action_counts["Hold"] += 1
            else:
                action_counts["Sell"] += 1

            # Environment step
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)

            # Update profit as net worth minus initial cash
            total_profit = env.portfolio_value - env.initial_cash

            # Move on
            state = next_state
            step += 1

        # Compute success metrics
        successful_trades = sum(
            1 for i, a in enumerate(actions) if a == 2 and rewards[i] > 0
        )
        unsuccessful_trades = sum(
            1 for i, a in enumerate(actions) if a == 2 and rewards[i] < 0
        )
        plr = successful_trades / max(1, successful_trades + unsuccessful_trades)

        # Record episode results
        results.append({
            "Episode": episode + 1,
            "Total Profit": total_profit,
            "Successful Trades": successful_trades,
            "Unsuccessful Trades": unsuccessful_trades,
            "Profit-Loss Ratio (PLR)": plr
        })

        # Print a brief summary
        print(f"Episode {episode + 1}/{episodes}")
        print(f"  Total Profit: ${total_profit:.2f}")
        print(f"  Successful Trades: {successful_trades}")
        print(f"  Unsuccessful Trades: {unsuccessful_trades}")
        print(f"  PLR: {plr:.2f}")
        print("-----------------------------------")

    # Convert to DataFrame for convenient plotting and/or exporting
    metrics_df = pd.DataFrame(results)

    # Optionally export to CSV
    if export_csv:
        metrics_df.to_csv("evaluation_results.csv", index=False)
        print("Evaluation results saved to evaluation_results.csv.")

    # ---------------------------
    # Single Aggregated Plot
    # ---------------------------
    # We'll show: 
    # 1) Total Profit per episode
    # 2) PLR per episode
    # 3) Action distribution (Buy/Hold/Sell across all episodes)
    # on one figure with subplots.
    # ---------------------------
    plt.figure(figsize=(12, 5))

    # (1) Total Profit
    plt.subplot(1, 3, 1)
    plt.plot(metrics_df["Episode"], metrics_df["Total Profit"], color="blue", marker="o")
    plt.title("Total Profit per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Profit ($)")
    
    # (2) PLR (Profit-Loss Ratio)
    plt.subplot(1, 3, 2)
    plt.plot(metrics_df["Episode"], metrics_df["Profit-Loss Ratio (PLR)"],
             color="green", marker="o")
    plt.title("PLR per Episode")
    plt.xlabel("Episode")
    plt.ylabel("PLR")

    # (3) Action Distribution
    plt.subplot(1, 3, 3)
    plt.bar(action_counts.keys(), action_counts.values(), 
            color=["green", "blue", "red"])
    plt.title("Action Distribution (All Episodes)")
    plt.xlabel("Action")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()
