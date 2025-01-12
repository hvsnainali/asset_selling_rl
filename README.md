# **Optimizing Cryptocurrency Trading Decisions Using Deep Q-Networks**

## **Overview**
This project implements a **Deep Q-Network (DQN)**-based reinforcement learning framework for optimizing cryptocurrency trading decisions. The study evaluates the performance of the DQN model in comparison to a **Sequential Decision Problem (SDP)** framework and a **Simple Policy** heuristic. It demonstrates the superiority of DQN in volatile market conditions through dynamic learning and adaptive decision-making.

---

## **Features**
- Reinforcement learning framework tailored for cryptocurrency trading.
- Implementation of DQN with:
  - Epsilon-greedy exploration-exploitation strategy.
  - Replay buffer and target network for stable learning.
- Comparison with SDP framework (static thresholds) and Simple Policy.
- Comprehensive evaluation on historical BTC and ETH price data using technical indicators like RSI, SMA, MACD, and more.
- Performance analysis with metrics such as profit-to-loss ratio, Sharpe ratio, and portfolio value.

---

## **Project Structure**
├── CryptoEnvRL.py # Custom environment for RL agents
├── DQNetwork.py # Implementation of the DQN

├── CryptoSDPModel.py # SDP-based trading framework
├── RLAgent.py # RL agent controlling DQN actions 

├── main.py # Main script for training and testing models 
├── test.py # Script for evaluating trained models 


---

---



## **Key Insights**
1. **DQN Model Superiority**:
   - Achieved the highest profitability and risk-adjusted returns.
   - Demonstrated dynamic adaptability using epsilon-greedy exploration.
2. **SDP Framework**:
   - Performed moderately but struggled during volatile conditions due to static thresholds.
3. **Simple Policy**:
   - Computationally efficient but insufficient for complex, volatile market environments.

---

## **Future Work**
- **Enhancements to DQN**:
  - More macroeconomic data, sentiment analysis, and advanced RL techniques like PPO or Double DQN.
- **Realistic Constraints**:
  - Incorporate transaction costs, slippage, and latency into the trading environment.
- **Broader Evaluation**:
  - Testing  on additional cryptocurrencies and across different market regimes (e.g., bull vs. bear markets).
