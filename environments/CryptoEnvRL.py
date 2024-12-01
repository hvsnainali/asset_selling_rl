import numpy as np
import pandas as pd
from gym import Env
from gym.spaces import Discrete, Box

class CryptoEnvRL(Env):
    def __init__(self, price_series, window_size=7):
        super(CryptoEnvRL, self).__init__()
        self.price_series = price_series
        self.window_size = window_size
        self.state_size = 7  # State: RSI, SMA, Momentum, Volatility, Stock Owned, Buy Price, Current Price
        self.action_space = Discrete(3)  # Actions: Buy (0), Hold (1), Sell (2)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.state_size,), dtype=np.float32)
        self.reset()

    def _calculate_indicators(self):
        prices = self.price_series
        delta = np.diff(prices, prepend=prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=14, min_periods=1).mean()
        avg_loss = pd.Series(loss).rolling(window=14, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        self.rsi = 100 - (100 / (1 + rs))
        self.sma = pd.Series(prices).rolling(window=14, min_periods=1).mean().values
        self.momentum = prices - np.roll(prices, 5)
        self.momentum[:5] = 0
        self.volatility = pd.Series(prices).rolling(window=14, min_periods=1).std().values

    def _get_observation(self):
        state = np.array([
            self.rsi[self.t],
            self.sma[self.t],
            self.momentum[self.t],
            self.volatility[self.t],
            self.stock_owned,
            self.buy_price,
            self.price_series[self.t]
        ], dtype=np.float32)
        state = np.nan_to_num(state)
        return (state - np.mean(state)) / (np.std(state) + 1e-8)

    def reset(self):
        self.t = 0
        self.stock_owned = 0
        self.buy_price = 0
        self.total_reward = 0
        self._calculate_indicators()
        return self._get_observation()

    def step(self, action):
        reward = -0.01  # Default time penalty
        if action == 0:  # Buy
            if self.stock_owned == 0:
                self.stock_owned = 1
                self.buy_price = self.price_series[self.t]
            else:
                reward = -0.1  # Penalty for invalid buy
        elif action == 1:  # Hold
            reward = -0.05
        elif action == 2:  # Sell
            if self.stock_owned == 1:
                reward = self.price_series[self.t] - self.buy_price
                reward = np.clip(reward, -100, 100)
                self.stock_owned = 0
                self.buy_price = 0
            else:
                reward = -0.1  # Penalty for invalid sell
        self.t += 1
        done = self.t >= len(self.price_series) - 1
        return self._get_observation(), reward, done, {}
