import numpy as np
import pandas as pd
from gym import Env
from gym.spaces import Discrete, Box


class CryptoEnvRL(Env):
    def __init__(self, price_series, window_size=7):
        """
        Initialize the cryptocurrency trading environment.

        Args:
            price_series (np.ndarray): Array of prices (e.g., closing prices).
            window_size (int): Number of past time steps included in the state.
        """
        super(CryptoEnvRL, self).__init__()

        # Price series and window size
        self.price_series = price_series
        self.window_size = window_size

        # State: RSI, SMA, Momentum, Volatility, Stock Owned, Buy Price, Current Price
        self.state_size = 7

        # Define action space: 0 (Buy), 1 (Hold), 2 (Sell)
        self.action_space = Discrete(3)

        # Observation space: Box to represent state variables
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.state_size,), dtype=np.float32)

        # Internal variables
        self.t = 0
        self.stock_owned = 0
        self.buy_price = 0
        self.total_reward = 0

        # Precompute indicators
        self._calculate_indicators()

    def _calculate_indicators(self):
        """
        Calculate technical indicators for the price series.
        """
        prices = self.price_series

        # Relative Strength Index (RSI)
        delta = np.diff(prices, prepend=prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=14, min_periods=1).mean()
        avg_loss = pd.Series(loss).rolling(window=14, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        self.rsi = 100 - (100 / (1 + rs))

        # Simple Moving Average (SMA)
        self.sma = pd.Series(prices).rolling(window=14, min_periods=1).mean().values

        # Momentum
        self.momentum = prices - np.roll(prices, 5)
        self.momentum[:5] = 0  # Handle first few undefined values

        # Volatility (standard deviation)
        self.volatility = pd.Series(prices).rolling(window=14, min_periods=1).std().values

    def reset(self):
        """
        Reset the environment to its initial state.
        """
        self.t = 0
        self.stock_owned = 0  # Whether the agent owns stock (0 or 1)
        self.buy_price = 0    # Price at which the stock was bought
        self.total_reward = 0

        # Return the initial observation
        return self._get_observation()

    def _get_observation(self):
        """
        Construct the current state as an array of indicators and variables.
        """
        return np.array([
            self.price_series[self.t],        # Current price
            self.sma[self.t],                # Simple Moving Average
            self.rsi[self.t],                # Relative Strength Index
            self.momentum[self.t],           # Momentum
            self.volatility[self.t],         # Volatility
            self.stock_owned,                # Whether stock is owned
            self.buy_price                   # Price at which stock was bought
        ], dtype=np.float32)

    def _transition_fn(self, action):
        """
        Handle state transitions and reward calculation.

        Args:
            action (int): Action taken by the agent.

        Returns:
            tuple:
                reward (float): The reward for the action.
                done (bool): Whether the episode has ended.
        """
        reward = -0.01  # Default penalty for time spent
        current_price = self.price_series[self.t]

        if action == 0:  # Buy
            if self.stock_owned == 0:  # Only buy if no stock is currently owned
                self.stock_owned = 1
                self.buy_price = current_price
            else:
                reward = -0.05  # Penalty for buying when already owning

        elif action == 1:  # Hold
            reward = -0.005  # Small penalty to discourage excessive holding

        elif action == 2:  # Sell
            if self.stock_owned == 1:  # Only sell if stock is owned
                reward = current_price - self.buy_price
                self.stock_owned = 0
                self.buy_price = 0
            else:
                reward = -1  # Penalty for trying to sell without owning

        # Update time step
        self.t += 1
        done = self.t >= len(self.price_series) - 1  # Done if at the end of the series

        # Accumulate rewards
        self.total_reward += reward

        return reward, done

    def step(self, action):
        """
        Execute a step in the environment.

        Args:
            action (int): The action taken by the agent.

        Returns:
            tuple:
                next_state (np.ndarray): The next state after taking the action.
                reward (float): The reward received for taking the action.
                done (bool): Whether the episode has ended.
                info (dict): Additional information for debugging (optional).
        """
        reward, done = self._transition_fn(action)
        next_state = self._get_observation()
        return next_state, reward, done, {}
