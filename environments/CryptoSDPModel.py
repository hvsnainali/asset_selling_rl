import numpy as np
import pandas as pd


class CryptoSDPModel:
    def __init__(self, price_series,volume_series):
        """
        Initialize the Sequential Decision Process (SDP) model for cryptocurrency trading.

        Args:
            price_series (np.ndarray): Array of cryptocurrency prices.
        """
        self.price_series = price_series 
        self.volume_series = volume_series 


        self.t = 0  # Time step, starts at the beginning
        self.state = None  # Current state
        self.stock_owned = 0  # Whether stock is owned (0: No, 1: Yes)
        self.buy_price = 0  # The price at which the stock was bought
        self.total_reward = 0  # Cumulative reward

        # Initialize exogenous data
        self.sma = self.calculate_sma(window=14)
        self.rsi = self.calculate_rsi(window=14)
        self.volatility = self.calculate_volatility(window=14)
        self.momentum = self.calculate_momentum(window=14)
        self.vwap = self.calculate_vwap()
        self.macd = self.calculate_macd()
    
    # --------------------------------------------------------------------------------------------
    
    #                   INDICATOR CALCULATION METHODS
    
    # --------------------------------------------------------------------------------------------
    
    def calculate_sma(self, window=14):
        """
        Calculate the Simple Moving Average (SMA) for the price series.

        Args:
            window (int): Window size for SMA calculation.

        Returns:
            np.ndarray: SMA values for the series.
        """
        sma = np.convolve(self.price_series, np.ones(window), 'valid') / window
        return np.concatenate((np.zeros(window - 1), sma))  # Pad to match length

    def calculate_rsi(self, window=14):
        """
        Calculate the Relative Strength Index (RSI) for the price series.

        Args:
            window (int): Window size for RSI calculation.

        Returns:
            np.ndarray: RSI values for the series.
        """
        deltas = np.diff(self.price_series, prepend=self.price_series[0])
        gains = np.maximum(deltas, 0)
        losses = -np.minimum(deltas, 0)
        avg_gain = np.convolve(gains, np.ones(window), 'valid') / window
        avg_loss = np.convolve(losses, np.ones(window), 'valid') / window
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
        rsi = 100 - (100 / (1 + rs))
        return np.concatenate((np.zeros(window - 1), rsi))  # Pad to match length

    def calculate_volatility(self, window=14):
        """
        Calculate the rolling volatility (standard deviation) for the price series.

        Args:
            window (int): Window size for volatility calculation.

        Returns:
            np.ndarray: Volatility values for the series.
        """
        volatility = np.array([
            np.std(self.price_series[max(0, i - window + 1):i + 1])
            for i in range(len(self.price_series))
        ])
        return volatility

    def calculate_momentum(self, window=14):
        """
        Calculate the momentum of the price series.

        Momentum is the difference between the current price and the price `window` steps ago.

        Args:
            window (int): Window size for momentum calculation.

        Returns:
            np.ndarray: Momentum values for the series.
        """
        momentum = np.zeros_like(self.price_series)
        for i in range(window, len(self.price_series)):
            momentum[i] = self.price_series[i] - self.price_series[i - window]
        return momentum
    def calculate_vwap(self):
        """
        Calculate Volume Weighted Average Price (VWAP).
        Pseudocode:
            cumulative_price_volume = cumsum(price * volume)
            cumulative_volume       = cumsum(volume)
            vwap_raw                = cumulative_price_volume / (cumulative_volume + 1e-9)
        Returns:
            np.ndarray: VWAP values (one per timestep).
        """
        cumulative_price_volume = np.cumsum(self.price_series * self.volume_series)
        cumulative_volume = np.cumsum(self.volume_series)
        vwap_raw = cumulative_price_volume / (cumulative_volume + 1e-9)
        return vwap_raw

    def calculate_macd(self):
        """
        Calculate MACD as the difference of 12-day EMA and 26-day EMA.
        Args:
            None (uses self.price_series).
        Returns:
            np.ndarray: MACD values for each timestep.
        """
        prices_series = pd.Series(self.price_series)
        ema_12 = prices_series.ewm(span=12, adjust=False).mean()
        ema_26 = prices_series.ewm(span=26, adjust=False).mean()
        macd_raw = ema_12 - ema_26
        return macd_raw.values

    # -------------------------------------------------------------------------------------------

    #                    RESET / EXOG / OBJ LOGIC

    # -------------------------------------------------------------------------------------------


    def reset(self):
        """
        Reset the model to its initial state.
        """
        self.t = 0
        self.stock_owned = 0
        self.buy_price = 0
        self.total_reward = 0

    def build_state(self):
        """
        Construct the state representation.

        Returns:
            np.ndarray: Current state of the environment.
        """
        return np.array([
            self.price_series[self.t],  # Current price
            self.sma[self.t],  # Simple Moving Average
            self.rsi[self.t],  # Relative Strength Index
            self.volatility[self.t],  # Volatility
            self.momentum[self.t],
            self.vwap[self.t],
            self.macd[self.t],  
            self.stock_owned,  # Whether stock is owned
            self.buy_price,  # Price at which stock was bought
            self.total_reward  # Cumulative reward
        ], dtype=np.float32)

    def exog_info_fn(self):
        """
        Generate exogenous information for the next state.

        Returns:
            dict: Dictionary containing SMA, RSI, volatility, and momentum.
        """
        return {
            "sma": self.sma[self.t],
            "rsi": self.rsi[self.t],
            "volatility": self.volatility[self.t],
            "momentum": self.momentum[self.t],
            "vwap": self.vwap[self.t],        
            "macd": self.macd[self.t]
        }

    def transition_fn(self, action):
        """
        Execute one step in the model based on the action taken by the agent.

        Args:
            action (int): Action taken by the agent:
                0: Buy
                1: Hold
                2: Sell

        Returns:
            tuple:
                reward (float): Reward for the action.
                done (bool): Whether the episode has ended.
        """
        reward = -0.01  # Default penalty for time spent (encourages timely decisions)
        current_price = self.price_series[self.t]

        if action == 0:  # Buy
            if self.stock_owned == 0:  # Only buy if no stock is currently owned
                self.stock_owned = 1
                self.buy_price = current_price
            else:
                reward = -0.05  # Penalty for trying to buy when stock is already owned

        elif action == 1:  # Hold
            reward = -0.05  # Small penalty to discourage unnecessary holding

        elif action == 2:  # Sell
            if self.stock_owned == 1:  # Can only sell if stock is owned
                reward = current_price - self.buy_price  # Profit or loss from selling
                self.stock_owned = 0
                self.buy_price = 0
            else:
                reward = -1  # Heavy penalty for trying to sell without owning

        # Transition to the next time step
        self.t += 1
        done = self.t >= len(self.price_series) - 1  # Episode ends when we reach the last price

        # Accumulate rewards
        self.total_reward += reward

        next_state = self.build_state()

        return next_state, reward, done

    def objective_fn(self):
        """
        Compute the cumulative reward (objective function).

        Returns:
            float: Total reward achieved so far.
        """
        return self.total_reward
    
    # -------------------------------------------------------------------------------------------

    #                   HEURISTIC POLICY

    # -------------------------------------------------------------------------------------------

def advanced_heuristic_policy_sdp(model):
        model.reset()
        done = False
        actions_taken = []
        rewards_collected = []

        while not done:
            info = model.exog_info_fn()
            price = model.price_series[model.t]

            
            if (info["rsi"] < 40) and (price < info["sma"]):
                action = 0  # Buy
            elif (info["rsi"] > 60) or (price > info["sma"] * 1.05):
                action = 2  # Sell
            else:
                action = 1  # Hold

            reward, done = model.transition_fn(action)
            actions_taken.append(action)
            rewards_collected.append(reward)

        total_reward = model.objective_fn()
        return total_reward, actions_taken, rewards_collected
