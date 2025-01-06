import numpy as np
import pandas as pd
from gym import Env
import matplotlib.pyplot as plt
from gym.spaces import Discrete, Box

class CryptoEnvRL(Env):
    def __init__(self, price_series, volume_series, asset_type="BTC", window_size=14, initial_cash=25000.0):
        super(CryptoEnvRL, self).__init__()
        self.price_series = price_series
        self.volume_series = volume_series
        self.asset_type = asset_type  # BTC or ETH
        self.window_size = window_size 
        self.state_size = 11  # State: RSI, SMA, Momentum, Volatility, Price, Stock Owned, Buy Price, Current Price, VWAP, Cash
        self.action_space = Discrete(3)  # Actions: Buy (0), Hold (1), Sell (2)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.state_size,), dtype=np.float32)
        self.last_action = None

        if asset_type == "BTC":
            self.low_vol_threshold = 200
            self.high_vol_threshold = 1300
            self.rsi_oversold = 40
            self.rsi_overbought = 60
        elif asset_type == "ETH":
            self.low_vol_threshold = 20
            self.high_vol_threshold = 120
            self.rsi_oversold = 20
            self.rsi_overbought = 80
        
        self.initial_cash = initial_cash
        self.reset()

    def _calculate_indicators(self):
        prices = self.price_series
        volumes = self.volume_series

        delta = np.diff(prices, prepend=prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=14, min_periods=1).mean()
        avg_loss = pd.Series(loss).rolling(window=14, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        self.rsi = 100 - (100 / (1 + rs))
        self.rsi = (self.rsi - np.mean(self.rsi)) / np.std(self.rsi)

        self.sma = pd.Series(prices).rolling(window=14, min_periods=1).mean().values
        self.sma = (self.sma - np.mean(self.sma)) / np.std(self.sma)

        window = 14
        self.momentum = prices - np.roll(prices, window)
        self.momentum[:window] = 0
        self.momentum = (self.momentum - np.mean(self.momentum)) / np.std(self.momentum)

    
        self.volatility = pd.Series(prices).rolling(window=14, min_periods=1).std().values
        self.volatility = (self.volatility - np.mean(self.volatility)) /  np.std(self.volatility)

        
        # VWAP
        cumulative_price_volume = np.cumsum(prices * volumes)
        cumulative_volume = np.cumsum(volumes)
        self.vwap = cumulative_price_volume / (cumulative_volume + 1e-9)  # Avoid division by zero

        self.vwap = (self.vwap - np.mean(self.vwap)) / np.std(self.vwap)

        #MACD
        ema_12 = pd.Series(prices).ewm(span=12, adjust=False).mean()
        ema_26 = pd.Series(prices).ewm(span=26, adjust=False).mean()
        self.macd = ema_12 - ema_26
        # signal_line = self.macd.ewm(span=9, adjust=False).mean()

        self.macd = (self.macd -np.mean(self.macd)) / np.std(self.macd)

        # Normalize Volume + Close Price
        self.volume_series = (volumes - np.mean(volumes)) / np.std(volumes)
        #self.price_series = (prices - np.mean(prices)) / np.std(prices)

    def exog_info_fn(self):
        """
        Return a dictionary of indicators at current time step self.t.
        """
        return {
            "rsi": self.rsi[self.t],
            "sma": self.sma[self.t],
            "momentum": self.momentum[self.t],
            "volatility": self.volatility[self.t],
            "vwap": self.vwap[self.t],
            "macd": self.macd[self.t],
        }

    def _calculate_stop_loss(self, current_price):
        # Calculate a 10% stop-loss threshold
        if self.stock_owned and ((current_price - self.buy_price) / self.buy_price < -0.10):
            return True
        return False
    
    def _get_observation(self):
        state = np.array([
            self.rsi[self.t],
            self.sma[self.t],
            self.momentum[self.t],
            self.volatility[self.t],
            self.stock_owned,
            self.buy_price,
            self.price_series[self.t],
            self.volume_series[self.t],
            self.cash,
            self.vwap[self.t],        
            self.macd[self.t],      
        ], dtype=np.float32)
        state = np.nan_to_num(state)
        return (state - np.mean(state)) / (np.std(state) + 1e-8)

    def reset(self):
        # Reset time step
        self.t = 0

        # Reset stock-related variables
        self.stock_owned = 0
        self.buy_price = 0

        self.cash = self.initial_cash

        # Reset cumulative metrics
        self.total_reward = 0
        self.total_profit = 0
       

        # Recalculate indicators (if necessary)
        self._calculate_indicators()

        # Return the initial observation
        return self._get_observation()


    def step(self, action):
        reward = -0.01  # Default time penalty
        current_price = self.price_series[self.t]
        # rsi = self.rsi[self.t]
        # sma_current = self.sma[self.t]
        # sma_previous = self.sma[self.t - 1] if self.t > 0 else self.sma[self.t]
        # macd = self.macd[self.t]

        # if action == 'auto':  # Allow automatic decision-making
        #     action = self.decide_trade(rsi, sma_current, sma_previous, macd)

        # recent_volatility = np.std(self.price_series[max(0, self.t - 10): self.t])
        # recent_trend = 0
        # if self.t > 5:  # Ensure enough history is available
        #         recent_trend = np.mean(self.price_series[self.t - 5:self.t]) - self.price_series[self.t - 6]

        # if self.last_action == action:
        #    reward -= 0.10  # Penalize repeated actions 

        # # #         # Save current action for the next step
        # self.last_action = action
        rsi = self.rsi[self.t]
        sma = self.sma[self.t]
        mom = self.momentum[self.t]
        vol = self.volatility[self.t]
        vwap = self.vwap[self.t]
        macd = self.macd[self.t]


        oversold = (rsi < self.rsi_oversold)
        overbought = (rsi > self.rsi_overbought)
        price_below_sma = (current_price < sma)
        bullish_momentum = (mom > 10)
        bullish_macd = (macd > 5)
        low_vol = (vol < self.low_vol_threshold)
        high_vol = (vol > self.high_vol_threshold)

        if action == 0:  # Buy
            if self.stock_owned == 0:
                if self.cash >= current_price:
                    self.stock_owned = 1
                    self.buy_price = current_price
                    self.cash -= current_price  # deduct cost of 1 share

                    if oversold or price_below_sma:
                        reward += 0.05
                    if bullish_momentum or bullish_macd:
                        reward += 0.05
                else:
                    reward -= 0.5
            else:
                reward = -1

        elif action == 1:  # Hold
            reward = -1
            if self.stock_owned and bullish_momentum:
                reward += 0.01

            if low_vol:
                reward += 0.02
            if high_vol:
                reward -= 0.05

        elif action == 2:  # Sell
            if self.stock_owned == 1:
                profit = current_price - self.buy_price
                reward += profit

                self.cash += current_price
                self.stock_owned = 0
                self.buy_price = 0.0

                if overbought:
                    reward += 0.02
                if bullish_momentum or bullish_macd:
                    reward -= 0.01
            else:
                reward = -1
        # if action == 0:  # Buy
        #     if self.stock_owned == 0:
        #         # Check if we have enough cash
        #         if self.cash >= current_price:
        #             self.stock_owned = 1
        #             self.buy_price = current_price
        #             self.cash -= current_price
        #         else:
        #             # Not enough cash
        #             reward -= 1

        # elif action == 1:  # Hold
        #     reward -= 1  # or smaller/larger penalty if you like

        # elif action == 2:  # Sell
        #     if self.stock_owned == 1:
        #         # Realize profit or loss
        #         profit = current_price - self.buy_price
        #         reward += profit
        #         self.cash += current_price
        #         self.stock_owned = 0
        #         self.buy_price = 0.0
        #     else:
        #         reward -= 1 
            
        reward = np.clip(reward, -50, 100)

        self.t += 1
        done = self.t >= len(self.price_series) - 1
        return self._get_observation(), reward, done, {}
    
    @property
    def portfolio_value(self):
        """
        cash + (stock_owned * current_price).
        """
        current_price = self.price_series[self.t]
        return self.cash + (self.stock_owned * current_price)
    
  
