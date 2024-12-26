import numpy as np
import pandas as pd
from gym import Env
from gym.spaces import Discrete, Box

class CryptoEnvRL(Env):
    def __init__(self, price_series, volume_series, window_size=14):
        super(CryptoEnvRL, self).__init__()
        self.price_series = price_series
        self.volume_series = volume_series
        self.window_size = window_size 
        #self.symbol = symbol
        self.state_size = 10  # State: RSI, SMA, Momentum, Volatility, Volume, Stock Owned, Buy Price, Current Price, VWAP
        self.action_space = Discrete(3)  # Actions: Buy (0), Hold (1), Sell (2)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.state_size,), dtype=np.float32)
        self.last_action = None
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

        self.momentum = prices - np.roll(prices, 5)
        self.momentum[:5] = 0
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
        self.macd = (self.macd -np.mean(self.macd)) / np.std(self.macd)

        # Normalize Volume + Close Price
        self.volume_series = (volumes - np.mean(volumes)) / np.std(volumes)
        # self.price_series = (prices - np.mean(prices)) / np.std(prices)

    def _calculate_stop_loss(self, current_price):
        # Calculate a 10% stop-loss threshold
        if self.stock_owned and ((current_price - self.buy_price) / self.buy_price < -0.10):
            return True
        return False
    
    # def decide_trade(self, rsi, sma_current, sma_previous, macd):
    #     """Determine the trading action based on combined indicators."""
    #     # Buy conditions
    #     if rsi < 30 and macd > 0 and sma_current > sma_previous:
    #         return 'buy'
    #     # Sell conditions
    #     elif rsi > 70 or (macd < 0 and sma_current < sma_previous):
    #         return 'sell'
    #     # Hold conditions
    #     else:
    #         return 'hold'


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

        recent_volatility = np.std(self.price_series[max(0, self.t - 10): self.t])
        recent_trend = 0
        if self.t > 5:  # Ensure enough history is available
                recent_trend = np.mean(self.price_series[self.t - 5:self.t]) - self.price_series[self.t - 6]

        # if self.last_action == action:
        #     reward -= 0.10  # Penalize repeated actions 

        #         # Save current action for the next step
        # self.last_action = action


        if action == 0:  # Buy
            if self.stock_owned == 0:
                self.stock_owned = 1
                self.buy_price = current_price
                reward = 0.2 # small positive for buying in time.
            else:
                reward = -1 # Penalty for invalid buy

        elif action == 1:  # Hold
              reward = -0.01
              if self.stock_owned == 1:
                holding_profit = current_price - self.buy_price
                reward += 0.05 * holding_profit
              else:
                reward = -0.05

              if recent_trend > 0:
                 reward += 0.2 * recent_trend

              if recent_volatility < 0.02:  # Low volatility threshold
                 reward += 0.1  # reward for holding in calm conditions
              else:
                 reward -= 0.01

        elif action == 2:  # Sell
            if self.stock_owned == 1:
                if self._calculate_stop_loss(current_price):
                   reward = -0.40  # Penalty for hitting stop loss
 
                profit = current_price - self.buy_price
                reward +=  profit 
            
                #reward = np.clip(reward, -100, 100)
                # remove this as it was added after
                # if self.macd[self.t] > 0:
                #    reward += 0.1 * profit
                # elif self.macd[self.t] < 0:
                #    reward -= 0.1 * abs(profit)

                if self.volume_series[self.t] > np.mean(self.volume_series):
                   reward += 0.05 * profit  # Higher reward for selling above VWAP

                if profit < 0:
                    reward += 2 * profit  # Amplify penalty for losses
                elif profit < 50:  # Trades with low profit
                    reward -= 10
                
                if profit > 0:
                   reward += 1

                self.stock_owned = 0
                self.buy_price = 0
            else:
                reward = -1  # Penalty for invalid sell

     
        reward = np.clip(reward, -50, 100)

        self.t += 1
        done = self.t >= len(self.price_series) - 1
        return self._get_observation(), reward, done, {}
