import gym
from gym import spaces
import numpy as np
from .CryptoSDPModel import CryptoSDPModel

class CryptoEnvRL(gym.Env):
    def __init__(self, price_series):
        super(CryptoEnvRL, self).__init__()
        self.model = CryptoSDPModel(price_series)
        self.action_space = spaces.Discrete(3)  # Buy, Hold, Sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

    def reset(self):
        self.model.reset()
        return np.array(self._get_observation(), dtype=np.float32)

    def _get_observation(self):
        state = self.model.state
        return [state.price, state.sma, state.rsi, state.momentum, state.volatility, state.stock_owned, state.cash]

    def step(self, action):
        decision = self.model.build_decision({"action": action})
        state_t_plus_1 = self.model.step(decision)
        #reward = self.model.objective_fn(decision, self.model.exog_info_fn(decision))
        reward = self.model.objective_fn(decision, self.model.exog_info_fn())
        done = self.model.is_finished()
        return np.array(self._get_observation(), dtype=np.float32), reward, done, {}

    def render(self):
        print(f"State: {self._get_observation()}")
