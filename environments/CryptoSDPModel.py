from collections import namedtuple
import numpy as np

class CryptoSDPModel:
    def __init__(self, price_series, initial_cash=1000, sma_window=5, rsi_window=14):
        """
        Initialize the Sequential Decision Process model.

        Args:
            price_series (list): List of historical prices.
            initial_cash (float): Initial amount of cash available.
            sma_window (int): Window size for Simple Moving Average.
            rsi_window (int): Window size for Relative Strength Index.
        """
        self.price_series = price_series
        self.initial_cash = initial_cash
        self.sma_window = sma_window
        self.rsi_window = rsi_window
        self.t = 0

        # Define the state as a named tuple
        self.State = namedtuple(
            "State",
            ["price", "sma", "rsi", "momentum", "volatility", "stock_owned", "cash", "buy_price"]
        )
        # Define the decision as a named tuple
        self.Decision = namedtuple("Decision", ["action"])

        # Initialize state
        self.state = self.reset()

    def reset(self):
        """
        Reset the model to its initial state.

        Returns:
            State: Initial state.
        """
        self.t = 0
        self.state = self.State(
            price=self.price_series[0],
            sma=self._calculate_sma(),
            rsi=self._calculate_rsi(),
            momentum=0,
            volatility=0,
            stock_owned=0,
            cash=self.initial_cash,
            buy_price=0  # No stock owned initially
        )
        return self.state

    def step(self, decision):
        """
        Perform a single step in the sequential decision process.

        Args:
            decision (namedtuple): The decision taken by the agent.

        Returns:
            State: The next state after taking the decision.
            float: The reward for the decision.
            bool: Whether the episode is done.
            dict: Additional information.
        """
        # Generate exogenous information
        exog_info = self.exog_info_fn()

        # Compute reward
        reward = self.objective_fn(decision, exog_info)

        # Update state
        exog_info.update(self.transition_fn(decision, exog_info))
        self.state = self.build_state(exog_info)

        # Increment time
        self.t += 1
        done = self.t >= len(self.price_series) - 1

        return self.state, reward, done, {}

    def exog_info_fn(self):
        """
        Generate exogenous information for the next state.

        Returns:
            dict: Exogenous information (e.g., next price).
        """
        next_price = self.price_series[self.t + 1] if self.t < len(self.price_series) - 1 else self.price_series[-1]
        return {"next_price": next_price}

    def transition_fn(self, decision, exog_info):
        """
        Compute state transitions based on the decision and exogenous information.

        Args:
            decision (namedtuple): The decision taken by the agent.
            exog_info (dict): Exogenous information (e.g., next price).

        Returns:
            dict: Updated state values.
        """
        current_price = self.state.price
        next_price = exog_info["next_price"]

        if decision.action == 0:  # Buy
            if self.state.cash >= current_price:
                return {
                    "cash": self.state.cash - current_price,
                    "stock_owned": self.state.stock_owned + 1,
                    "buy_price": current_price
                }
            else:
                return {"cash": self.state.cash, "stock_owned": self.state.stock_owned, "buy_price": self.state.buy_price}

        elif decision.action == 2:  # Sell
            if self.state.stock_owned > 0:
                return {
                    "cash": self.state.cash + self.state.stock_owned * current_price,
                    "stock_owned": 0,
                    "buy_price": 0  # Reset buy price after selling
                }
            else:
                return {"cash": self.state.cash, "stock_owned": self.state.stock_owned, "buy_price": self.state.buy_price}

        else:  # Hold
            return {"cash": self.state.cash, "stock_owned": self.state.stock_owned, "buy_price": self.state.buy_price}

    def objective_fn(self, decision, exog_info):
        """
        Compute the reward based on the current decision and exogenous information.

        Args:
            decision (namedtuple): The decision taken by the agent.
            exog_info (dict): Exogenous information (e.g., next price).

        Returns:
            float: The reward for the decision.
        """
        current_price = self.state.price
        next_price = exog_info["next_price"]
        reward = 0

        if decision.action == 0:  # Buy
            reward = -0.01  # Small penalty to discourage unnecessary buys
        elif decision.action == 2:  # Sell
            if self.state.stock_owned > 0:
                reward = self.state.stock_owned * (current_price - self.state.buy_price)
            else:
                reward = -1  # Penalty for selling with no stock owned
        elif decision.action == 1:  # Hold
            reward = -0.01  # Small penalty for inactivity

        print(f"Action: {decision.action}, Reward: {reward}, Current Price: {current_price}, Next Price: {next_price}")
        return reward

    def build_decision(self, info):
        """
        Builds a decision object using the provided information.

        Args:
            info (dict): A dictionary containing decision details.

        Returns:
            Decision: The constructed decision object.
        """
        return self.Decision(*[info[k] for k in self.Decision._fields])

    def _calculate_sma(self):
        return np.mean(self.price_series[max(0, self.t - self.sma_window + 1): self.t + 1])

    def _calculate_rsi(self):
        # Simplified RSI calculation
        if self.t < self.rsi_window:
            return 50  # Neutral RSI
        gains = [self.price_series[i + 1] - self.price_series[i] for i in range(self.t - self.rsi_window, self.t) if self.price_series[i + 1] > self.price_series[i]]
        losses = [self.price_series[i] - self.price_series[i + 1] for i in range(self.t - self.rsi_window, self.t) if self.price_series[i + 1] < self.price_series[i]]
        avg_gain = np.mean(gains) if gains else 0
        avg_loss = np.mean(losses) if losses else 0
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    def is_finished(self):
        """
        Check if the model has reached the end of the price series.

        Returns:
            bool: True if the episode is finished, False otherwise.
        """
        return self.t >= len(self.price_series) - 1

    def build_state(self, info):
        """
        Build a new state object.

        Args:
            info (dict): State information.

        Returns:
            State: New state object.
        """
        return self.State(
            price=info.get("next_price", self.state.price),
            sma=self._calculate_sma(),
            rsi=self._calculate_rsi(),
            momentum=info.get("momentum", 0),
            volatility=info.get("volatility", 0),
            stock_owned=info.get("stock_owned", self.state.stock_owned),
            cash=info.get("cash", self.state.cash),
            buy_price=info.get("buy_price", self.state.buy_price)
        )
