import gymnasium as gym
from gymnasium import spaces
import numpy as np
from tqdm import tqdm

class AmadeusTradingEnv(gym.Env):
    """
    Amadeus Cartridge: Trader (IBN-5100 Theme)
    Gymnasium Environment for Reinforcement Learning
    """
    metadata = {'render_modes': ['human']}

    # --- Environment Constants ---
    TICKS_PER_DAY = 12
    TRADING_DAYS_PER_YEAR = 252
    ACTIONS_PER_ASSET = 4
    SHARE_MULTIPLIER = 10
    MAX_ACTIVE_OPTIONS = 4
    RISK_FREE_RATE = 0.05
    OPTION_DURATION_DAYS = 7
    OBSERVATION_SIZE = 64  # Padded to 64 for optimal GPU tensor alignment

    def __init__(self, render_mode=None):
        super(AmadeusTradingEnv, self).__init__()
        self.render_mode = render_mode

        # 14 Discrete Actions:
        # 0: Hold
        # 1: Buy 10 AMDS, 2: Sell 10 AMDS, 3: Buy AMDS Call, 4: Buy AMDS Put
        # 5: Buy 10 SERN, 6: Sell 10 SERN, 7: Buy SERN Call, 8: Buy SERN Put
        # 9: Buy 10 D-ML, 10: Sell 10 D-ML, 11: Buy D-ML Call, 12: Buy D-ML Put
        # 13: Close All Active Options (Take Profit / Cut Loss)
        self.action_space = spaces.Discrete(14)

        # Observation Space: 64 flattened continuous/discrete features
        # [0] Cash
        # [1] Economy (0=STABLE, 1=BOOM, 2=BUST)
        # [2-7] Stocks: (Price, Owned) for AMDS, SERN, D-ML
        # [8-27] Options: 4 active slots * 5 features per slot
        #        Slot features: (is_active, type_call_0_put_1, stock_idx, strike, days_left)
        # [28-63] Zero padding for tensor alignment
        low = np.array([-np.inf] * self.OBSERVATION_SIZE, dtype=np.float32)
        high = np.array([np.inf] * self.OBSERVATION_SIZE, dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.tx_fee_pct = 0.001  # 0.1% of trade value (proportional, not flat)
        self.dividend_paid = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.ticks = 0
        self.days = 1
        self.dividend_paid = 0.0
        self.cash = 1000.00
        self.net_worth = 1000.00
        self.economy = 0 # 0: STABLE, 1: BOOM, 2: BUST

        self.stocks = [
            {"name": "AMDS", "price": 45.50, "owned": 0, "vol": 1.2, "cost_basis": 0.0},
            {"name": "SERN", "price": 120.00, "owned": 0, "vol": 2.5, "cost_basis": 0.0},
            {"name": "D-ML", "price": 15.20, "owned": 0, "vol": 0.8, "cost_basis": 0.0}
        ]

        self.active_options = []

        return self._get_obs(), {}

    def step(self, action):
        prev_net_worth = self.net_worth

        # 1. Execute agent action
        self._execute_trade(action)

        # 2. Update market prices, economy, and process options/dividends
        self._update_market()

        # 3. Dense Reward: Profit/Loss generated in this specific step
        reward = self.net_worth - prev_net_worth

        # 4. End condition: True bankruptcy only — let agent experience full episodes
        terminated = bool(self.net_worth <= 0.0)

        # 5. Truncation: 1 year of trading
        truncated = bool(self.ticks >= (self.TRADING_DAYS_PER_YEAR * self.TICKS_PER_DAY))

        info = {
            "net_worth": self.net_worth,
            "days": self.days
        }

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        obs = np.zeros(self.OBSERVATION_SIZE, dtype=np.float32)
        obs[0] = self.cash
        obs[1] = float(self.economy)

        idx = 2
        for s in self.stocks:
            obs[idx] = s["price"]
            obs[idx+1] = float(s["owned"])
            idx += 2

        opt_idx = 8
        for i in range(self.MAX_ACTIVE_OPTIONS):
            if i < len(self.active_options):
                opt = self.active_options[i]
                obs[opt_idx] = 1.0 # is_active flag
                obs[opt_idx+1] = 0.0 if opt["type"] == "CALLS" else 1.0
                obs[opt_idx+2] = float(opt["stock_idx"])
                obs[opt_idx+3] = opt["strike"]
                obs[opt_idx+4] = float(opt["days_left"])
            else:
                obs[opt_idx:opt_idx+5] = 0.0
            opt_idx += 5

        return obs

    def _execute_trade(self, action):
        if action == 0:
            return # Hold position

        if action == 13: # Close All Active Options simultaneously (Take Profit / Cut Loss)
            for opt in self.active_options:
                st = self.stocks[opt["stock_idx"]]
                time_to_expiry_years = max(0.0, opt["days_left"] / self.TRADING_DAYS_PER_YEAR)
                annualized_vol = st["vol"] / self.TICKS_PER_DAY
                price_per_share = self._binomial_option_pricing(
                    S=st["price"],
                    K=opt["strike"],
                    T=time_to_expiry_years,
                    r=self.RISK_FREE_RATE,
                    sigma=annualized_vol,
                    opt_type=opt["type"]
                )
                val = price_per_share * 100 * opt["qty"]
                fee = val * self.tx_fee_pct
                self.cash += max(0.0, val - fee)
            self.active_options = []
            return

        stock_idx = (action - 1) // self.ACTIONS_PER_ASSET
        action_type = (action - 1) % self.ACTIONS_PER_ASSET

        st = self.stocks[stock_idx]

        if action_type == 0: # Buy Shares
            trade_value = st["price"] * self.SHARE_MULTIPLIER
            fee = trade_value * self.tx_fee_pct
            cost = trade_value + fee
            if self.cash >= cost:
                self.cash -= cost

                # Update Cost Basis
                if st["owned"] >= 0:
                    total_spent = (st["owned"] * st["cost_basis"]) + (self.SHARE_MULTIPLIER * st["price"])
                    st["owned"] += self.SHARE_MULTIPLIER
                    st["cost_basis"] = total_spent / st["owned"]
                else: # Covering short, might flip to long
                    st["owned"] += self.SHARE_MULTIPLIER
                    if st["owned"] > 0:
                        st["cost_basis"] = st["price"]
                    elif st["owned"] == 0:
                        st["cost_basis"] = 0.0

        elif action_type == 1: # Sell (or Short) Shares
            # Case A: Selling shares we already own
            if st["owned"] >= self.SHARE_MULTIPLIER:
                trade_value = st["price"] * self.SHARE_MULTIPLIER
                revenue = trade_value - (trade_value * self.tx_fee_pct)
                self.cash += revenue
                st["owned"] -= self.SHARE_MULTIPLIER
                if st["owned"] == 0:
                    st["cost_basis"] = 0.0

            # Case B: Short selling (Requires margin compliance)
            else:
                # Calculate current total short liability across all assets
                total_short_liability = sum([abs(s["owned"]) * s["price"] for s in self.stocks if s["owned"] < 0])
                new_short_value = st["price"] * self.SHARE_MULTIPLIER

                # Strict 150% Initial Margin Requirement
                if self.net_worth > 1.5 * (total_short_liability + new_short_value):
                    revenue = new_short_value - (new_short_value * self.tx_fee_pct)
                    self.cash += revenue

                    # Update cost basis for short
                    if st["owned"] <= 0:
                        total_shorted = (abs(st["owned"]) * st["cost_basis"]) + (self.SHARE_MULTIPLIER * st["price"])
                        st["owned"] -= self.SHARE_MULTIPLIER
                        st["cost_basis"] = total_shorted / abs(st["owned"])
                    else: # Flipping from long to short
                        st["owned"] -= self.SHARE_MULTIPLIER
                        st["cost_basis"] = st["price"]

        elif action_type == 2: # Buy Call
            self._buy_option(stock_idx, "CALLS")

        elif action_type == 3: # Buy Put
            self._buy_option(stock_idx, "PUTS")

    def _binomial_option_pricing(self, S, K, T, r, sigma, opt_type, N=10):
        """
        Calculates option price using the Cox-Ross-Rubinstein Binomial Model.
        S: Current stock price
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free interest rate
        sigma: Annualized volatility
        opt_type: 'CALLS' or 'PUTS'
        N: Number of binomial steps
        """
        if T <= 0:
            return max(0.0, S - K) if opt_type == "CALLS" else max(0.0, K - S)

        dt = T / N
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u

        # Prevent division by zero if parameters are extreme
        if u == d:
            return max(0.0, S - K) if opt_type == "CALLS" else max(0.0, K - S)

        p = (np.exp(r * dt) - d) / (u - d)

        # Initialize asset prices at maturity
        up_steps = np.arange(N, -1, -1)
        down_steps = np.arange(0, N + 1, 1)
        prices = S * (u ** up_steps) * (d ** down_steps)

        # Initialize option values at maturity
        if opt_type == "CALLS":
            values = np.maximum(0.0, prices - K)
        else:
            values = np.maximum(0.0, K - prices)

        # Step backwards through the tree
        discount = np.exp(-r * dt)
        for _ in range(N):
            values = discount * (p * values[:-1] + (1 - p) * values[1:])

        return values[0]

    def _buy_option(self, stock_idx, opt_type):
        if len(self.active_options) >= self.MAX_ACTIVE_OPTIONS:
            return

        st = self.stocks[stock_idx]
        qty = 1

        time_to_expiry_years = self.OPTION_DURATION_DAYS / self.TRADING_DAYS_PER_YEAR

        annualized_vol = st["vol"] / self.TICKS_PER_DAY

        price_per_share = self._binomial_option_pricing(
            S=st["price"],
            K=st["price"],
            T=time_to_expiry_years,
            r=self.RISK_FREE_RATE,
            sigma=annualized_vol,
            opt_type=opt_type
        )

        option_value = price_per_share * 100 * qty
        premium = option_value + (option_value * self.tx_fee_pct)

        if self.cash >= premium:
            self.cash -= premium
            self.active_options.append({
                "type": opt_type,
                "stock_idx": stock_idx,
                "strike": st["price"],
                "qty": qty,
                "days_left": self.OPTION_DURATION_DAYS
            })

    def _update_market(self):
        self.ticks += 1

        # Economy shift probability matching Lua frequency
        if self.np_random.random() < 0.035:
            self._change_economy()

        # Daily processing
        if self.ticks % self.TICKS_PER_DAY == 0:
            self.days += 1
            self._process_daily_events()

        self.net_worth = self.cash

        # Calculate new share prices and asset value
        for st in self.stocks:
            change = self._get_price_change(st["vol"], st["price"])
            st["price"] = max(1.0, st["price"] + change)
            self.net_worth += (st["price"] * st["owned"])

        for opt in self.active_options:
            st = self.stocks[opt["stock_idx"]]

            # Use binomial model to assess the ongoing market time-value of options in portfolio
            time_to_expiry_years = max(0.0, opt["days_left"] / self.TRADING_DAYS_PER_YEAR)
            annualized_vol = st["vol"] / self.TICKS_PER_DAY

            price_per_share = self._binomial_option_pricing(
                S=st["price"],
                K=opt["strike"],
                T=time_to_expiry_years,
                r=self.RISK_FREE_RATE,
                sigma=annualized_vol,
                opt_type=opt["type"]
            )

            val = price_per_share * 100 * opt["qty"]
            self.net_worth += val

        # Margin Call Liquidation: If cash is negative, forcefully sell long stocks to cover the deficit
        if self.cash < 0:
            for st in self.stocks:
                if st["owned"] > 0:
                    # Sell in blocks of self.SHARE_MULTIPLIER as long as we have shares and cash is negative
                    while st["owned"] >= self.SHARE_MULTIPLIER and self.cash < 0:
                        trade_value = st["price"] * self.SHARE_MULTIPLIER
                        fee = trade_value * self.tx_fee_pct
                        revenue = trade_value - fee
                        self.cash += revenue
                        st["owned"] -= self.SHARE_MULTIPLIER
                        if st["owned"] == 0:
                            st["cost_basis"] = 0.0

    def _get_price_change(self, volatility, price):
        r1 = self.np_random.random()
        r2 = self.np_random.random()
        normal = (r1 + r2 - 1.0) * 2.0

        if self.economy == 1:
            trend = 0.5 # BOOM
        elif self.economy == 2:
            trend = -0.5 # BUST
        else:
            trend = (r1 - 0.5) * 0.2 # STABLE

        return price * (normal + trend) * (volatility / self.TICKS_PER_DAY) * 0.01

    def _change_economy(self):
        r = self.np_random.random()
        if self.economy == 0:
            if r < 0.40:
                self.economy = 1
            elif r < 0.80:
                self.economy = 2
            else:
                self.economy = 0
        else:
            if r < 0.80:
                self.economy = 0
            elif r < 0.90:
                self.economy = 1
            else:
                self.economy = 2

    def _process_daily_events(self):
        # Options Expirations
        for i in range(len(self.active_options) - 1, -1, -1):
            opt = self.active_options[i]
            opt["days_left"] -= 1
            if opt["days_left"] <= 0:
                st = self.stocks[opt["stock_idx"]]
                payoff = 0.0

                # At expiration (days_left = 0), intrinsic value is used
                if opt["type"] == "CALLS":
                    payoff = max(0.0, st["price"] - opt["strike"]) * 100 * opt["qty"]
                else:
                    payoff = max(0.0, opt["strike"] - st["price"]) * 100 * opt["qty"]

                self.cash += payoff
                self.active_options.pop(i)

        # Dividends logic
        if self.np_random.random() < 0.05:
            s_idx = self.np_random.integers(0, 3)
            st = self.stocks[s_idx]
            # Process dividends (1% chance to pay 2-5% dividend)
            if self.np_random.random() < 0.01:
                yield_val = self.np_random.uniform(0.02, 0.05)
                payout = st["owned"] * (st["price"] * yield_val)
                self.cash += payout
                if payout < 0:
                    self.dividend_paid += abs(payout)

    def render(self):
        econ_names = ["STABLE", "BOOM", "BUST"]
        econ_str = econ_names[self.economy] if self.economy < len(econ_names) else "UNKNOWN"
        hour = (self.ticks % self.TICKS_PER_DAY) + 1
        pnl = self.net_worth - 1000.00
        pnl_sign = "+" if pnl >= 0 else ""

        print("\n" + "=" * 65)
        print(f" [AMADEUS TERMINAL] Day: {self.days:03d} | Tick: {hour:02d}/{self.TICKS_PER_DAY:02d} | Economy: {econ_str:<6} ")
        print(f" Cash: ${self.cash:>9.2f} | Net Worth: ${self.net_worth:>9.2f} ({pnl_sign}${pnl:.2f})")
        print("-" * 65)
        print(f" {'ASSET':<6} | {'PRICE':>8} | {'OWNED':>6} | {'BASIS':>8} | {'UNREAL P/L':>11}")
        print("-" * 65)
        for s in self.stocks:
            owned = s["owned"]
            price = s["price"]
            basis = s["cost_basis"]
            if owned > 0:
                unreal = (price - basis) * owned
            elif owned < 0:
                unreal = (basis - price) * abs(owned)
            else:
                unreal = 0.0
            u_sign = "+" if unreal >= 0 else ""
            u_str = f"{u_sign}${unreal:.2f}" if owned != 0 else "-"
            b_str = f"${basis:.2f}" if owned != 0 else "-"
            print(f" {s['name']:<6} | ${price:>7.2f} | {owned:>6} | {b_str:>8} | {u_str:>11}")

        if self.active_options:
            print("-" * 65)
            print(" ACTIVE OPTIONS:")
            for i, opt in enumerate(self.active_options):
                st_name = self.stocks[opt["stock_idx"]]["name"]
                print(f"  [{i+1}] {st_name} {opt['type']} | Strike: ${opt['strike']:.2f} | Expires in: {opt['days_left']} days")
        print("=" * 65)


if __name__ == "__main__":
    env = AmadeusTradingEnv()

    # Simulating a random agent taking random trades
    episodes = 500
    total_rewards = []

    for ep in tqdm(range(episodes), desc="Simulating Random Agent Actions"):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        total_rewards.append(ep_reward)

    print(f"Simulation completed. Average Net Worth Change: {np.mean(total_rewards):.2f}")
