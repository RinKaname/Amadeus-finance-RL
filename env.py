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

    def __init__(self, render_mode=None):
        super(AmadeusTradingEnv, self).__init__()
        self.render_mode = render_mode
        
        # 13 Discrete Actions:
        # 0: Hold
        # 1: Buy 10 AMDS, 2: Sell 10 AMDS, 3: Buy AMDS Call, 4: Buy AMDS Put
        # 5: Buy 10 SERN, 6: Sell 10 SERN, 7: Buy SERN Call, 8: Buy SERN Put
        # 9: Buy 10 D-ML, 10: Sell 10 D-ML, 11: Buy D-ML Call, 12: Buy D-ML Put
        self.action_space = spaces.Discrete(13)
        
        # Observation Space: 64 flattened continuous/discrete features
        # [0] Cash
        # [1] Economy (0=STABLE, 1=BOOM, 2=BUST)
        # [2-7] Stocks: (Price, Owned) for AMDS, SERN, D-ML
        # [8-63] Options: 4 active slots * 5 features per slot
        #        Slot features: (is_active, type_call_0_put_1, stock_idx, strike, days_left)
        low = np.array([-np.inf] * 64, dtype=np.float32)
        high = np.array([np.inf] * 64, dtype=np.float32)
        
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.tx_fee = 0.50
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.ticks = 0
        self.days = 1
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
        
        # 4. End condition: Bankruptcy or 25% Drawdown (Strict Risk Management)
        terminated = bool(self.net_worth <= 750.0)
        
        # 5. Truncation: 1 year of trading (252 days * 12 market updates a day)
        truncated = bool(self.ticks >= 3024)
        
        info = {
            "net_worth": self.net_worth,
            "days": self.days
        }
        
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        obs = np.zeros(64, dtype=np.float32)
        obs[0] = self.cash
        obs[1] = float(self.economy)
        
        idx = 2
        for s in self.stocks:
            obs[idx] = s["price"]
            obs[idx+1] = float(s["owned"])
            idx += 2
            
        opt_idx = 8
        for i in range(4):
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
            
        stock_idx = (action - 1) // 4
        action_type = (action - 1) % 4
        
        st = self.stocks[stock_idx]
        share_multiplier = 10 
        
        if action_type == 0: # Buy Shares
            cost = (st["price"] * share_multiplier) + self.tx_fee
            if self.cash >= cost:
                self.cash -= cost
                
                # Update Cost Basis
                if st["owned"] >= 0:
                    total_spent = (st["owned"] * st["cost_basis"]) + (share_multiplier * st["price"])
                    st["owned"] += share_multiplier
                    st["cost_basis"] = total_spent / st["owned"]
                else: # Covering short, might flip to long
                    st["owned"] += share_multiplier
                    if st["owned"] > 0:
                        st["cost_basis"] = st["price"]
                    elif st["owned"] == 0:
                        st["cost_basis"] = 0.0
                
        elif action_type == 1: # Sell (or Short) Shares
            # Case A: Selling shares we already own
            if st["owned"] >= share_multiplier:
                revenue = (st["price"] * share_multiplier) - self.tx_fee
                self.cash += revenue
                st["owned"] -= share_multiplier
                if st["owned"] == 0:
                    st["cost_basis"] = 0.0
            
            # Case B: Short selling (Requires margin compliance)
            else:
                # Calculate current total short liability across all assets
                total_short_liability = sum([abs(s["owned"]) * s["price"] for s in self.stocks if s["owned"] < 0])
                new_short_value = st["price"] * share_multiplier
                
                # Strict 150% Initial Margin Requirement
                if self.net_worth > 1.5 * (total_short_liability + new_short_value):
                    revenue = new_short_value - self.tx_fee
                    self.cash += revenue
                    
                    # Update cost basis for short
                    if st["owned"] <= 0:
                        total_shorted = (abs(st["owned"]) * st["cost_basis"]) + (share_multiplier * st["price"])
                        st["owned"] -= share_multiplier
                        st["cost_basis"] = total_shorted / abs(st["owned"])
                    else: # Flipping from long to short
                        st["owned"] -= share_multiplier
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
        prices = S * (u ** np.arange(N, -1, -1)) * (d ** np.arange(0, N + 1, 1))
        
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
        if len(self.active_options) >= 4:
            return 
            
        st = self.stocks[stock_idx]
        qty = 1 
        
        # Options are assumed to be generated with 7 days to expiry (7 / 252 years)
        time_to_expiry_years = 7.0 / 252.0
        
        # Derive annualized volatility based on environment's random normal step generation
        # (Std Dev of random walk * sqrt(total ticks in a year)) -> approx vol * 0.448
        annualized_vol = st["vol"] * 0.448
        risk_free_rate = 0.05
        
        price_per_share = self._binomial_option_pricing(
            S=st["price"], 
            K=st["price"], 
            T=time_to_expiry_years, 
            r=risk_free_rate, 
            sigma=annualized_vol, 
            opt_type=opt_type
        )
        
        premium = (price_per_share * 100 * qty) + self.tx_fee
        
        if self.cash >= premium:
            self.cash -= premium
            self.active_options.append({
                "type": opt_type,
                "stock_idx": stock_idx,
                "strike": st["price"],
                "qty": qty,
                "days_left": 7
            })

    def _update_market(self):
        self.ticks += 1
        
        # Economy shift probability matching Lua frequency
        if self.np_random.random() < 0.035:
            self._change_economy()
            
        # Daily processing (1 day = 12 market updates)
        if self.ticks % 12 == 0:
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
            time_to_expiry_years = max(0.0, opt["days_left"] / 252.0)
            annualized_vol = st["vol"] * 0.448
            risk_free_rate = 0.05
            
            price_per_share = self._binomial_option_pricing(
                S=st["price"], 
                K=opt["strike"], 
                T=time_to_expiry_years, 
                r=risk_free_rate, 
                sigma=annualized_vol, 
                opt_type=opt["type"]
            )
            
            val = price_per_share * 100 * opt["qty"]
            self.net_worth += val
            
        # RRR Bracket Orders (10% Stop Loss, 20% Take Profit)
        sl_pct = 0.10
        tp_pct = 0.20
        
        for st in self.stocks:
            if st["owned"] > 0: # Long Position
                if st["price"] <= st["cost_basis"] * (1.0 - sl_pct) or st["price"] >= st["cost_basis"] * (1.0 + tp_pct):
                    revenue = (st["price"] * st["owned"]) - self.tx_fee
                    self.cash += revenue
                    st["owned"] = 0
                    st["cost_basis"] = 0.0
                    
            elif st["owned"] < 0: # Short Position
                if st["price"] >= st["cost_basis"] * (1.0 + sl_pct) or st["price"] <= st["cost_basis"] * (1.0 - tp_pct):
                    cost = (st["price"] * abs(st["owned"])) + self.tx_fee
                    self.cash -= cost
                    st["owned"] = 0
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
            
        return price * (normal + trend) * (volatility) * 0.01

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
            yield_val = 0.02 + (self.np_random.random() * 0.04)
            payout = st["owned"] * (st["price"] * yield_val)
            self.cash += payout


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