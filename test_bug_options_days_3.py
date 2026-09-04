import numpy as np
from env import AmadeusTradingEnv

env = AmadeusTradingEnv()

print("Testing binomial tree bug")
# Check binomial tree again
# if T <= 0: return ...
# dt = T / N
# u = np.exp(sigma * np.sqrt(dt))
# p = (np.exp(r * dt) - d) / (u - d)

# Wait...
# time_to_expiry_years = 7.0 / 252.0 = 0.0277
# N = 10
# dt = 0.00277

# prices = S * (u ** np.arange(N, -1, -1)) * (d ** np.arange(0, N + 1, 1))

# What if N is too low?
# No, N=10 is just slightly inaccurate, but not 100,000% profit inaccurate.

# What about the option payoff?
# payoff = max(0.0, st["price"] - opt["strike"]) * 100 * opt["qty"]
# This is standard.

# What about option premium?
# premium = (price_per_share * 100 * qty) + self.tx_fee
# self.cash -= premium

# What if `qty` is wrong?
# opt["qty"] is 1.
# premium is `price_per_share * 100 + 0.5`.

# Where does the massive profit come from?
env.reset(seed=42)

for ep in range(1):
    obs, info = env.reset(seed=42+ep)
    done = False

    while not done:
        action = 3 # Buy AMDS Call
        obs, r, term, trunc, info = env.step(action)
        if env.net_worth > 5000:
            print("NW > 5000 at tick", env.ticks)
            print("Price:", env.stocks[0]["price"])
            print("Options:", env.active_options)
            break
        done = term or trunc
