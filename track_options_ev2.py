import numpy as np
from env import AmadeusTradingEnv

# Let's inspect the binomial model again
# price_per_share = self._binomial_option_pricing(
#     S=st["price"], K=opt["strike"], T=time_to_expiry_years, r=risk_free_rate, sigma=annualized_vol, opt_type=opt["type"]
# )

# Wait, `sigma = annualized_vol`.
# `annualized_vol = st["vol"] * 0.448`
# Earlier I calculated:
# stddev(normal) = sqrt(2/3) ≈ 0.816
# Ticks in a year = 3024
# sqrt(3024) ≈ 54.99
# So annualized_vol = 0.816 * vol * 0.01 * 54.99 = vol * 0.448.
# This assumes the volatility is correct.
# What if the price change is NOT normal?
# normal = (r1 + r2 - 1.0) * 2.0
# trend = 0.5 (BOOM), -0.5 (BUST), or (r1 - 0.5) * 0.2 (STABLE)
# Since the economy can be BOOM or BUST, the trend adds MASSIVE drift!
# BOOM adds +0.5 to the daily change factor.
# A +0.5 drift per tick, over 3024 ticks!
# If the options are priced using ONLY volatility, and ignoring the MASSIVE drift from the economy...
# Then Call options during a BOOM economy are massively underpriced!
# Put options during a BUST economy are massively underpriced!
# Is this the bug?
# "Average profit from random: 114074.6"
# Random actions make 114,000 profit on average from a 1,000 starting balance?
# This is absurdly high for random trading. It means the environment has a massive positive expected value exploit.
# If random trading makes 100,000% return, there is a fundamental pricing bug.

print("Testing random trading without options")
profits_no_opt = []
env = AmadeusTradingEnv()

for ep in range(100):
    obs, info = env.reset(seed=42+ep)
    done = False

    while not done:
        action = env.np_random.integers(0, 3) # only hold, buy 10 AMDS, sell 10 AMDS
        obs, r, term, trunc, info = env.step(action)
        done = term or trunc

    profits_no_opt.append(env.net_worth - 1000)

print(f"Average profit from random (no options): {np.mean(profits_no_opt)}")
