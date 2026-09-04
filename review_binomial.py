from env import AmadeusTradingEnv
env = AmadeusTradingEnv()
# It seems the options are genuinely just mispriced relative to the *economic trend*.
# In the Black-Scholes/Binomial model, the assumption is that the asset follows a random walk (Geometric Brownian Motion) with drift exactly equal to the risk-free rate under the risk-neutral measure.
# However, this environment has distinct "BOOM" and "BUST" phases which introduce a massive directional drift (+0.5 or -0.5 * volatility factor).
# Because the binomial model used to price the options DOES NOT account for this predictable drift (it just uses `annualized_vol`), the options are systematically underpriced when a strong trend (BOOM/BUST) is active.
# An agent can just look at `obs[1]` (the economy state), and buy Calls in a BOOM, and Puts in a BUST, and print infinite money.
# I already documented the negative cash / infinite margin bug which is arguably a clearer structural flaw in the engine (since it allows a broken state).
# Is the mispriced options due to predictable drift a "bug"? Yes, it's an arbitrage opportunity.
