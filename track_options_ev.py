import numpy as np
from env import AmadeusTradingEnv

env = AmadeusTradingEnv()

# Let's see if options have positive EV
np.random.seed(42)
profits = []

for ep in range(100):
    obs, info = env.reset(seed=42+ep)
    done = False

    # Buy 4 AMDS calls on tick 0
    env.step(3)
    env.step(3)
    env.step(3)
    env.step(3)

    while not done:
        obs, r, term, trunc, info = env.step(0)
        done = term or trunc

    profits.append(env.net_worth - 1000)

print(f"Average profit from just buying 4 AMDS calls: {np.mean(profits)}")
