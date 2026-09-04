import numpy as np
from env import AmadeusTradingEnv

env = AmadeusTradingEnv()

env.reset(seed=42)

# wait what if I just randomly trade?
profits = []

for ep in range(100):
    obs, info = env.reset(seed=42+ep)
    done = False

    while not done:
        action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)
        done = term or trunc

    profits.append(env.net_worth - 1000)

print(f"Average profit from random: {np.mean(profits)}")
