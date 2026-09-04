import numpy as np
from env import AmadeusTradingEnv

env = AmadeusTradingEnv()

print("Testing random trading WITH options")
profits_opt = []

for ep in range(100):
    obs, info = env.reset(seed=42+ep)
    done = False

    while not done:
        action = env.np_random.integers(2, 5) # Buy calls/puts for AMDS
        obs, r, term, trunc, info = env.step(action)
        done = term or trunc

    profits_opt.append(env.net_worth - 1000)

print(f"Average profit from random (AMDS options only): {np.mean(profits_opt)}")
