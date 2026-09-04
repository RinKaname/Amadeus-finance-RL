import numpy as np
from env import AmadeusTradingEnv

env = AmadeusTradingEnv()

# Let's find exactly why random option buying is so profitable
profits_opt = []
for ep in range(10):
    obs, info = env.reset(seed=42+ep)
    done = False
    max_nw = 0
    while not done:
        action = 3 # Buy AMDS Call
        obs, r, term, trunc, info = env.step(action)
        max_nw = max(max_nw, env.net_worth)
        done = term or trunc
    print(f"Seed {42+ep}, Final NW: {env.net_worth}, Max NW: {max_nw}, Final Price: {env.stocks[0]['price']}")
