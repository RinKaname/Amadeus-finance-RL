import numpy as np
from env import AmadeusTradingEnv

env = AmadeusTradingEnv()

# Let's see the EV of options for seed 48 when randomly trading
env.reset(seed=48)
done = False
max_nw = 0
while not done:
    action = 0
    obs, r, term, trunc, info = env.step(action)
    done = term or trunc

print("Final price for Seed 48 (Hold only):", env.stocks[0]['price'])

# What if economy goes into BOOM and we buy options?
env.reset(seed=48)
done = False
max_nw = 0
while not done:
    action = 0
    if env.economy == 1:
        action = 3 # Buy call
    elif env.economy == 2:
        action = 4 # Buy put
    obs, r, term, trunc, info = env.step(action)
    max_nw = max(max_nw, env.net_worth)
    done = term or trunc
print("Heuristic option buying NW:", env.net_worth)
