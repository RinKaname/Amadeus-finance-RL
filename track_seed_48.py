import numpy as np
from env import AmadeusTradingEnv

env = AmadeusTradingEnv()

# Let's inspect Seed 48
env.reset(seed=48)
done = False
while not done:
    action = 3 # Buy AMDS Call
    obs, r, term, trunc, info = env.step(action)
    done = term or trunc

    if env.ticks > 1500 and env.ticks < 1550:
        print(f"Tick: {env.ticks}, Price: {env.stocks[0]['price']}, Economy: {env.economy}")

print(f"Final NW: {env.net_worth}")
