import numpy as np
from env import AmadeusTradingEnv

env = AmadeusTradingEnv()

# Let's inspect Seed 48 where net worth hits 34000
env.reset(seed=48)
done = False
while not done:
    action = 3 # Buy AMDS Call
    obs, r, term, trunc, info = env.step(action)
    done = term or trunc

    if env.net_worth > 20000:
        print(f"NW > 20000 at tick {env.ticks}")
        print(f"Economy: {env.economy}")
        print(f"Price: {env.stocks[0]['price']}")
        print(f"Options:")
        for opt in env.active_options:
            print(opt)
        break
