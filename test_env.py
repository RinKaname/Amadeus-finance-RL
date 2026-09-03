import numpy as np
from env import AmadeusTradingEnv

def run_heuristic_bot():
    """
    Heuristic Bot for AmadeusTradingEnv.
    This bot exploits a vulnerability in the environment's RRR bracket logic.

    The Bug (Deduction):
    When a short position hits its Stop Loss or Take Profit (RRR Bracket Orders),
    the environment automatically covers the short by doing `self.cash -= cost`.
    However, it DOES NOT check if `self.cash` is sufficient to cover this cost.
    This forces `self.cash` to go negative. Because the termination condition
    only checks if `self.net_worth <= 750.0`, the agent can maintain a massive
    negative cash balance indefinitely, effectively getting an infinite, interest-free
    margin loan that prevents bankruptcy as long as overall net worth is > 750.
    """
    env = AmadeusTradingEnv()

    # Try different seeds to make sure we hit the bug
    for seed in range(50):
        obs, info = env.reset(seed=seed)
        done = False

        while not done:
            # We want to force a situation where we have no cash, but we have a short position
            # that hits stop loss/take profit.
            # So we short a stock, get cash, use that cash to buy ANOTHER stock/option,
            # and then wait for the short to hit its RRR bracket.

            # 1. Short AMDS (action 2)
            env.step(2)

            # 2. Spend all cash on SERN (action 5) and D-ML (action 9)
            env.step(5)
            env.step(9)

            # 3. Buy options to drain remaining cash
            env.step(3)
            env.step(4)

            # Step forward to trigger market updates
            obs, reward, terminated, truncated, info = env.step(0)
            done = terminated or truncated

            # Check if we successfully exploited the bug
            if env.cash < 0:
                print("\n--- BUG DETECTED ---")
                print("The agent's cash balance has gone NEGATIVE, but the episode did not terminate!")
                print(f"Current Cash: {env.cash:.2f}")
                print(f"Current Net Worth: {env.net_worth:.2f}")
                print("Reason: RRR bracket order (stop loss/take profit) closed a short position and deducted the cover cost without checking available cash.")
                print("--------------------\n")
                return

    print("Could not trigger bug.")

if __name__ == "__main__":
    run_heuristic_bot()
