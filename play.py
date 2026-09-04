import os
import sys
from env import AmadeusTradingEnv

ACTIONS_MAP = {
    0: "Hold / Wait",
    1: "Buy 10 AMDS",
    2: "Sell/Short 10 AMDS",
    3: "Buy AMDS Call",
    4: "Buy AMDS Put",
    5: "Buy 10 SERN",
    6: "Sell/Short 10 SERN",
    7: "Buy SERN Call",
    8: "Buy SERN Put",
    9: "Buy 10 D-ML",
    10: "Sell/Short 10 D-ML",
    11: "Buy D-ML Call",
    12: "Buy D-ML Put",
    13: "Close All Options"
}

def print_menu():
    print("""
 ACTIONS:
  [0] Hold (or press Enter)
  --- AMDS ($45) ---          --- SERN ($120) ---         --- D-ML ($15) ---
  [1] Buy 10 AMDS             [5] Buy 10 SERN             [9] Buy 10 D-ML
  [2] Sell/Short 10 AMDS      [6] Sell/Short 10 SERN      [10] Sell/Short 10 D-ML
  [3] Buy AMDS Call           [7] Buy SERN Call           [11] Buy D-ML Call
  [4] Buy AMDS Put            [8] Buy SERN Put            [12] Buy D-ML Put
 
  Options Management:
  [13] CLOSE ALL ACTIVE OPTIONS (Take Profit / Cash Out)
 
  Other: [skip N] Skip N ticks | [q] Quit
""")

def main():
    print("\n" + "=" * 65)
    print("       AMADEUS TRADING SIMULATOR (HUMAN PLAYABLE MODE)       ")
    print("=" * 65)
    
    env = AmadeusTradingEnv(render_mode="human")
    obs, info = env.reset()
    
    step_num = 0
    total_reward = 0.0
    
    while True:
        env.render()
        print_menu()
        
        user_input = input(">> Select Action [0-13]: ").strip().lower()
        
        if user_input in ["q", "quit", "exit"]:
            pnl = env.net_worth - 1000.00
            roi = (pnl / 1000.00) * 100.0
            avg_ret_day = pnl / max(1, env.days)
            avg_ret_tick = pnl / max(1, env.ticks)
            sign = "+" if pnl >= 0 else ""
            print("\n" + "=" * 65)
            print("                SESSION PERFORMANCE REPORT                       ")
            print("=" * 65)
            print(f" Days Traded    : {env.days} days ({env.ticks} ticks)")
            print(f" Final Net Worth: ${env.net_worth:.2f}")
            print(f" Total Return   : {sign}${pnl:.2f} ({sign}{roi:.2f}%)")
            print(f" Avg Return/Day : {sign}${avg_ret_day:.2f}")
            print(f" Avg Return/Tick: {sign}${avg_ret_tick:.4f}")
            print("=" * 65 + "\n")
            break
            
        skip_count = 1
        action = 0
        
        if user_input == "" or user_input == "0":
            action = 0
        elif user_input.startswith("skip"):
            parts = user_input.split()
            if len(parts) > 1 and parts[1].isdigit():
                skip_count = max(1, int(parts[1]))
            else:
                skip_count = 12 # Default skip 1 day
            action = 0
        elif user_input.isdigit() and int(user_input) in ACTIONS_MAP:
            action = int(user_input)
        else:
            print(f"Invalid input '{user_input}'. Please enter a number 0-12, or press Enter to Hold.")
            continue
            
        for s in range(skip_count):
            current_action = action if s == 0 else 0
            prev_nw = env.net_worth
            prev_cash = env.cash
            prev_stocks = {s["name"]: s["owned"] for s in env.stocks}
            prev_opts_count = len(env.active_options)
            
            obs, reward, terminated, truncated, info = env.step(current_action)
            total_reward += reward
            step_num += 1
            
            if s == 0 and current_action != 0:
                if current_action == 13:
                    realized = env.cash - prev_cash
                    print(f"\n>> [EXECUTED] Closed all {prev_opts_count} active options! Cash Realized: +${realized:.2f} | Tick P/L: ${reward:+.2f}")
                else:
                    stock_idx = (current_action - 1) // 4
                    action_type = (current_action - 1) % 4
                    st_name = env.stocks[stock_idx]["name"]
                    
                    if action_type in [0, 1]:
                        owned_diff = env.stocks[stock_idx]["owned"] - prev_stocks[st_name]
                        if owned_diff != 0:
                            print(f"\n>> [EXECUTED] {ACTIONS_MAP[current_action]} | New Owned: {env.stocks[stock_idx]['owned']} | Tick P/L: ${reward:+.2f}")
                        else:
                            needed = env.stocks[stock_idx]["price"] * 10
                            print(f"\n>> [REJECTED] {ACTIONS_MAP[current_action]} failed! Insufficient cash (Need ~${needed:.2f}, only have ${prev_cash:.2f})")
                    elif action_type in [2, 3]:
                        if len(env.active_options) > prev_opts_count:
                            print(f"\n>> [EXECUTED] {ACTIONS_MAP[current_action]} | Tick P/L: ${reward:+.2f}")
                        else:
                            print(f"\n>> [REJECTED] {ACTIONS_MAP[current_action]} failed! (Insufficient cash or max 4 option slots full)")
                
            if terminated:
                env.render()
                print("\n" + "!" * 65)
                print(f" GAME OVER: BANKRUPT! Final Net Worth: ${env.net_worth:.2f}")
                print("!" * 65)
                return
                
            if truncated:
                env.render()
                print("\n" + "*" * 65)
                print(f" TRADING YEAR COMPLETED! Final Net Worth: ${env.net_worth:.2f} (Total P/L: ${env.net_worth - 1000.0:+.2f})")
                print("*" * 65)
                return

if __name__ == "__main__":
    main()
