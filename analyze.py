import numpy as np
import glob
import os
from collections import Counter

def analyze_eval_recordings(folder_path="./eval_recordings"):
    npz_files = glob.glob(os.path.join(folder_path, "*.npz"))
    if not npz_files:
        print("No .npz files found in", folder_path)
        return

    total_episodes = len(npz_files)
    all_rewards = []
    all_lengths = []
    all_max_dd = []
    all_sharpe = []
    all_sortino = []
    global_action_counts = Counter()
    bankruptcies = 0
    
    action_names = {
        0: "Hold",
        1: "Buy 10 AMDS", 2: "Sell 10 AMDS", 3: "Buy AMDS Call", 4: "Buy AMDS Put",
        5: "Buy 10 SERN", 6: "Sell 10 SERN", 7: "Buy SERN Call", 8: "Buy SERN Put",
        9: "Buy 10 D-ML", 10: "Sell 10 D-ML", 11: "Buy D-ML Call", 12: "Buy D-ML Put"
    }

    for f in npz_files:
        data = np.load(f)
        actions = data['action']
        rewards = data['reward']
        
        # Calculate PnL (sum of rewards)
        ep_pnl = np.sum(rewards)
        all_rewards.append(ep_pnl)
        
        # Calculate episode length
        ep_len = len(actions) - 1 # first action is padded 0
        all_lengths.append(ep_len)
        
        # Check bankruptcy (if game ended before 3024 steps)
        # Note: 3024 is the max steps in a year
        if ep_len < 3024:
            bankruptcies += 1
            
        # Financial Metrics
        nw_history = 1000.0 + np.cumsum(rewards)
        nw_history = np.insert(nw_history, 0, 1000.0)
        
        running_max = np.maximum.accumulate(nw_history)
        drawdowns = (nw_history - running_max) / running_max
        all_max_dd.append(np.min(drawdowns))
        
        safe_nw = np.where(nw_history[:-1] <= 0, 1e-8, nw_history[:-1])
        step_returns = rewards / safe_nw
        
        mean_ret = np.mean(step_returns)
        std_ret = np.std(step_returns) + 1e-8
        sharpe = (mean_ret / std_ret) * np.sqrt(3024)
        all_sharpe.append(sharpe)
        
        negative_returns = step_returns[step_returns < 0]
        downside_std = np.std(negative_returns) + 1e-8 if len(negative_returns) > 0 else 1e-8
        sortino = (mean_ret / downside_std) * np.sqrt(3024)
        all_sortino.append(sortino)
            
        # Count actions
        for act in actions[1:]: # skip padded action
            global_action_counts[act] += 1

    avg_pnl = np.mean(all_rewards)
    median_pnl = np.median(all_rewards)
    max_pnl = np.max(all_rewards)
    min_pnl = np.min(all_rewards)
    win_rate = sum(1 for r in all_rewards if r > 0) / total_episodes * 100
    
    avg_max_dd = np.mean(all_max_dd) * 100
    avg_sharpe = np.mean(all_sharpe)
    avg_sortino = np.mean(all_sortino)
    
    print(f"=== Evaluation Analysis ({total_episodes} Episodes) ===")
    print(f"Average PnL: ${avg_pnl:,.2f}")
    print(f"Median PnL:  ${median_pnl:,.2f}")
    print(f"Max PnL:     ${max_pnl:,.2f}")
    print(f"Min PnL:     ${min_pnl:,.2f}")
    print(f"Profitable Episodes: {win_rate:.1f}%")
    print(f"Bankruptcies: {bankruptcies} ({(bankruptcies/total_episodes)*100:.1f}%)")
    print(f"Average Episode Length: {np.mean(all_lengths):.1f} steps")
    
    print("\n=== Advanced Financial Metrics ===")
    print(f"Average Max Drawdown: {avg_max_dd:.1f}%")
    print(f"Average Sharpe Ratio: {avg_sharpe:.2f}")
    print(f"Average Sortino Ratio: {avg_sortino:.2f}")
    
    print("\n=== Global Action Distribution ===")
    total_actions = sum(global_action_counts.values())
    for act, count in global_action_counts.most_common():
        pct = (count / total_actions) * 100
        print(f"{action_names.get(act, f'Action {act}'):>15}: {count:8,d} ({pct:.1f}%)")

if __name__ == "__main__":
    analyze_eval_recordings()
