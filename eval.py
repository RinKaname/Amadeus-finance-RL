import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from model import WorldModel, ActorCritic, symlog
from env import AmadeusTradingEnv
import os
from collections import Counter
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model_path="trader_wm_final.safetensors", actor_path="trader_ac_final.safetensors", episodes=3):
    env = AmadeusTradingEnv()
    os.makedirs('./eval_recordings', exist_ok=True)
    
    action_dim = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    
    print(f"Loading World Model from {model_path}...")
    world_model = WorldModel(obs_dim=obs_dim, action_dim=action_dim).to(device)
    if os.path.exists(model_path):
        world_model.load_state_dict(load_file(model_path))
    else:
        print(f"Warning: {model_path} not found. Running with untrained model.")
    world_model.eval()
    
    print(f"Loading Actor-Critic from {actor_path}...")
    actor_critic = ActorCritic(feat_dim=world_model.feat_dim, action_dim=action_dim).to(device)
    if os.path.exists(actor_path):
        actor_critic.load_state_dict(load_file(actor_path))
    else:
        print(f"Warning: {actor_path} not found. Running with untrained model.")
    actor_critic.eval()
    
    action_names = {
        0: "Hold",
        1: "Buy 10 AMDS", 2: "Sell 10 AMDS", 3: "Buy AMDS Call", 4: "Buy AMDS Put",
        5: "Buy 10 SERN", 6: "Sell 10 SERN", 7: "Buy SERN Call", 8: "Buy SERN Put",
        9: "Buy 10 D-ML", 10: "Sell 10 D-ML", 11: "Buy D-ML Call", 12: "Buy D-ML Put"
    }
    
    print(f"\nStarting evaluation for {episodes} trading years...")
    for ep in range(episodes):
        obs, _ = env.reset()
        actions_list = [0]  # pad first step
        rewards_list = [0.0]
        dones_list = [False]
        obs_list = [obs]
        
        h, z = world_model.rssm.initial_state(batch_size=1, device=device)
        
        done = False
        total_reward = 0
        step = 0
        action_counts = Counter()
        final_net_worth = 0.0
        
        while not done:
            with torch.no_grad():
                # Process 1D observation, apply symlog, and reshape for the MLP
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                enc_out = world_model.encoder(symlog(obs_tensor))
                
                # Get posterior state
                z_post, _ = world_model.rssm.step_posterior(h, enc_out)
                feat = torch.cat([h, z_post], dim=-1)
                
                # Select action deterministically
                action = actor_critic.select_action(feat, explore=False).item()
                action_counts[action] += 1
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            final_net_worth = info.get("net_worth", 0.0)
            
            actions_list.append(action)
            rewards_list.append(reward)
            dones_list.append(done)
            obs_list.append(next_obs)
            total_reward += reward
            step += 1
            
            # Advance prior state based on the action taken
            with torch.no_grad():
                act_onehot = F.one_hot(torch.tensor([action], device=device), num_classes=action_dim).float()
                h, z, _ = world_model.rssm.step_prior(h, z_post, act_onehot)
                
            obs = next_obs
            
        print(f"\nEpisode {ep + 1} finished in {step} steps ({info.get('days', 0)} Days).")
        print(f"Starting Cash: $1000.00 | Final Net Worth: ${final_net_worth:.2f}")
        print(f"Total Reward (PnL): {total_reward:.2f}")
        
        # Print a breakdown of the actions it took
        print("Action Breakdown:")
        for act, count in action_counts.most_common(5):
            print(f"  {action_names[act]:>15}: {count} times")
            
        # Save trajectory as .npz file (no image key anymore, just vectors)
        npz_path = f"./eval_recordings/eval_ep{ep+1}.npz"
        np.savez_compressed(
            npz_path,
            obs=np.array(obs_list, dtype=np.float32),
            action=np.array(actions_list, dtype=np.int64),
            reward=np.array(rewards_list, dtype=np.float32),
            done=np.array(dones_list, dtype=bool)
        )
        
    print("\nEvaluation complete! Trajectories saved to ./eval_recordings/")

if __name__ == "__main__":
    evaluate("trader_wm_final.safetensors", "trader_ac_final.safetensors", episodes=100)