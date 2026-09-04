import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from safetensors.torch import save_file, load_file
import os
import torch.nn.functional as F
import torch.distributions as D
from collections import deque

from model import WorldModel, ActorCritic, symlog, symexp, twohot_loss, twohot_decode
from env import AmadeusTradingEnv

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOTAL_STEPS = 50_000
TRAIN_EVERY = 5          
TRAIN_STEPS = 1          
PREFILL_STEPS = 500     
BATCH_SIZE = 32
SEQ_LEN = 31
IMAGINE_HORIZON = 85
torch.distributions.Distribution.set_default_validate_args(False)

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity=100_000, obs_dim=64):
        self.capacity = capacity
        # CHANGED: from 64x64x3 np.uint8 to 1D vector np.float32
        self.obs = np.empty((capacity, obs_dim), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.bool_)
        self.idx = 0
        self.size = 0

    def add(self, obs, action, reward, done):
        self.obs[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_sequence(self, batch_size=BATCH_SIZE, seq_len=SEQ_LEN):
        required_len = seq_len + 1
        
        if self.size < self.capacity:
            valid_max = max(1, self.size - required_len)
            start_indices = np.random.randint(0, valid_max, size=batch_size)
        else:
            start_indices = []
            while len(start_indices) < batch_size:
                i = np.random.randint(0, self.capacity - required_len)
                if not (i <= self.idx < i + required_len):
                    start_indices.append(i)
            start_indices = np.array(start_indices)
            
        obs_seq = np.stack([self.obs[i : i + required_len] for i in start_indices])
        act_seq = np.stack([self.actions[i : i + required_len] for i in start_indices])
        rew_seq = np.stack([self.rewards[i : i + required_len] for i in start_indices])
        don_seq = np.stack([self.dones[i : i + required_len] for i in start_indices])
        
        # CHANGED: Removed .permute(0, 1, 4, 2, 3) as there are no image channels
        obs_tensor = torch.tensor(obs_seq, dtype=torch.float32, device=device)
        act_tensor = torch.tensor(act_seq, dtype=torch.int64, device=device)
        rew_tensor = torch.tensor(rew_seq, dtype=torch.float32, device=device).unsqueeze(-1)
        cont_tensor = 1.0 - torch.tensor(don_seq, dtype=torch.float32, device=device)
        
        return obs_tensor, act_tensor, rew_tensor, cont_tensor

def compute_lambda_returns(rewards, continues, next_values, lambda_=0.95, gamma=0.99):
    H, B = rewards.shape[:2]
    returns = torch.zeros_like(next_values)
    last_val = next_values[-1]
    
    for t in reversed(range(H)):
        returns[t] = rewards[t] + continues[t] * gamma * ((1 - lambda_) * next_values[t] + lambda_ * last_val)
        last_val = returns[t]
    return returns

def train():
    env = AmadeusTradingEnv()
    action_dim = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    
    world_model = WorldModel(obs_dim=obs_dim, action_dim=action_dim).to(device)
    actor_critic = ActorCritic(feat_dim=world_model.feat_dim, action_dim=action_dim).to(device)
    
    wm_path = "trader_wm_final.safetensors"
    ac_path = "trader_ac_final.safetensors"
    
    if os.path.exists(wm_path):
        print(f"Resuming World Model from {wm_path}...")
        world_model.load_state_dict(load_file(wm_path))
    if os.path.exists(ac_path):
        print(f"Resuming Actor-Critic from {ac_path}...")
        actor_critic.load_state_dict(load_file(ac_path))
    
    wm_opt = torch.optim.Adam(world_model.parameters(), lr=1e-4, eps=1e-8)
    ac_opt = torch.optim.Adam(actor_critic.parameters(), lr=3e-5, eps=1e-5)
    
    buffer = ReplayBuffer(capacity=100_000, obs_dim=obs_dim)
    
    obs, _ = env.reset()
    h, z = world_model.rssm.initial_state(batch_size=1, device=device)
        
    if buffer.size < PREFILL_STEPS:
        print("Prefilling buffer with random trading steps...")
        for _ in tqdm(range(PREFILL_STEPS - buffer.size), desc="Prefill"):
            act = env.action_space.sample()
            next_obs, rew, done, truncated, _ = env.step(act)
            buffer.add(obs, act, rew, done)
            if done or truncated:
                obs, _ = env.reset()
            else:
                obs = next_obs

    ep_reward = 0.0
    recent_returns = deque(maxlen=50)
    max_net_worth = 0.0 # CHANGED: from max_score to max_net_worth
    
    print("Starting Training...")
    pbar = tqdm(range(TOTAL_STEPS), desc="Training Steps")
    
    for step in pbar:
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            # --- FIX 1: Apply symlog to the observation before encoding ---
            enc_out = world_model.encoder(symlog(obs_tensor))
            z_post, _ = world_model.rssm.step_posterior(h, enc_out)
            feat = torch.cat([h, z_post], dim=-1)
            action = actor_critic.select_action(feat, explore=True).item()

        next_obs, rew, done, truncated, info = env.step(action)
        ep_reward += rew
        
        # --- FIX 2: Track max net worth continuously, not just on death ---
        if info["net_worth"] > max_net_worth:
            max_net_worth = info["net_worth"]
            
        buffer.add(obs, action, rew, done)
        
        if done or truncated:
            recent_returns.append(ep_reward)
            ep_reward = 0.0
            obs, _ = env.reset()
            h, z = world_model.rssm.initial_state(batch_size=1, device=device)
        else:
            obs = next_obs
            act_onehot = F.one_hot(torch.tensor([action], device=device), num_classes=action_dim).float()
            h, z, _ = world_model.rssm.step_prior(h, z_post, act_onehot)

        if step % TRAIN_EVERY == 0 and buffer.size >= PREFILL_STEPS:
            for _ in range(TRAIN_STEPS):
                b_obs, b_act, b_rew, b_cont = buffer.sample_sequence()
                
                # --- FIX 3: Apply symlog to the batch target observations ---
                b_obs_sym = symlog(b_obs)
                
                # --- Train World Model ---
                wm_opt.zero_grad()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    mask_cont = torch.cat([torch.ones_like(b_cont[:, :1]), b_cont[:, :-1]], dim=1)
                    
                    # Pass the symlog observations into unroll
                    post_states, prior_logits, post_logits = world_model.unroll(b_obs_sym, b_act, mask_cont)
                    
                    rec_obs = world_model.decoder(post_states)
                    
                    # Target is now safely compressed by symlog
                    rec_loss = F.mse_loss(rec_obs, b_obs_sym) 
                    
                    rew_preds = world_model.reward_predictor(post_states)
                    rew_loss = twohot_loss(rew_preds[:, 1:], symlog(b_rew[:, :-1].squeeze(-1)))
                    
                    cont_preds = world_model.continue_predictor(post_states)
                    cont_loss = F.binary_cross_entropy_with_logits(cont_preds[:, 1:].squeeze(-1), b_cont[:, :-1])
                    
                    kl_loss = world_model.kl_loss(post_logits, prior_logits)
                    wm_total_loss = rec_loss + rew_loss + cont_loss + 0.5 * kl_loss

                wm_total_loss.backward()
                nn.utils.clip_grad_norm_(world_model.parameters(), 1000.0)
                wm_opt.step()
                
                # --- Train Actor-Critic ---
                ac_opt.zero_grad()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    initial_states = post_states.detach().view(-1, world_model.feat_dim)
                    imag_feats, imag_actions, next_feats = world_model.imagine(initial_states, actor_critic, horizon=IMAGINE_HORIZON)
                    
                    imag_rew_logits = world_model.reward_predictor(next_feats)
                    imag_rewards = symexp(twohot_decode(imag_rew_logits)).squeeze(-1)
                    
                    imag_cont_logits = world_model.continue_predictor(next_feats)
                    imag_cont = torch.sigmoid(imag_cont_logits).squeeze(-1)
                    
                    val_logits = actor_critic.ema_critic(next_feats)
                    next_values = symexp(twohot_decode(val_logits)).squeeze(-1)
                    
                    returns = compute_lambda_returns(imag_rewards, imag_cont, next_values)
                    
                    policy_logits = actor_critic.actor(imag_feats.detach())
                    policy_dist = D.Categorical(logits=policy_logits)
                    log_probs = policy_dist.log_prob(imag_actions)
                    
                    baseline_logits = actor_critic.critic(imag_feats.detach())
                    baseline = symexp(twohot_decode(baseline_logits)).squeeze(-1)
                    
                    ret_percentiles = torch.quantile(returns.detach(), torch.tensor([0.05, 0.95], device=device))
                    ret_scale = torch.clamp(ret_percentiles[1] - ret_percentiles[0], min=1.0)
                    
                    advantage = returns.detach() - baseline.detach()
                    actor_loss = -torch.mean(log_probs * (advantage / ret_scale))
                    
                    entropy_loss = -1e-4 * policy_dist.entropy().mean()
                    critic_loss = twohot_loss(baseline_logits, symlog(returns.detach()))
                    
                    ac_total_loss = actor_loss + entropy_loss + critic_loss
                    
                ac_total_loss.backward()
                nn.utils.clip_grad_norm_(actor_critic.parameters(), 100.0)
                ac_opt.step()
                actor_critic.update_ema()
                torch.cuda.empty_cache()

            if step % 500 == 0:
                avg_ret = np.mean(recent_returns) if recent_returns else 0.0
                pbar.set_postfix({
                    "AvgRet": f"{avg_ret:.1f}",
                    "MaxNW": f"{max_net_worth:.1f}",
                    "WM_Loss": f"{wm_total_loss.item():.2f}",
                    "AC_Loss": f"{ac_total_loss.item():.2f}"
                })
                
            if step > 0 and step % 100000 == 0:
                save_file(world_model.state_dict(), f"trader_wm_{step}.safetensors")
                save_file(actor_critic.state_dict(), f"trader_ac_{step}.safetensors")

    save_file(world_model.state_dict(), "trader_wm_final.safetensors")
    save_file(actor_critic.state_dict(), "trader_ac_final.safetensors")
    print("Training complete! Models saved.")

if __name__ == "__main__":
    train()