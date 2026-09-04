import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

# --- Utilities ---
def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

def twohot(x, bins=255, min_val=-20.0, max_val=20.0):
    x = torch.clamp(x, min_val, max_val)
    bin_width = (max_val - min_val) / (bins - 1)
    bin_indices = (x - min_val) / bin_width
    lower = torch.floor(bin_indices).long()
    upper = torch.ceil(bin_indices).long()
    
    lower_weight = upper.float() - bin_indices
    upper_weight = bin_indices - lower.float()
    
    same = (lower == upper)
    lower_weight[same] = 1.0
    upper_weight[same] = 0.0
    
    batch_shape = x.shape
    two_hot = torch.zeros((*batch_shape, bins), device=x.device)
    two_hot.scatter_add_(-1, lower.unsqueeze(-1), lower_weight.unsqueeze(-1))
    two_hot.scatter_add_(-1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
    return two_hot

def twohot_loss(logits, targets, bins=255, min_val=-20.0, max_val=20.0):
    targets_twohot = twohot(targets, bins, min_val, max_val).detach()
    loss = -torch.sum(targets_twohot * F.log_softmax(logits, dim=-1), dim=-1)
    return loss.mean()

def twohot_decode(logits, bins=255, min_val=-20.0, max_val=20.0):
    probs = F.softmax(logits, dim=-1)
    bin_width = (max_val - min_val) / (bins - 1)
    supports = torch.linspace(min_val, max_val, bins, device=logits.device)
    expected_value = torch.sum(probs * supports, dim=-1, keepdim=True)
    return expected_value

class CategoricalLatent(nn.Module):
    def __init__(self, num_categoricals=16, num_classes=16, unimix_ratio=0.01):
        super().__init__()
        self.num_categoricals = num_categoricals
        self.num_classes = num_classes
        self.unimix_ratio = unimix_ratio

    def forward(self, logits):
        logits = logits.view(-1, self.num_categoricals, self.num_classes)
        probs = F.softmax(logits, dim=-1)
        
        if self.unimix_ratio > 0.0:
            probs = (1.0 - self.unimix_ratio) * probs + self.unimix_ratio / self.num_classes
        
        dist = D.OneHotCategorical(probs=probs)
        sample = dist.sample()
        
        st_sample = sample + (probs - probs.detach())
        flat_latent = st_sample.view(-1, self.num_categoricals * self.num_classes)
        
        return flat_latent, logits

# --- Networks ---
class MLPEncoder(nn.Module):
    def __init__(self, obs_dim, hidden_dim=256, out_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim * 2, out_dim)
        )
        self.out_dim = out_dim
        
    def forward(self, x):
        # x: [B, obs_dim]
        return self.net(x)

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, obs_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, obs_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class RSSM(nn.Module):
    def __init__(self, action_dim=14, hidden_dim=256, enc_dim=1024, num_categoricals=16, num_classes=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_categoricals = num_categoricals
        self.num_classes = num_classes
        self.latent_dim = num_categoricals * num_classes
        
        self.rnn = nn.GRUCell(self.latent_dim + action_dim, hidden_dim)
        
        self.prior_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, self.latent_dim)
        )
        
        self.post_mlp = nn.Sequential(
            nn.Linear(hidden_dim + enc_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, self.latent_dim)
        )
        self.latent_sampler = CategoricalLatent(num_categoricals, num_classes)
        
    def initial_state(self, batch_size, device):
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        z = torch.zeros(batch_size, self.latent_dim, device=device)
        return h, z

    def step_prior(self, h_prev, z_prev, action):
        x = torch.cat([z_prev, action], dim=-1)
        h = self.rnn(x, h_prev)
        prior_logits = self.prior_mlp(h)
        z_prior, prior_d_logits = self.latent_sampler(prior_logits)
        return h, z_prior, prior_d_logits

    def step_posterior(self, h, enc_out):
        post_logits = self.post_mlp(torch.cat([h, enc_out], dim=-1))
        z_post, post_d_logits = self.latent_sampler(post_logits)
        return z_post, post_d_logits

class DensePredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, out_dim=255, layers=2):
        super().__init__()
        net = []
        for i in range(layers):
            d_in = in_dim if i == 0 else hidden_dim
            net.extend([nn.Linear(d_in, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(inplace=True)])
        net.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*net)
        
    def forward(self, x):
        return self.net(x)

class ActorCritic(nn.Module):
    def __init__(self, feat_dim, action_dim=14, hidden_dim=256, layers=1):
        super().__init__()
        actor_net = []
        for i in range(layers):
            d_in = feat_dim if i == 0 else hidden_dim
            actor_net.extend([nn.Linear(d_in, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(inplace=True)])
        actor_net.append(nn.Linear(hidden_dim, action_dim))
        self.actor = nn.Sequential(*actor_net)
        
        self.critic = DensePredictor(feat_dim, hidden_dim, out_dim=255, layers=layers)
        
        self.ema_critic = DensePredictor(feat_dim, hidden_dim, out_dim=255, layers=layers)
        self.ema_critic.load_state_dict(self.critic.state_dict())
        for param in self.ema_critic.parameters():
            param.requires_grad = False

    def update_ema(self, decay=0.98):
        with torch.no_grad():
            for param, ema_param in zip(self.critic.parameters(), self.ema_critic.parameters()):
                ema_param.data.copy_(decay * ema_param.data + (1 - decay) * param.data)

    def select_action(self, feat, explore=True):
        logits = self.actor(feat)
        probs = F.softmax(logits, dim=-1)
        if explore:
            probs = 0.99 * probs + 0.01 / logits.shape[-1]
            dist = D.Categorical(probs=probs)
            action = dist.sample()
        else:
            action = torch.argmax(probs, dim=-1)
        return action

class WorldModel(nn.Module):
    def __init__(self, obs_dim=64, action_dim=14):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = 256
        
        self.encoder = MLPEncoder(obs_dim=self.obs_dim, hidden_dim=self.hidden_dim, out_dim=1024)
        self.rssm = RSSM(action_dim=action_dim, hidden_dim=self.hidden_dim, enc_dim=self.encoder.out_dim)
        self.feat_dim = self.hidden_dim + self.rssm.latent_dim
        
        self.decoder = MLPDecoder(in_dim=self.feat_dim, obs_dim=self.obs_dim, hidden_dim=self.hidden_dim)
        self.reward_predictor = DensePredictor(self.feat_dim, self.hidden_dim, out_dim=255, layers=2)
        self.continue_predictor = DensePredictor(self.feat_dim, self.hidden_dim, out_dim=1, layers=2)
        
    def unroll(self, obs, act, cont):
        B, T = act.shape
        
        # Flatten temporal observation vector properly
        obs_flat = obs.view(B * T, -1)
        enc_out = self.encoder(obs_flat).view(B, T, -1)
        
        act_onehot = F.one_hot(act, num_classes=self.action_dim).float()
        
        h, z = self.rssm.initial_state(B, obs.device)
        
        post_states = []
        prior_logits_list = []
        post_logits_list = []
        
        for t in range(T):
            mask = cont[:, t].unsqueeze(-1)
            h = h * mask
            z = z * mask
            
            prior_logits = self.rssm.prior_mlp(h)
            z_prior, prior_d_logits = self.rssm.latent_sampler(prior_logits)
            
            z_post, post_logits = self.rssm.step_posterior(h, enc_out[:, t])
            
            post_states.append(torch.cat([h, z_post], dim=-1))
            prior_logits_list.append(prior_logits)
            post_logits_list.append(post_logits)
            
            x = torch.cat([z_post, act_onehot[:, t]], dim=-1)
            h = self.rssm.rnn(x, h)
            z = z_post
            
        post_states = torch.stack(post_states, dim=1)
        prior_logits = torch.stack(prior_logits_list, dim=1)
        post_logits = torch.stack(post_logits_list, dim=1)
        
        return post_states, prior_logits, post_logits

    def kl_loss(self, post_logits, prior_logits, free_bits=1.0, kl_balance=0.8):
        B, T = post_logits.shape[:2]
        post_logits = post_logits.view(B*T, self.rssm.num_categoricals, self.rssm.num_classes)
        prior_logits = prior_logits.view(B*T, self.rssm.num_categoricals, self.rssm.num_classes)
        
        post_dist = D.Independent(D.OneHotCategorical(logits=post_logits), 1)
        prior_dist = D.Independent(D.OneHotCategorical(logits=prior_logits), 1)
        
        kl_prior = D.kl.kl_divergence(
            D.Independent(D.OneHotCategorical(logits=post_logits.detach()), 1), 
            prior_dist
        )
        kl_post = D.kl.kl_divergence(
            post_dist, 
            D.Independent(D.OneHotCategorical(logits=prior_logits.detach()), 1)
        )
        
        kl_prior = torch.clamp(kl_prior, min=free_bits)
        kl_post = torch.clamp(kl_post, min=free_bits)
        
        loss = (kl_balance * kl_prior + (1.0 - kl_balance) * kl_post).mean()
        return loss

    def imagine(self, start_feat, actor, horizon=15):
        h = start_feat[:, :self.hidden_dim]
        z = start_feat[:, self.hidden_dim:]
        
        imagined_feats = []
        imagined_actions = []
        next_feats = []
        
        for _ in range(horizon):
            feat = torch.cat([h, z], dim=-1)
            imagined_feats.append(feat)
            
            action = actor.select_action(feat.detach(), explore=True)
            imagined_actions.append(action)
            act_onehot = F.one_hot(action, num_classes=self.action_dim).float()
            
            h, z, _ = self.rssm.step_prior(h, z, act_onehot)
            next_feats.append(torch.cat([h, z], dim=-1))
            
        return torch.stack(imagined_feats, dim=0), torch.stack(imagined_actions, dim=0), torch.stack(next_feats, dim=0)