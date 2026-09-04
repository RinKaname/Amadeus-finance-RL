import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.modules.models.model_based_v3 import RSSMPriorV3, RSSMPosteriorV3
from torchrl.objectives.dreamer_v3 import DreamerV3ModelLoss, DreamerV3ActorLoss, DreamerV3ValueLoss

# ==============================================================================
# 1. DreamerV3 MLP Building Block (RMS-Normalized SiLU MLP with Outscale)
# ==============================================================================
class DreamerV3MLP(nn.Module):
    """
    DreamerV3 RMS-normalized MLP building block as specified in:
    https://docs.pytorch.org/rl/main/reference/dreamer_v3.html
    and Hafner et al., 2023.
    """
    DEFAULT_DEPTH = 2
    DEFAULT_OUTSCALE = 1.0
    DEFAULT_NORM_EPS = 1e-4

    def __init__(
        self,
        in_features: int,
        out_features: int | None = None,
        depth: int = DEFAULT_DEPTH,
        num_cells: int = 512,
        outscale: float = DEFAULT_OUTSCALE,
        norm_eps: float = DEFAULT_NORM_EPS,
        device=None,
    ):
        super().__init__()
        out_features = out_features if out_features is not None else num_cells
        layers = []
        curr_in = in_features

        for _ in range(depth):
            layers.append(nn.Linear(curr_in, num_cells, device=device))
            layers.append(nn.RMSNorm(num_cells, eps=norm_eps, device=device))
            layers.append(nn.SiLU())
            curr_in = num_cells

        # Final projection layer scaled by outscale
        out_layer = nn.Linear(curr_in, out_features, device=device)
        bound = outscale / (curr_in ** 0.5)
        nn.init.uniform_(out_layer.weight, -bound, bound)
        nn.init.zeros_(out_layer.bias)
        layers.append(out_layer)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ==============================================================================
# 2. Complete DreamerV3 World Model
# ==============================================================================
class DreamerV3WorldModel(nn.Module):
    """
    Full World Model combining:
    - Observation Encoder
    - RSSM Dynamics (Prior & Posterior)
    - Observation Reconstruction Decoder
    - Reward Predictor (Two-Hot Categorical)
    - Continuation Predictor
    """
    MLP_DEPTH = 2
    ENCODER_OUTSCALE = 1.0
    DECODER_OUTSCALE = 1.0
    REWARD_OUTSCALE = 0.0
    CONTINUE_OUTSCALE = 1.0

    def __init__(
        self,
        obs_dim: int = 64,
        action_dim: int = 14,
        embed_dim: int = 512,
        hidden_dim: int = 512,
        num_categoricals: int = 16,
        num_classes: int = 16,
        num_reward_bins: int = 255,
        device=None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_categoricals = num_categoricals
        self.num_classes = num_classes
        self.state_dim = num_categoricals * num_classes

        # 1. Observation Encoder
        self.encoder = DreamerV3MLP(
            in_features=obs_dim,
            out_features=embed_dim,
            depth=self.MLP_DEPTH,
            num_cells=hidden_dim,
            outscale=self.ENCODER_OUTSCALE,
            device=device,
        )

        # 2. RSSM Prior (Dynamics Transition Model)
        self.rssm_prior = RSSMPriorV3(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            rnn_hidden_dim=hidden_dim,
            num_categoricals=num_categoricals,
            num_classes=num_classes,
            device=device,
        )

        # 3. RSSM Posterior (Representation Model)
        self.rssm_posterior = RSSMPosteriorV3(
            hidden_dim=hidden_dim,
            num_categoricals=num_categoricals,
            num_classes=num_classes,
            rnn_hidden_dim=hidden_dim,
            obs_embed_dim=embed_dim,
            device=device,
        )

        # Features fed to decoders: [state_dim + rnn_hidden_dim]
        feature_dim = self.state_dim + hidden_dim

        # 4. Observation Decoder (Reconstruction)
        self.decoder = DreamerV3MLP(
            in_features=feature_dim,
            out_features=obs_dim,
            depth=self.MLP_DEPTH,
            num_cells=hidden_dim,
            outscale=self.DECODER_OUTSCALE,
            device=device,
        )

        # 5. Reward Predictor Head (Symlog Two-Hot Logits)
        self.reward_head = DreamerV3MLP(
            in_features=feature_dim,
            out_features=num_reward_bins,
            depth=self.MLP_DEPTH,
            num_cells=hidden_dim,
            outscale=self.REWARD_OUTSCALE,
            device=device,
        )

        # 6. Continuation / Discount Predictor Head (Optional)
        self.continue_head = DreamerV3MLP(
            in_features=feature_dim,
            out_features=1,
            depth=self.MLP_DEPTH,
            num_cells=hidden_dim,
            outscale=self.CONTINUE_OUTSCALE,
            device=device,
        )

    def init_state(self, batch_size: int, device=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns initial stochastic state and deterministic belief."""
        state = torch.zeros(batch_size, self.state_dim, device=device)
        belief = torch.zeros(batch_size, self.hidden_dim, device=device)
        return state, belief

    def forward(self, td: TensorDict) -> TensorDict:
        """
        Processes a sequence batch [B, T] or single step TensorDict [B].
        Input keys required in td:
          - 'observation'
          - 'action'
          - 'state'
          - 'belief'
          - ('next', 'observation')
        """
        obs = td.get(("next", "observation"), td.get("observation"))
        action = td.get("action")
        prev_state = td.get("state")
        prev_belief = td.get("belief")

        # Sequence unroll if input has a time dimension (3D: [B, T, D])
        if obs.ndim == 3:
            B, T, _ = obs.shape
            obs_flat = obs.view(B * T, self.obs_dim)
            obs_embed = self.encoder(obs_flat).view(B, T, self.embed_dim)

            curr_state = prev_state[:, 0]
            curr_belief = prev_belief[:, 0]

            prior_logits_list, post_logits_list = [], []
            state_list, belief_list = [], []

            for t in range(T):
                act_t = action[:, t]
                emb_t = obs_embed[:, t]

                p_log, p_st, curr_belief = self.rssm_prior(curr_state, curr_belief, act_t)
                q_log, curr_state = self.rssm_posterior(curr_belief, emb_t)

                prior_logits_list.append(p_log)
                post_logits_list.append(q_log)
                state_list.append(curr_state)
                belief_list.append(curr_belief)

            prior_logits = torch.stack(prior_logits_list, dim=1)
            post_logits = torch.stack(post_logits_list, dim=1)
            next_state = torch.stack(state_list, dim=1)
            next_belief = torch.stack(belief_list, dim=1)

            features = torch.cat([next_state, next_belief], dim=-1)
            feat_flat = features.view(B * T, -1)

            reco_obs = self.decoder(feat_flat).view(B, T, self.obs_dim)
            reward_logits = self.reward_head(feat_flat).view(B, T, -1)
            continue_logits = self.continue_head(feat_flat).view(B, T, -1)

        else: # Single step (2D: [B, D])
            obs_embed = self.encoder(obs)
            prior_logits, prior_state, next_belief = self.rssm_prior(prev_state, prev_belief, action)
            post_logits, next_state = self.rssm_posterior(next_belief, obs_embed)

            features = torch.cat([next_state, next_belief], dim=-1)
            reco_obs = self.decoder(features)
            reward_logits = self.reward_head(features)
            continue_logits = self.continue_head(features)

        # Write predictions and updated states into TensorDict
        td.set(("next", "prior_logits"), prior_logits)
        td.set(("next", "posterior_logits"), post_logits)
        td.set(("next", "state"), next_state)
        td.set(("next", "belief"), next_belief)
        td.set(("next", "reco_observation"), reco_obs)
        td.set(("next", "reward"), reward_logits)
        td.set(("next", "continue_pred"), continue_logits)

        return td


# ==============================================================================
# 3. DreamerV3 Actor & Critic Networks
# ==============================================================================
class DreamerV3Actor(nn.Module):
    """
    Policy Actor: Maps latent features [state + belief] -> Action logits.
    """
    MLP_DEPTH = 2
    ACTOR_OUTSCALE = 0.01  # Small initial logits for high entropy exploration

    def __init__(
        self,
        action_dim: int = 14,
        state_dim: int = 256,
        belief_dim: int = 512,
        hidden_dim: int = 512,
        device=None,
    ):
        super().__init__()
        self.mlp = DreamerV3MLP(
            in_features=state_dim + belief_dim,
            out_features=action_dim,
            depth=self.MLP_DEPTH,
            num_cells=hidden_dim,
            outscale=self.ACTOR_OUTSCALE,
            device=device,
        )

    def forward(self, state: torch.Tensor, belief: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, belief], dim=-1)
        logits = self.mlp(x)
        return logits


class DreamerV3Critic(nn.Module):
    """
    Value Critic: Maps latent features [state + belief] -> Value prediction (Two-Hot or Symlog).
    """
    MLP_DEPTH = 2
    CRITIC_OUTSCALE = 0.0

    def __init__(
        self,
        num_value_bins: int = 255,
        state_dim: int = 256,
        belief_dim: int = 512,
        hidden_dim: int = 512,
        device=None,
    ):
        super().__init__()
        self.mlp = DreamerV3MLP(
            in_features=state_dim + belief_dim,
            out_features=num_value_bins,
            depth=self.MLP_DEPTH,
            num_cells=hidden_dim,
            outscale=self.CRITIC_OUTSCALE,
            device=device,
        )

    def forward(self, state: torch.Tensor, belief: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, belief], dim=-1)
        return self.mlp(x)


# ==============================================================================
# 4. End-to-End Verification Test
# ==============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing DreamerV3 implementation on: {device}")

    OBS_DIM = 64
    ACTION_DIM = 14
    NUM_CATEGORICALS = 16
    NUM_CLASSES = 16
    HIDDEN_DIM = 512
    NUM_BINS = 255

    # 1. Instantiate World Model
    world_model = DreamerV3WorldModel(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        embed_dim=HIDDEN_DIM,
        hidden_dim=HIDDEN_DIM,
        num_categoricals=NUM_CATEGORICALS,
        num_classes=NUM_CLASSES,
        num_reward_bins=NUM_BINS,
        device=device,
    )

    # 2. Instantiate Actor & Critic
    state_dim = NUM_CATEGORICALS * NUM_CLASSES
    actor = DreamerV3Actor(
        action_dim=ACTION_DIM,
        state_dim=state_dim,
        belief_dim=HIDDEN_DIM,
        hidden_dim=HIDDEN_DIM,
        device=device,
    )
    critic = DreamerV3Critic(
        num_value_bins=NUM_BINS,
        state_dim=state_dim,
        belief_dim=HIDDEN_DIM,
        hidden_dim=HIDDEN_DIM,
        device=device,
    )

    # 3. Wrap in TensorDictModules
    actor_module = TensorDictModule(
        actor,
        in_keys=["state", "belief"],
        out_keys=["action_logits"],
    )
    critic_module = TensorDictModule(
        critic,
        in_keys=["state", "belief"],
        out_keys=["state_value"],
    )

    # 4. Create sample batch (Batch=4, Sequence=10)
    B, T = 4, 10
    raw_td = TensorDict({
        "observation": torch.randn(B, T, OBS_DIM, device=device),
        "action": torch.randn(B, T, ACTION_DIM, device=device),
        "state": torch.zeros(B, T, state_dim, device=device),
        "belief": torch.zeros(B, T, HIDDEN_DIM, device=device),
        "next": {
            "observation": torch.randn(B, T, OBS_DIM, device=device),
            "reward": torch.randn(B, T, 1, device=device),
            "done": torch.zeros(B, T, 1, dtype=torch.bool, device=device),
        }
    }, [B, T], device=device)

    # 5. Forward Pass through World Model directly
    out_td = world_model(raw_td.clone())
    print("\n[OK] World Model sequence forward pass successful!")
    print(f"  Reconstructed obs: {out_td.get(('next', 'reco_observation')).shape}")
    print(f"  Predicted reward:  {out_td.get(('next', 'reward')).shape}")
    print(f"  Posterior logits:  {out_td.get(('next', 'posterior_logits')).shape}")
    print(f"  Prior logits:      {out_td.get(('next', 'prior_logits')).shape}")

    # 6. Compute DreamerV3 Model Loss (LossModule calls world_model internally)
    wm_loss_fn = DreamerV3ModelLoss(
        world_model,
        lambda_kl=1.0,
        lambda_reco=1.0,
        lambda_reward=1.0,
        num_reward_bins=NUM_BINS,
        free_bits=1.0,
        global_average=True,  # For vector observation inputs
    )
    wm_loss_fn.set_keys(pixels="observation", reco_pixels="reco_observation")

    loss_td, _ = wm_loss_fn(raw_td.clone())
    print("\n[OK] DreamerV3 Model Loss successfully computed:")
    for k, v in loss_td.items():
        print(f"  {k}: {v.item():.4f}")

    # 7. Actor & Critic forward passes
    actor_td = actor_module(out_td.clone())
    critic_td = critic_module(out_td.clone())
    print(f"\n[OK] Actor logits shape:  {actor_td['action_logits'].shape}")
    print(f"[OK] Critic values shape: {critic_td['state_value'].shape}")
    print("\nAll DreamerV3 components initialized, tested, and working cleanly!")
