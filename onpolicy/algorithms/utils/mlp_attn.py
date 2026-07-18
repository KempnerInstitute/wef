import torch
import torch.nn as nn
from .util import init, get_clones
from torch.nn import functional as F

"""MLP modules."""

class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        # self.fc_h = nn.Sequential(init_(
        #     nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        # self.fc2 = get_clones(self.fc_h, self._layer_N)
        self.fc2 = nn.ModuleList([nn.Sequential(init_(
            nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size)) for i in range(self._layer_N)])

    def forward(self, x):
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        return x


class MLPBase(nn.Module):
    def __init__(self, args, obs_shape, cat_self=True, attn_internal=False):
        super(MLPBase, self).__init__()

        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._stacked_frames = args.stacked_frames
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size

        obs_dim = obs_shape[0]

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.mlp = MLPLayer(obs_dim, self.hidden_size,
                              self._layer_N, self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)

        return x
    
class AttnMLPBase(MLPBase):
    def __init__(self, args, obs_shape, attn_mode):
        super().__init__(args, obs_shape)

        self.attn_mode = attn_mode
        self.use_softmax = getattr(args, "attn_use_softmax", False)
        self.hidden_size = args.hidden_size
        if attn_mode == "x":
            input_width = obs_shape[0]
        elif attn_mode == "hx": 
            input_width = args.hidden_size
        elif attn_mode == "x+hx": 
            input_width = obs_shape[0] + args.hidden_size
        else:
            raise ValueError(f"Unknown attention mode: {attn_mode}")

        self.attn_mlp = nn.Sequential(
            nn.Linear(input_width, input_width),
            nn.Tanh(),
            nn.Linear(input_width, obs_shape[0])
        )
        self.last_attn_mask = None  # for logging

        # Define the bottleneck that compresses then expands
        self.bottleneck_width = 24
        self.bottleneck = nn.Sequential(
            nn.Linear(obs_shape[0], self.bottleneck_width),
            nn.Tanh(),  
            nn.Linear(self.bottleneck_width, obs_shape[0]),
            nn.Tanh(),
        )

    def apply_attention(self, x, hxs=None):
        # x: [batch*T, obs_dim]
        # hxs: [batch, 1, hidden_dim] or [batch*T, 1, hidden_dim] depending on where it's called

        if self.attn_mode == "x":
            attn_input = x

        elif self.attn_mode in ["hx", "x+hx"]:
            if hxs is None:
                raise ValueError("Attention mode requires hxs, but got None.")

            if hxs.dim() == 3:
                hxs = hxs.squeeze(1)  # [B, H]

            # GPT: RNN state (hxs) is shaped differently during training vs rollout. 
            # During training, especially in the PPO update step, the shape of obs and hxs diverges 
            # due to sequence flattening. 
            if hxs.shape[0] != x.shape[0]:
                # In training, x is flattened across time; match hxs
                repeat_factor = x.shape[0] // hxs.shape[0]
                hxs = hxs.repeat_interleave(repeat_factor, dim=0)

            if self.attn_mode == "hx":
                attn_input = hxs
            elif self.attn_mode == "x+hx":
                attn_input = torch.cat([x, hxs], dim=-1)
        else:
            raise ValueError(f"Unknown attention mode: {self.attn_mode}")

        logits = self.attn_mlp(attn_input)
        mask = F.softmax(logits, dim=-1) if self.use_softmax else torch.sigmoid(logits)
        self.last_attn_mask = mask.detach().cpu()
        # print(f"[ATTN] x: {x.shape}, hxs: {hxs.shape}, attn_input: {attn_input.shape}, mask: {mask.shape}")
        return x * mask



    def forward(self, x, hxs=None):
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        if self.attn_mode is not None:
            x = self.apply_attention(x, hxs=hxs)

        # x = self.bottleneck(x) 

        x = self.mlp(x)
        return x
