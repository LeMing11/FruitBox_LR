import torch.nn as nn
import torch
from torch.distributions import Categorical
from typing import Dict, Optional

class SimpleConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.skip = nn.Conv2d(in_channels, out_channels, 1, padding="same") if in_channels != out_channels else nn.Identity()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding="same")
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding="same")
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_con = self.skip(x)
        y = self.conv1(x)
        y = self.relu(y)                            
        y = self.conv2(y)
        return self.relu(y + skip_con)

class ResNet(nn.Module):
    def __init__(self, out_channels: int = 128):
        super().__init__()
        self.conv1 = SimpleConvBlock(1, out_channels)
        self.conv2 = SimpleConvBlock(out_channels, out_channels)
        self.conv3 = SimpleConvBlock(out_channels, out_channels)
    
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x
    
class AttentionBlock(nn.Module):
    def __init__(self, embed_dim: int = 128, num_heads: int = 4, num_layers: int = 3, seq_len: int = 170):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            dropout=0,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2).transpose(1,2)             # (B, 170, 128)

        x = x + self.pos_embed                      # 위치 정보 결합
        x = self.transformer_encoder(x)             # (B, 170, 128)
        return x

class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)      # (B, outdim, ...)

class ActorNet(nn.Module):
    def __init__(self, board_size: tuple = (1, 10, 17)):
        _, h, w = board_size
        num_actions = (h * (h + 1) // 2) * (w * (w + 1) // 2) - (h * w)

        super().__init__()
        self.resnet = ResNet()
        self.attention = AttentionBlock(embed_dim=128, num_heads=4, num_layers=3, seq_len=h * w)
        self.fc = MLPHead(in_dim=128, hidden_dim=256, out_dim=num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet(x)                              # (B, 128, 10, 17)
        x = self.attention(x)                           # (B, 170, 128)
        board_embed = x.mean(dim=1)                     # (B, 128)
        logits = self.fc(board_embed)                   # (B, num_actions)

        return logits

class Actor():
    def __init__(self, board_size: tuple, device: Optional[torch.device] = None):
        self.device = device or torch.device('cpu')
        self.model = ActorNet(board_size).to(self.device)

    def get_action(self, state: torch.Tensor, actions: Dict, deterministic: bool = False):
        state = state.to(self.device) / 10.0
        logits = self.model(state)

        mask = torch.full_like(logits, -1e9).to(self.device)
        valid_actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        mask[:, valid_actions] = 0
        masked_logits = logits + mask
        dist = Categorical(logits=masked_logits)

        if deterministic:
            action = masked_logits.argmax(dim=-1)
        else:
            action = dist.sample()

        return (action.item(), dist.log_prob(action), dist.entropy(), mask)

    def evaluate(self, states: torch.Tensor) -> torch.Tensor:
        states = states.to(self.device) / 10.0
        return self.model(states)

class CriticNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = ResNet()
        self.attention = AttentionBlock(embed_dim=128, num_heads=4, num_layers=3)
        self.fc_value = MLPHead(in_dim=129, hidden_dim=256, out_dim=1)
    
    def forward(self, x: torch.Tensor, x_num_actions: torch.Tensor):
        x = self.resnet(x)                          # (B, 128, 10, 17)
        x = self.attention(x)                       # (B, 170, 128)

        x = x.mean(dim=1)                           # (B, 128)
        x = torch.cat((x, x_num_actions), dim=1)    # (B, 129)

        return self.fc_value(x)                     # (B, 1)

class Critic():
    def __init__(self, device: Optional[torch.device] = None):
        self.model = CriticNet()
        self.device = device or torch.device('cpu')
        self.model.to(self.device)
    
    def get_value(self, state: torch.Tensor, num_actions: torch.Tensor):
        return self.evaluate(state, num_actions)

    def evaluate(self, states: torch.Tensor, num_actions: torch.Tensor):
        if num_actions.dtype != torch.float32:
            num_actions = num_actions.float()

        states = states.to(self.device) / 10.0
        num_actions_tensor = torch.log(num_actions.to(self.device) + 1.0).view(-1, 1)
        
        return self.model(states, num_actions_tensor)
