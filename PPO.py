import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from colorama import Back, init
from datetime import datetime
from Environment_Ensure_Perfect import Environment
from torch.distributions import Categorical
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Optional

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
    def __init__(self):
        super().__init__()
        self.resnet = ResNet()
        self.attention = AttentionBlock(embed_dim=128, num_heads=4, num_layers=3)

        self.fc_start = MLPHead(in_dim=128, hidden_dim=128, out_dim=1)
        self.action_embed = nn.Embedding(num_embeddings=170, embedding_dim=128)
        self.fc_end = MLPHead(in_dim=256, hidden_dim=256, out_dim=1)

    def forward_start(self, x: torch.Tensor):
        x = self.resnet(x)                                      # (B, 128, 10, 17)
        state_embed = self.attention(x)                         # (B, 170, 128)
        logits_first = self.fc_start(state_embed).squeeze(-1)   # (B, 170, 1) -> (B, 170)

        return logits_first, state_embed

    def forward_end(self, state_embed: torch.Tensor, action_first: torch.Tensor):
        act_embed: torch.Tensor = self.action_embed(action_first.long())    # (B, 1) -> (B, 128)

        act_embed_expanded = act_embed.unsqueeze(1).expand(-1, 170, -1)     # (B, 128) -> (B, 1, 128) -> (B, 170, 128)
        x_combined = torch.cat((state_embed, act_embed_expanded), dim = -1) # (B, 170, 128) + (B, 170, 128) -> (B, 170, 256)

        return self.fc_end(x_combined).squeeze(-1)                          # (B, 170)
    
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

class Actor():
    def __init__(self, device: Optional[torch.device] = None):
        self.model = ActorNet()
        self.device = device or torch.device('cpu')
        self.model.to(self.device)

    def get_action(self, state: torch.Tensor, actions: Dict, deterministic: bool = False):
        state = state.to(self.device) / 10.0
        first_actions, state_embed = self.model.forward_start(state)
        logits_first = first_actions
        mask_first = torch.full_like(logits_first, -1e9).to(self.device)
        valid_starts = list(actions.keys())
        mask_first[0, valid_starts] = 0
        logits_first = logits_first + mask_first
        dist_first = Categorical(logits=logits_first)

        if deterministic:
            action_first = torch.argmax(logits_first, dim=1)
        else:
            action_first = dist_first.sample()
        start_idx = action_first.item()

        valid_ends = actions[start_idx]

        second_actions = self.model.forward_end(state_embed, action_first)
        logits_second = second_actions.clone()
        mask_second = torch.full_like(logits_second, -1e9).to(self.device)
        mask_second[0, valid_ends] = 0
        logits_second = logits_second + mask_second
        dist_second = Categorical(logits=logits_second)

        if deterministic:
            action_second = torch.argmax(logits_second, dim=1)
        else:
            action_second = dist_second.sample()
        
        total_entropy = dist_first.entropy() + dist_second.entropy()
        total_log_prob = dist_first.log_prob(action_first) + dist_second.log_prob(action_second)

        return action_first.item() * 170 + action_second.item(), total_log_prob, total_entropy, mask_first, mask_second

    def evaluate(self, states: torch.Tensor, action_first: torch.Tensor):
        states = states.to(self.device) / 10.0
        policy_1, state_embed = self.model.forward_start(states)
        policy_2 = self.model.forward_end(state_embed, action_first)
        return policy_1, policy_2

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

class PPO_Agent():
    def __init__(self, env: Environment, device: Optional[torch.device] = None, 
                 actor_model_path: Optional[str] = None, critic_model_path: Optional[str] = None, 
                 discount_factor: float = 0.99, gae_lambda: float = 0.95, PPO_clip: float = 0.2,
                 start_entropy: float = 0.1, end_entropy: float = 0.001, lr: float = 1e-4,
                 batch_size: int = 2048, mini_batch_size: int = 128, train_repeat: int = 5, max_epoch: int = 1000,
                 prev_max_score: float = 0, print_freq: int = 5):
        
        self.device = device
        self.env = env
        self.val_env = Environment(validation=True)

        self.actor = Actor(self.device)
        self.critic = Critic(self.device)

        self.actor_optim = optim.Adam(self.actor.model.parameters(), lr=lr)
        self.critic_optim = optim.Adam(self.critic.model.parameters(), lr=lr)

        self.scheduler_actor = CosineAnnealingLR(
            self.actor_optim, T_max=max_epoch, eta_min=5e-6
        )
        self.scheduler_critic = CosineAnnealingLR(
            self.critic_optim, T_max=max_epoch, eta_min=5e-6
        )

        self.critic_loss = nn.SmoothL1Loss()
        self.first_print = True
        self.max_epoch = max_epoch
        self.current_episode = 0

        if actor_model_path != None:
            self.max_avg_score = prev_max_score
            self.actor.model.load_state_dict(torch.load(actor_model_path))

            for param_group in self.actor_optim.param_groups:
                param_group['lr'] = lr
        
        if critic_model_path != None:
            self.critic.model.load_state_dict(torch.load(critic_model_path))

            for param_group in self.critic_optim.param_groups:
                param_group['lr'] = lr


        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda
        self.start_entropy = start_entropy
        self.end_entropy = end_entropy
        self.batch_size = batch_size
        self.print_freq = print_freq
        self.max_avg_score = prev_max_score
        self.train_repeat = train_repeat
        self.mini_batch_size = mini_batch_size
        self.current_epoch = 0
        self.datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.reward_window = deque(maxlen=100)
        self.PPO_clip = PPO_clip
        self.datetime = self.datetime[2:] # "YYMMDD_HHMMSS" 형식으로 변경

        self.writer = SummaryWriter(log_dir=f'./runs/{self.datetime}')

        self.plot_history = {
            'scores': [], 'rewards': [], 
            'actor_loss': [], 'critic_loss': [], 'val_scores': []
        }
        self.batch_history = {
            'state': [], 'action': [], 'can_action': [], 'mask_first': [], 'mask_second': [], 'reward': [], 'value': [], 'done': [], 'log_prob': []
        }

    def print_info(self, episode: int, state: torch.Tensor, a1: int, a2: int, reward: float, v_s: torch.Tensor, entropy: torch.Tensor, log_prob: torch.Tensor, attempt: int):
        clear_lines = "\033[A\033[2K" * 20
        if self.first_print:
            self.first_print = False
        else: print(clear_lines, end = "")
        print(f"--- [Episode {episode} | Batch: {len(self.batch_history['state'])} / {self.batch_size}] ---")
        print(f"Score: {self.env.score} | Attempt: {attempt}")
        print("-" * 30)
        
        start_row, start_col, end_row, end_col = self.env.get_selected_coord(a1, a2)
        rect = torch.zeros_like(state.squeeze(0)[0], dtype=torch.bool)
        rect[start_row : end_row + 1, start_col:end_col + 1] = True

        for row in range(state.shape[2]):
            for col in range(state.shape[3]):
                value = int(state[0][0][row][col])
                value = value if value != 0 else " "
                
                if rect[row][col].cpu().detach():
                    print(f"{Back.RED}{value} ", end='')
                else: print(f"{value}", end=" ")
            print("")


        print("-" * 30)
        print(f"Action (a1.x, a1.y), (a2.x, a2.y): ({start_row}, {start_col}), ({end_row}, {end_col})")
        print(f"Step Reward: {reward}")
        print(f"State Value V(s_t): {v_s.item():.4f}")
        print(f"Policy Entropy: {entropy.item():.4f}")
        print(f"Action LogProb: {log_prob.item():.4f}")
        print("-" * 30)

    def train_start(self):
        self.actor_optim.zero_grad(set_to_none=True)
        self.critic_optim.zero_grad(set_to_none=True)

        while self.current_epoch < self.max_epoch:
            self.episode()
        
        save_dir = f'./Model/{self.datetime}/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        torch.save(self.actor.model.state_dict(), f'{save_dir}/Last_Actor_{self.datetime}.pth')
        torch.save(self.critic.model.state_dict(), f'{save_dir}/Last_Critic_{self.datetime}.pth')

    def episode(self):
        self.env.reset()

        state = self.env.state.unsqueeze(0) if self.env.state.dim() == 3 else self.env.state  # 배치 차원 추가
        done = False
        attempt = 0
        total_reward = 0.

        while not done:
            attempt += 1

            with torch.no_grad():
                v_s = self.critic.get_value(state, torch.tensor(len(self.env.actions)))
                action_dict = self.env.get_actions_dict()
                a, log_prob, entropy, mask_first, mask_second = self.actor.get_action(state, action_dict)

            a = int(a)
            start_node = a // 170
            end_node = a % 170

            can_action = len(self.env.actions)
            reward = self.env.step(start_node, end_node)
            total_reward += reward
            next_state = self.env.state.unsqueeze(0) if self.env.state.dim() == 3 else self.env.state
            
            if attempt % self.print_freq == 0:
                self.print_info(self.current_episode, state, start_node, end_node, reward, v_s, entropy, log_prob, attempt)

            done = len(self.env.actions) <= 0

            self.batch_history['state'].append(state)
            self.batch_history['action'].append(torch.tensor(start_node * 170 + end_node))
            self.batch_history['can_action'].append(can_action)
            self.batch_history['mask_first'].append(mask_first.squeeze(0).cpu())
            self.batch_history['mask_second'].append(mask_second.squeeze(0).cpu())
            self.batch_history['reward'].append(reward)
            self.batch_history['value'].append(v_s.squeeze())
            self.batch_history['done'].append(done)
            self.batch_history['log_prob'].append(log_prob.squeeze().detach().cpu())

            if len(self.batch_history['state']) >= self.batch_size:
                states = torch.cat(self.batch_history['state'], dim=0).to(self.device).detach()
                actions = torch.stack(self.batch_history['action']).to(self.device)
                can_actions = torch.tensor(self.batch_history['can_action'], dtype=torch.float32).to(self.device)
                mask_first = torch.stack(self.batch_history['mask_first']).to(self.device)
                mask_second = torch.stack(self.batch_history['mask_second']).to(self.device)
                rewards = torch.tensor(self.batch_history['reward'], dtype=torch.float32).to(self.device)
                values = torch.stack(self.batch_history['value']).to(self.device).detach()
                log_probs = torch.stack(self.batch_history['log_prob']).to(self.device).detach()
                dones = self.batch_history['done']

                self.train(states, actions, can_actions, mask_first, mask_second, rewards, values, dones, log_probs, next_state)
                avg_score = self.validation()

                self.plot_history['val_scores'].append(avg_score)
                self.writer.add_scalar("Score/Validation", avg_score, self.current_epoch)
                self.plot_graph()
                self.current_epoch += 1

                if avg_score > self.max_avg_score:
                    new_line = "\n" * 20
                    remove_line = "\033[A\033[2K" * 20
                    print(f"{remove_line}[{self.current_epoch} / {self.max_epoch}] New high score, Score: {avg_score:.2f}. save the model... {new_line}")
                    self.max_avg_score = avg_score
                    
                    save_dir = f'./Model/{self.datetime}/'
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)

                    torch.save(self.actor.model.state_dict(), f'{save_dir}/Best_Actor_{self.datetime}.pth')
                    torch.save(self.critic.model.state_dict(), f'{save_dir}/Best_Critic_{self.datetime}.pth')

            state = next_state
        
        self.print_info(self.current_episode, state, start_node, end_node, reward, v_s, entropy, log_prob, attempt)
        
        self.reward_window.append(total_reward)
        avg_reward = sum(self.reward_window) / len(self.reward_window)
        self.plot_history['rewards'].append(avg_reward)
        self.writer.add_scalar("Reward/Total", avg_reward, self.current_episode)
        self.current_episode += 1

    def plot_graph(self):
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        fig.subplots_adjust(hspace=0.4)

        ax[0][0].plot(self.plot_history['val_scores'])
        ax[0][0].set_title("Validation Score")

        ax[0][1].plot(self.plot_history['rewards'])
        ax[0][1].set_title("Total Reward")

        ax[1][0].plot(self.plot_history['actor_loss'])
        ax[1][0].set_title("Actor Loss")

        ax[1][1].plot(self.plot_history['critic_loss'])
        ax[1][1].set_title("Critic Loss")

        plt.savefig(f"./Graph/graph_{self.datetime}.png")
        plt.close()

    def get_gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: List, next_state):
        advantages = []
        gae = torch.tensor(0., dtype=torch.float32, device=self.device)

        with torch.no_grad():
            if dones[-1]:
                next_value = torch.tensor(0., dtype=torch.float32, device=self.device)
            else:
                next_value = self.critic.get_value(next_state, torch.tensor(len(self.env.actions), device=self.device)).squeeze().detach()
        
        for t in reversed(range(len(rewards))):
            reward_t = rewards[t] if isinstance(rewards[t], torch.Tensor) else torch.tensor(rewards[t], dtype=torch.float32, device=self.device)
            value_t = values[t] if isinstance(values[t], torch.Tensor) else torch.tensor(values[t], dtype=torch.float32, device=self.device)

            is_not_terminal = torch.tensor(1.0 - dones[t], dtype=torch.float32, device=self.device)
            
            # GAE 계산
            delta = reward_t + self.discount_factor * next_value * is_not_terminal - value_t
            gae = delta + self.discount_factor * self.gae_lambda * gae * is_not_terminal
            
            advantages.insert(0, gae)
            next_value = value_t

        advantages_tensor = torch.stack(advantages)
        
        return advantages_tensor

    def train(self, states: torch.Tensor, actions: torch.Tensor, can_actions: torch.Tensor, mask_first: torch.Tensor, mask_second: torch.Tensor, rewards: torch.Tensor, values: torch.Tensor, dones: List, log_prob: torch.Tensor, next_state: torch.Tensor):
        advantages = self.get_gae(rewards, values, dones, next_state)
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        old_log_probs = torch.stack(self.batch_history['log_prob']).to(self.device)
        
        # PPO 학습을 여러 epoch 동안 반복
        for repeat in range(self.train_repeat):
            # 미니배치로 나누어 학습
            batch_indices = torch.randperm(len(states))
            
            actor_loss_sum = 0.0
            critic_loss_sum = 0.0
            entropy_sum = 0.0
            
            for i in range(0, len(states), self.mini_batch_size):
                batch_idx = batch_indices[i:min(i+self.mini_batch_size, len(states))]
                
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_can_actions = can_actions[batch_idx]
                
                # ============= Critic 학습 =============
                batch_values = self.critic.evaluate(batch_states, batch_can_actions)
                critic_loss = self.critic_loss(batch_values.view(-1), batch_returns.view(-1))
                
                # Critic 역전파
                self.critic_optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.model.parameters(), 1.0)
                self.critic_optim.step()
                
                critic_loss_sum += critic_loss.item()
                
                # ============= Actor 학습 =============
                self.actor.model.train()
                
                # 현재 정책의 log probability 계산
                action_1 = (batch_actions // 170).long()
                action_2 = (batch_actions % 170).long()
                policy_1, policy_2 = self.actor.evaluate(batch_states, action_1)

                logits_1 = policy_1 + mask_first[batch_idx]
                logits_2 = policy_2 + mask_second[batch_idx]

                logits_1 = torch.clamp(logits_1, min=-1e9, max=1e9)
                logits_2 = torch.clamp(logits_2, min=-1e9, max=1e9)

                dist_1 = Categorical(logits=logits_1)
                dist_2 = Categorical(logits=logits_2)

                curr_log_probs_1 = dist_1.log_prob(action_1)
                curr_log_probs_2 = dist_2.log_prob(action_2)
                curr_log_probs = curr_log_probs_1 + curr_log_probs_2
                
                # Entropy 계산
                entropy = dist_1.entropy() + dist_2.entropy()
                
                # PPO Clipped Surrogate Loss
                ratio = torch.exp(curr_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.PPO_clip, 1.0 + self.PPO_clip) * batch_advantages
                
                # Entropy coefficient 감소 (탐험량 감소)
                progress = (self.current_epoch / self.max_epoch)
                end = self.end_entropy
                start = self.start_entropy
                current_entropy_coef = end + (start - end) * (1 - progress) ** 2
                
                actor_loss = -torch.min(surr1, surr2).mean() - current_entropy_coef * entropy.mean()
                
                # Actor 역전파
                self.actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.model.parameters(), 1.0)
                self.actor_optim.step()
                
                actor_loss_sum += actor_loss.item()
                entropy_sum += entropy.mean().item()
            
            # 배치별 평균 손실 저장
            avg_actor_loss = actor_loss_sum / max((len(states) + self.mini_batch_size - 1) // self.mini_batch_size, 1)
            avg_critic_loss = critic_loss_sum / max((len(states) + self.mini_batch_size - 1) // self.mini_batch_size, 1)
            
            self.plot_history['actor_loss'].append(avg_actor_loss)
            self.plot_history['critic_loss'].append(avg_critic_loss)
            self.writer.add_scalar("Loss/Actor", avg_actor_loss, self.current_epoch)
            self.writer.add_scalar("Loss/Critic", avg_critic_loss, self.current_epoch)
        
        # 배치 히스토리 초기화
        self.batch_history = {
            'state': [], 'action': [], 'can_action': [], 'mask_first': [], 'mask_second': [], 'reward': [], 'value': [], 'done': [], 'log_prob': []
        }

        self.scheduler_actor.step()
        self.scheduler_critic.step()
           
    def validation(self, val_size: int = 32):
        self.actor.model.eval()
        total_score = 0
        iteration = 0

        with torch.no_grad():
            while iteration < val_size:
                self.val_env.reset()
                done = False

                while not done:
                    state = self.val_env.state.unsqueeze(0).to(self.device)
                    action_dict = self.val_env.get_actions_dict()
                    
                    action, _, _, _, _ = self.actor.get_action(state, action_dict, deterministic=True)
                    action = int(action)
                    start_node = action // 170
                    end_node = action % 170
                    
                    self.val_env.step(start_node, end_node)
                    next_state = self.val_env.state

                    done = len(self.val_env.actions) <= 0
                    state = next_state

                total_score += self.val_env.score
                iteration += 1
        
        self.actor.model.train()
        avg_score = total_score / val_size
        return avg_score

if __name__ == "__main__":
    init(autoreset=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = Environment()
    agent = PPO_Agent(env, device, batch_size=2048, lr=1e-4, start_entropy=5e-2, end_entropy=1e-3, print_freq=100)
    # agent = PPO_Agent(env, device, batch_size=2048, lr=5e-5, print_freq=100, 
    #                   start_entropy=1e-3, end_entropy=1e-7,
    #                   actor_model_path="./Model/260224_164211/Best_Actor_lvl4_260224_164211.pth",
    #                   critic_model_path="./Model/260224_164211/Best_Critic_lvl4_260224_164211.pth",
    #                   prev_max_score=109.25)
    agent.train_start()