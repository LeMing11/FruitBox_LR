import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Environment_Ensure_Perfect import Environment
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from Model import ActorNet

def custom_collate_fn(batch):
    states = torch.stack([item[0] for item in batch])
    start_nodes = torch.tensor([item[1] for item in batch])
    end_nodes = torch.tensor([item[2] for item in batch])
    action_dicts = [item[3] for item in batch]
    return states, start_nodes, end_nodes, action_dicts

class ExpertDataset(Dataset):
    def __init__(self, map_dir: str, cache_path: str = 'Maps/Imitation/processed_data.pth'):
        self.data = []
        self.max_size = (1, 10, 17)

        if os.path.exists(cache_path):
            print("캐시된 데이터 로드 중...")
            self.data = torch.load(cache_path)
            return
        
        # 1. 맵 데이터 로드
        states = []

        for roots, _, files in os.walk(map_dir):
            for file in files:
                if file.endswith('.npy'):
                    path = os.path.join(roots, file)
                    state = torch.from_numpy(np.load(path))
                    states.append(state)
        
        if not states:
            print("데이터를 찾을 수 없습니다.")
            return
            
        states = torch.cat(states, dim=0)
        
        # 2. 정답 궤적 추출
        for map_data in tqdm(states, total = len(states), desc="데이터 처리 중"):
            board_state = map_data[0].clone().float()
            answer_map = map_data[1].clone().int()
            
            # 정답 맵에서 0보다 큰 인덱스들을 추출하고 내림차순 정렬 (큰 수부터 제거)
            indices = torch.unique(answer_map)
            indices = indices[indices > 0]
            indices = torch.sort(indices, descending=True)[0]

            step_count = 0
            
            for idx in indices:
                # 학습에 사용할 현재 상태 저장 (1, 10, 17)
                state_to_save = board_state.unsqueeze(0).clone()
                
                # 해당 인덱스(사각형)의 좌표 탐색
                coords = torch.nonzero(answer_map == idx)
                if coords.numel() == 0:
                    continue
                    
                min_r = torch.min(coords[:, 0]).item()
                max_r = torch.max(coords[:, 0]).item()
                min_c = torch.min(coords[:, 1]).item()
                max_c = torch.max(coords[:, 1]).item()

                if step_count % 5 == 0:
                    # Environment_Ensure_Perfect.py의 방식과 동일하게 노드 변환
                    start_node = min_c * self.max_size[1] + min_r
                    end_node = max_c * self.max_size[1] + max_r

                    actions = Environment.get_actions(state_to_save, self.max_size)
                    action_dict = Environment.get_actions_dict(actions)

                    self.data.append((state_to_save, start_node, end_node, action_dict))
                
                # 다음 스텝을 위해 보드에서 해당 사각형을 지움(0으로 변경)
                board_state[min_r:max_r+1, min_c:max_c+1] = 0
                step_count += 1

        # 캐시 파일 저장
        torch.save(self.data, cache_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def train_behavior_cloning(actor_model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, epochs: int = 50, lr: float = 1e-4):
    actor_model.to(device)
    optimizer = optim.Adam(actor_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    save_dir = './Model/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(epochs):
        # 학습 (Train) 단계
        actor_model.train()
        train_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for states, start_nodes, end_nodes, action_dicts in train_pbar:
            states = states.to(device) / 10.0
            start_nodes = start_nodes.long().to(device)
            end_nodes = end_nodes.long().to(device)

            optimizer.zero_grad()

            first_actions, state_embed = actor_model.forward_start(states)
            start_mask = torch.full_like(first_actions, -1e9)

            for i, d in enumerate(action_dicts):
                valid_starts = list(d.keys())
                start_mask[i, valid_starts] = 0
            
            first_actions = first_actions + start_mask
            loss_start = criterion(first_actions, start_nodes)

            second_actions = actor_model.forward_end(state_embed, start_nodes)
            end_mask = torch.full_like(second_actions, -1e9)

            for i, d in enumerate(action_dicts):
                valid_ends = d[start_nodes[i].item()]
                end_mask[i, valid_ends] = 0

            second_actions = second_actions + end_mask
            loss_end = criterion(second_actions, end_nodes)

            loss = loss_start + loss_end
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        # 검증 (Validation) 단계
        actor_model.eval()
        val_loss = 0.0

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        with torch.no_grad():
            for states, start_nodes, end_nodes, action_dicts in val_pbar:
                states = states.to(device) / 10.0
                start_nodes = start_nodes.long().to(device)
                end_nodes = end_nodes.long().to(device)

                # 검증 단계에도 동일하게 마스킹 적용
                first_actions, state_embed = actor_model.forward_start(states)
                start_mask = torch.full_like(first_actions, -1e9)
                for i, d in enumerate(action_dicts):
                    valid_starts = list(d.keys())
                    start_mask[i, valid_starts] = 0
                first_actions = first_actions + start_mask
                loss_start = criterion(first_actions, start_nodes)

                second_actions = actor_model.forward_end(state_embed, start_nodes)
                end_mask = torch.full_like(second_actions, -1e9)
                for i, d in enumerate(action_dicts):
                    valid_ends = d[start_nodes[i].item()]
                    end_mask[i, valid_ends] = 0
                second_actions = second_actions + end_mask
                loss_end = criterion(second_actions, end_nodes)

                loss = loss_start + loss_end
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1} Result | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        torch.save(actor_model.state_dict(), os.path.join(save_dir, 'Last_Actor_BC.pth'))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(actor_model.state_dict(), os.path.join(save_dir, 'Best_Actor_BC.pth'))
            print(f"  -> Best_Actor_BC.pth 저장 완료 (Val Loss: {best_val_loss:.4f})")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = ExpertDataset('Maps/Imitation')
    print(f"총 데이터 수: {len(dataset)}")
    
    # 데이터셋을 학습 세트(90%)와 검증 세트(10%)로 분할
    dataset_size = len(dataset)
    val_size = int(dataset_size * 0.1)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, collate_fn=custom_collate_fn)


    actor = ActorNet()
    train_behavior_cloning(actor, train_loader, val_loader, device, epochs=50, lr=1e-4)