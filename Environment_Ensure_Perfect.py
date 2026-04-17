import torch
import random
import os
import numpy as np
import MakeMap
from numba import jit

@jit(nopython=True)
def find_rects_sum_equals(ps, H, W, target=10):
    results = []

    for top in range(0, H):
        for left in range(0, W):
            for bottom in range(top, H):
                for right in range(left, W):
                    s = ps[bottom + 1, right + 1] - ps[top, right + 1] - ps[bottom + 1, left] + ps[top, left]

                    if s == target:
                        results.append((top, left, bottom, right))
                    elif s > target:
                        break

    return results

class Environment():
    def load_states(self):
        if self.validation:
            MakeMap.num_maps = 32
            MakeMap.using_directly = True
            self.states = MakeMap.generate_map()
            self.states = [torch.from_numpy(s).unsqueeze(0) for s in self.states]

        else:
            for roots, _, files in os.walk(self.map_dir):
                for file in files:
                    if file.endswith('.npy'):
                        path = os.path.join(roots, file)
                        state = torch.from_numpy(np.load(path))
                        self.states.append(state)

        self.states = torch.cat(self.states, dim=0)

    def get_selected_coord(self, pos1: int, pos2: int):
        r1 = pos1 % self.max_size[1]
        c1 = pos1 // self.max_size[1]

        r2 = pos2 % self.max_size[1]
        c2 = pos2 // self.max_size[1]

        start_row = min(r1, r2)
        end_row = max(r1, r2)

        start_col = min(c1, c2)
        end_col = max(c1, c2)

        return start_row, start_col, end_row, end_col

    def reset(self):
        self.score = 0
        state_idx = random.randint(0, len(self.states) - 1) if not self.validation else self.val_idx
        self.state = self.states[state_idx][0].type(torch.float32).unsqueeze(0).clone()
        self.actions = self.get_actions(self.state)

        if self.validation:
            self.val_idx = (self.val_idx + 1) % len(self.states)
    
    def step(self, pos1: int, pos2: int):
        start_row, start_col, end_row, end_col = self.get_selected_coord(pos1, pos2)
        selected = self.state[0][start_row : end_row + 1, start_col : end_col + 1]
        cnt_nonzero = torch.count_nonzero(selected).item()
        cnt_nonzero = max(int(cnt_nonzero), 1)

        if selected.sum().item() == 10:
            self.score += cnt_nonzero

            self.state[0][start_row : end_row + 1, start_col : end_col + 1] = 0
            self.actions = self.get_actions(self.state)
            score_reward = 0.1

            if self.state[0].sum().item() == 0:
                score_reward += 20.0
            elif len(self.actions) == 0:
                remaining_blocks = torch.count_nonzero(self.state[0]).item()
                score_reward -= (remaining_blocks / 10.0)
            
            return score_reward
    
        else: return -0.05
    
    def make_prefix_sum(self, tensor: torch.Tensor):
        _, H, W = tensor.shape
        ps = torch.zeros((H + 1, W + 1), dtype = torch.int64)
        
        for i in range(H):
            row_cum = 0
            for j in range(W):
                row_cum += tensor[0][i, j]
                ps[i + 1, j + 1] = ps[i, j + 1] + row_cum
        
        return ps

    def get_actions(self, tensor: torch.Tensor):
        ps = self.make_prefix_sum(tensor)
        ps_np = ps.cpu().numpy()
        _, h, w = tensor.shape
        rects = find_rects_sum_equals(ps_np, int(h), int(w))
        actions = set()
        
        for rect in rects:
            (r1, c1, r2, c2) = rect
            sub = tensor[0][r1:r2 + 1, c1:c2 + 1]
            indices_rel = torch.nonzero(sub, as_tuple=True)

            if indices_rel[0].numel() == 0:
                continue
            
            min_r = torch.min(indices_rel[0]) + r1
            max_r = torch.max(indices_rel[0]) + r1
            min_c = torch.min(indices_rel[1]) + c1
            max_c = torch.max(indices_rel[1]) + c1
            actions.add((int(min_c * self.max_size[1] + min_r), int(max_c * self.max_size[1] + max_r)))

        return sorted(list(actions))
    
    def get_actions_dict(self):
        actions = self.actions
        result = {}

        for action in actions:
            if not action[0] in result.keys():
                result[action[0]] = []

            result[action[0]].append(action[1])
        
        return result

    def __init__(self, validation: bool = False, map_dir: str = './Maps'):
        self.states = []
        self.validation = validation
        self.val_idx = 0
        self.map_dir = map_dir
        self.load_states()
        self.max_size = (1, 10, 17)
        self.reset()

if __name__ == "__main__":
    env = Environment(validation=True)
    print(env.state)
    print("="*30)
    while (True):
        print(env.actions)
        p1 = int(input("P1 입력: "))
        p2 = int(input("P2 입력: "))
        env.step(p1, p2)
        print(env.state)