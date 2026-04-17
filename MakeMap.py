import os

import numpy as np
import random
import tqdm
import argparse
from scipy.special import softmax


parser = argparse.ArgumentParser()
parser.add_argument('--num_maps', type=int, default=500000, help='Number of maps to generate')
parser.add_argument('--num_split', type=int, default=10000, help='Number of splits for saving maps')
parser.add_argument('--save_dir', type=str, default='./Maps', help='Directory to save generated maps')
parser.add_argument('--save_answer', type=bool, default=True, help='Whether to save the answer maps')
args = parser.parse_args()
num_maps = args.num_maps
num_split = args.num_split
save_dir = args.save_dir
save_answer = args.save_answer
using_directly = False

map_size = (10, 17)
combination = {
    2: [[1, 9], [2, 8], [3, 7], [4, 6], [5, 5]],
    3: [[1, 1, 8], [1, 2, 7], [1, 3, 6], [1, 4, 5], [2, 2, 6], [2, 3, 5], [2, 4, 4], [3, 3, 4]],
    4: [[1, 1, 1, 7], [1, 1, 2, 6], [1, 1, 3, 5], [1, 1, 4, 4], [1, 2, 2, 5], [1, 2, 3, 4], [1, 3, 3, 3], [2, 2, 2, 4], [2, 2, 3, 3]],
    5: [[1, 1, 1, 1, 6], [1, 1, 1, 2, 5], [1, 1, 1, 3, 4], [1, 1, 2, 2, 4], [1, 1, 2, 3, 3], [1, 2, 2, 2, 3], [2, 2, 2, 2, 2]],
    6: [[1, 1, 1, 1, 1, 5], [1, 1, 1, 1, 2, 4], [1, 1, 1, 1, 3, 3], [1, 1, 1, 2, 2, 3], [1, 1, 2, 2, 2, 3], [1, 2, 2, 2, 2, 2]],
    7: [[1, 1, 1, 1, 1, 1, 4], [1, 1, 1, 1, 1, 2, 3], [1, 1, 1, 1, 2, 2, 2]],
    8: [[1, 1, 1, 1, 1, 1, 1, 3], [1, 1, 1, 1, 1, 1, 2, 2]],
    9: [[1, 1, 1, 1, 1, 1, 1, 1, 2]],
    10: [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
}

T_list, L_list, B_list, R_list = [], [], [], []
for top in range(map_size[0]):
    for left in range(map_size[1]):
        for bottom in range(top, map_size[0]):
            for right in range(left, map_size[1]):
                if top == bottom and left == right:
                    continue
                T_list.append(top)
                L_list.append(left)
                B_list.append(bottom)
                R_list.append(right)

T_arr = np.array(T_list, dtype=np.int32)
L_arr = np.array(L_list, dtype=np.int32)
B_arr = np.array(B_list, dtype=np.int32)
R_arr = np.array(R_list, dtype=np.int32)
C_arr = (B_arr - T_arr + 1) * (R_arr - L_arr + 1)

def can_insert(map, r1, c1, r2, c2):
    if r1 == r2 and c1 == c2:
        return False
    return map[1, r1:r2+1, c1:c2+1].sum() == 0

def get_elements(num_fill):
    return combination[num_fill][random.randint(0, len(combination[num_fill]) - 1)]

def generate_map():
    map = np.zeros((2, *map_size), dtype=np.uint8)

    map_list = []
    total_created = 0
    with tqdm.tqdm(total=num_maps) as pbar:
        while total_created < num_maps:
            idx = 1
            while True:
                occ = (map[1] != 0).astype(np.int32)
                P = np.zeros((map_size[0] + 1, map_size[1] + 1), dtype=np.int32)
                P[1:, 1:] = occ.cumsum(axis=0).cumsum(axis=1)

                area_sums = P[B_arr + 1, R_arr + 1] - P[T_arr, R_arr + 1] - P[B_arr + 1, L_arr] + P[T_arr, L_arr]
                
                valid_mask = (area_sums == 0)
                valid_indices = np.nonzero(valid_mask)[0]

                if valid_indices.size == 0:
                    break

                random_index = valid_indices[random.randint(0, len(valid_indices) - 1)]
                top, left, bottom, right = T_arr[random_index], L_arr[random_index], B_arr[random_index], R_arr[random_index]
                cell_count = (bottom - top + 1) * (right - left + 1)
                
                num_fill = min(random.randint(2, 10), cell_count)

                fill_prob = [i for i in range(num_fill, 1, -1)]
                fill_prob = softmax(fill_prob)

                num_fill = np.random.choice(range(2, num_fill + 1), 1, p=fill_prob)[0]
                selected = np.random.choice(cell_count, num_fill, replace=False)

                for i in range(num_fill):
                    s = selected[i]
                    r = top + s // (right - left + 1)
                    c = left + s % (right - left + 1)
                    map[1, r, c] = idx
                
                idx += 1

            zero_pos = np.where(map[1] == 0)

            for i in range(len(zero_pos[0])):
                r = zero_pos[0][i]
                c = zero_pos[1][i]
                
                directions = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
                candidate = 0

                for dr, dc in directions:
                    if 0 <= dr < map_size[0] and 0 <= dc < map_size[1] and map[1, dr, dc] != 0:
                        crt_idx = map[1, dr, dc]
                        map[1, r, c] = crt_idx

                        rect_idx = map[1, dr, dc]
                        rect_range = list(np.where(map[1] == rect_idx))

                        if rect_range[0].size >= 10:
                            break

                        rect_range[0] = np.append(rect_range[0], r)
                        rect_range[1] = np.append(rect_range[1], c)
                        
                        top, bottom = rect_range[0].min(), rect_range[0].max()
                        left, right = rect_range[1].min(), rect_range[1].max()
                        rect_idxs = set(map[1, top:bottom+1, left:right+1].flatten())
                        
                        if all(crt_idx <= x for x in rect_idxs):
                            if candidate < crt_idx:
                                candidate = crt_idx
                
                map[1, r, c] = candidate

            zero_pos = np.where(map[1] == 0)

            for i in range(len(zero_pos[0])):
                r = zero_pos[0][i]
                c = zero_pos[1][i]
                directions = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]

                for dr, dc in directions:
                    if 0 <= dr < map_size[0] and 0 <= dc < map_size[1] and map[1, dr, dc] != 0:
                        rect_idx = map[1, dr, dc]
                        rect_elems = (map[1] == rect_idx).sum()

                        if rect_elems > 2:
                            map[1, dr, dc] = idx
                            map[1, r, c] = idx
                            
                            idx += 1
                            break

            if all(map[1].flatten() != 0):
                for i in range(1, idx):
                    coords = np.where(map[1] == i)
                    num_fill = len(coords[0])
                    elems = get_elements(num_fill)

                    for j in range(num_fill):
                        r = coords[0][j]
                        c = coords[1][j]
                        map[0, r, c] = elems[j]

                total_created += 1

                save_map = map.astype(np.uint8).copy()
                if not save_answer:
                    save_map = save_map[0]

                map_list.append(save_map)
                pbar.update(1)

                if total_created % num_split == 0 and not using_directly:
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    np.save(f'{save_dir}/maps_{total_created // num_split}.npy', np.array(map_list))
                    map_list = []
                    
            map.fill(0)

    return map_list


if __name__ == "__main__":
    generate_map()