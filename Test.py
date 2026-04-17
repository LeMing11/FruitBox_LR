from AC_GAE  import AC_Agent
from Environment import Environment
from colorama import Back, init
import torch

def print_info(env: Environment, state: torch.Tensor, a1: torch.Tensor, a2: torch.Tensor, reward: float, v_s: torch.Tensor, entropy: torch.Tensor, log_prob: torch.Tensor, attempt: int):
    clear_lines = "\033[A\033[2K" * 9
    print(clear_lines, end = "")
    print("-" * 30)
    
    # 현재 상태(s_t)가 아닌 다음 상태(s_{t+1}) 보드를 출력
    start_row, start_col, end_row, end_col = Environment.get_selected_coord(int(a1), int(a2))
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
    print(f"Score: {env.score} | Attempt: {attempt}") # env.score 가정
    print(f"Action (a1.x, a1.y), (a2.x, a2.y): ({start_row}, {start_col}), ({end_row}, {end_col})")
    print(f"Step Reward: {reward}")
    print(f"State Value V(s_t): {v_s.item():.4f}")
    print(f"Policy Entropy: {entropy.item():.4f}")
    print(f"Action LogProb: {log_prob.item():.4f}")
    print("-" * 30)

if __name__ == "__main__":
    init(autoreset=True)
    lvl = 0
    env = Environment(lvl)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = AC_Agent(env, device)

    agent.actor.model.load_state_dict(torch.load(f"./Model/Best_Actor_lvl_{lvl}.pth"))
    agent.critic.model.load_state_dict(torch.load(f"./Model/Best_Critic_lvl_{lvl}.pth"))

    state = env.state
    done = False
    attempt = 0
    total_reward = 0.
    
    log_probs, values, rewards, entropies, dones = [], [], [], [], []

    # 학습 모드 설정
    agent.actor.model.eval()
    agent.critic.model.eval()

    print_info(env, state, 0, 0, 0, torch.tensor(0), torch.tensor(0), torch.tensor(0), 0)

    while not done:
        input('계속하려면 엔터를 입력하십시오')
        attempt += 1

        v_s = agent.critic.get_value(state, len(env.actions))
        a, log_prob, entropy = agent.actor.get_action(state, env.actions)
        a_int = int(a.item())
        act = env.actions[a_int]
        
        reward = env.step(act[0], act[1])
        total_reward += reward
        next_state = env.state

        print_info(env, state, act[0], act[1], reward, v_s, entropy, log_prob, attempt)

        done = len(env.actions) <= 0
        state = next_state