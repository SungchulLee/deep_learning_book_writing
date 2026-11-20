# DQN 구현

실제로 쓸 수 있는 DQN 구현은 경험 리플레이, 타깃 네트워크, 그래디언트 클리핑, 체계적인 평가를 하나의 학습 흐름으로 모은다. 이 구현은 로스·Q 값·그래디언트 노름의 로깅, 그리디 폴리시(늘 Q 값이 가장 큰 행동을 고르는 폴리시, 곧 $\varepsilon = 0$)로 하는 주기적 평가, 학습을 이어 가거나 결과를 다시 불러올 수 있게 하는 모델 체크포인트를 담는다. 후버 로스와 흔히 쓰는 하이퍼파라미터로 CartPole-v1 에서 학습 루프 전체를 보여 준다.

## 1. 코드

??? note "코드 (408줄)"

    ```python
    """
    35.1.4 DQN 짜기
    ==========================

    적기까지 갖춘, 참으로 굴릴 수 있는 온전한 DQN 짜기,
    CartPole-v1에서 값 매기기와 중간 갈무리.
    """

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import gymnasium as gym
    from collections import deque, namedtuple
    from typing import Tuple, List, Dict, Optional
    import random
    import time
    import json
    import os

    # ========================================================================
    # 메인
    # ========================================================================

    Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


    # ---------------------------------------------------------------------------
    # 되돌려 보기 버퍼
    # ---------------------------------------------------------------------------

    class ReplayBuffer:
        def __init__(self, capacity: int, state_dim: int):
            self.capacity = capacity
            self.size = 0
            self.ptr = 0
            self.states = np.zeros((capacity, state_dim), dtype=np.float32)
            self.actions = np.zeros(capacity, dtype=np.int64)
            self.rewards = np.zeros(capacity, dtype=np.float32)
            self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
            self.dones = np.zeros(capacity, dtype=np.float32)

        def push(self, state, action, reward, next_state, done):
            self.states[self.ptr] = state
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.next_states[self.ptr] = next_state
            self.dones[self.ptr] = float(done)
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

        def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
            idx = np.random.randint(0, self.size, size=batch_size)
            return (
                torch.FloatTensor(self.states[idx]),
                torch.LongTensor(self.actions[idx]),
                torch.FloatTensor(self.rewards[idx]),
                torch.FloatTensor(self.next_states[idx]),
                torch.FloatTensor(self.dones[idx]),
            )

        def __len__(self):
            return self.size


    # ---------------------------------------------------------------------------
    # Q 네트워크
    # ---------------------------------------------------------------------------

    class QNetwork(nn.Module):
        def __init__(self, state_dim: int, action_dim: int,
                     hidden_dims: Tuple[int, ...] = (128, 128)):
            super().__init__()
            layers = []
            prev = state_dim
            for h in hidden_dims:
                layers.extend([nn.Linear(prev, h), nn.ReLU()])
                prev = h
            layers.append(nn.Linear(prev, action_dim))
            self.net = nn.Sequential(*layers)

            # 사비에르 초기화
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


    # ---------------------------------------------------------------------------
    # 온전한 DQN 에이전트
    # ---------------------------------------------------------------------------

    class DQNAgent:
        """겪음 되돌려 보기, 타깃 네트워크, 기록을 갖춘 온전한 DQN 에이전트."""

        def __init__(self, state_dim: int, action_dim: int,
                     # 네트워크
                     hidden_dims: Tuple[int, ...] = (128, 128),
                     # 학습
                     lr: float = 1e-3,
                     gamma: float = 0.99,
                     batch_size: int = 64,
                     # 되돌려 보기 버퍼
                     buffer_capacity: int = 50000,
                     min_buffer_size: int = 1000,
                     # 타깃 네트워크
                     target_update_freq: int = 200,
                     # 살펴보기
                     eps_start: float = 1.0,
                     eps_end: float = 0.01,
                     eps_decay_steps: int = 10000,
                     # 그 밖
                     grad_clip: float = 10.0,
                     loss_fn: str = 'huber',  # 'mse' 또는 'huber'
                     device: str = 'cpu'):

            self.state_dim = state_dim
            self.action_dim = action_dim
            self.gamma = gamma
            self.batch_size = batch_size
            self.min_buffer_size = min_buffer_size
            self.target_update_freq = target_update_freq
            self.grad_clip = grad_clip
            self.device = device

            # 엡실론 차례표
            self.eps_start = eps_start
            self.eps_end = eps_end
            self.eps_decay_steps = eps_decay_steps

            # 네트워크
            self.online_net = QNetwork(state_dim, action_dim, hidden_dims).to(device)
            self.target_net = QNetwork(state_dim, action_dim, hidden_dims).to(device)
            self.target_net.load_state_dict(self.online_net.state_dict())
            self.target_net.eval()

            # 다듬개와 로스
            self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
            if loss_fn == 'huber':
                self.loss_fn = nn.SmoothL1Loss()
            else:
                self.loss_fn = nn.MSELoss()

            # 되돌려 보기 버퍼
            self.buffer = ReplayBuffer(buffer_capacity, state_dim)

            # 셈틀
            self.total_steps = 0
            self.update_count = 0

            # 기록
            self.log: Dict[str, List] = {
                'losses': [],
                'q_values': [],
                'grad_norms': [],
                'epsilons': [],
            }

        @property
        def epsilon(self) -> float:
            frac = min(1.0, self.total_steps / self.eps_decay_steps)
            return self.eps_start + frac * (self.eps_end - self.eps_start)

        def select_action(self, state: np.ndarray, training: bool = True) -> int:
            if training and random.random() < self.epsilon:
                return random.randrange(self.action_dim)
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                return self.online_net(state_t).argmax(dim=1).item()

        def store_transition(self, state, action, reward, next_state, done):
            self.buffer.push(state, action, reward, next_state, done)
            self.total_steps += 1

        def train_step(self) -> Optional[float]:
            """익히기 한 스텝을 한다. 로스를 돌려주되 버퍼가 너무 작으면 None을 돌려준다."""
            if len(self.buffer) < self.min_buffer_size:
                return None

            states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            dones = dones.to(self.device)

            # 고른 액션의 지금 Q 값
            q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            # 타깃 Q 값
            with torch.no_grad():
                next_q = self.target_net(next_states).max(dim=1)[0]
                targets = rewards + (1 - dones) * self.gamma * next_q

            # 로스
            loss = self.loss_fn(q_values, targets)

            # 최적화
            self.optimizer.zero_grad()
            loss.backward()

            # 그래디언트 자르기
            grad_norm = nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip)

            self.optimizer.step()
            self.update_count += 1

            # 타깃 네트워크 고치기
            if self.update_count % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())

            # 기록
            loss_val = loss.item()
            self.log['losses'].append(loss_val)
            self.log['q_values'].append(q_values.mean().item())
            self.log['grad_norms'].append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
            self.log['epsilons'].append(self.epsilon)

            return loss_val

        def evaluate(self, env, n_episodes: int = 10) -> Dict[str, float]:
            """욕심쟁이 폴리시(ε=0)으로 따진다."""
            returns = []
            lengths = []
            for _ in range(n_episodes):
                state, _ = env.reset()
                total_r = 0.0
                steps = 0
                done = False
                while not done:
                    action = self.select_action(state, training=False)
                    state, r, term, trunc, _ = env.step(action)
                    total_r += r
                    steps += 1
                    done = term or trunc
                returns.append(total_r)
                lengths.append(steps)
            return {
                'mean_return': np.mean(returns),
                'std_return': np.std(returns),
                'min_return': np.min(returns),
                'max_return': np.max(returns),
                'mean_length': np.mean(lengths),
            }

        def save(self, path: str):
            torch.save({
                'online_net': self.online_net.state_dict(),
                'target_net': self.target_net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'total_steps': self.total_steps,
                'update_count': self.update_count,
            }, path)

        def load(self, path: str):
            ckpt = torch.load(path, map_location=self.device)
            self.online_net.load_state_dict(ckpt['online_net'])
            self.target_net.load_state_dict(ckpt['target_net'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.total_steps = ckpt['total_steps']
            self.update_count = ckpt['update_count']


    # ---------------------------------------------------------------------------
    # 학습 루프
    # ---------------------------------------------------------------------------

    def train_dqn(env_name: str = 'CartPole-v1',
                  n_episodes: int = 500,
                  eval_freq: int = 50,
                  eval_episodes: int = 10,
                  seed: int = 42,
                  **agent_kwargs) -> Tuple[DQNAgent, Dict]:
        """값 매기기를 포함한 온전한 DQN 익히기 회귀."""

        # 모든 씨앗 정하기
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        env = gym.make(env_name)
        eval_env = gym.make(env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        agent = DQNAgent(state_dim, action_dim, **agent_kwargs)

        # 학습 기록
        history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'eval_returns': [],
            'eval_episodes': [],
            'wall_time': [],
        }

        start_time = time.time()

        for ep in range(1, n_episodes + 1):
            state, _ = env.reset()
            total_reward = 0
            ep_length = 0
            done = False

            while not done:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.store_transition(state, action, reward, next_state, float(done))
                agent.train_step()
                state = next_state
                total_reward += reward
                ep_length += 1

            history['episode_rewards'].append(total_reward)
            history['episode_lengths'].append(ep_length)
            history['wall_time'].append(time.time() - start_time)

            # 때맞춰 값 매기기
            if ep % eval_freq == 0:
                eval_result = agent.evaluate(eval_env, eval_episodes)
                history['eval_returns'].append(eval_result['mean_return'])
                history['eval_episodes'].append(ep)

                # 흐르는 평균
                recent = history['episode_rewards'][-50:]
                print(f"Episode {ep:>4d} | "
                      f"Avg50: {np.mean(recent):>7.1f} | "
                      f"Eval: {eval_result['mean_return']:>7.1f} ± {eval_result['std_return']:.1f} | "
                      f"ε: {agent.epsilon:.3f} | "
                      f"Steps: {agent.total_steps:,} | "
                      f"Loss: {np.mean(agent.log['losses'][-100:]):.4f} | "
                      f"Q̄: {np.mean(agent.log['q_values'][-100:]):.2f}")

        env.close()
        eval_env.close()
        return agent, history


    # ---------------------------------------------------------------------------
    # 시연
    # ---------------------------------------------------------------------------

    def demo_dqn_implementation():
        """CartPole에서 온전한 DQN 익히기 보이기."""
        print("=" * 70)
        print("DQN Implementation Demo — CartPole-v1")
        print("=" * 70)

        agent, history = train_dqn(
            env_name='CartPole-v1',
            n_episodes=300,
            eval_freq=50,
            lr=1e-3,
            gamma=0.99,
            batch_size=64,
            buffer_capacity=50000,
            min_buffer_size=1000,
            target_update_freq=200,
            eps_start=1.0,
            eps_end=0.01,
            eps_decay_steps=5000,
            loss_fn='huber',
        )

        # --- 마지막 값 매기기 ---
        print("\n--- Final Evaluation (20 episodes, greedy) ---")
        eval_env = gym.make('CartPole-v1')
        final_eval = agent.evaluate(eval_env, n_episodes=20)
        eval_env.close()
        for k, v in final_eval.items():
            print(f"  {k}: {v:.2f}")

        # --- 익히기 셈밝힘 ---
        print("\n--- Training Statistics ---")
        rewards = history['episode_rewards']
        print(f"  Total episodes: {len(rewards)}")
        print(f"  Total steps: {agent.total_steps:,}")
        print(f"  Total updates: {agent.update_count:,}")
        print(f"  Best episode reward: {max(rewards):.0f}")
        print(f"  Last 50 avg: {np.mean(rewards[-50:]):.1f}")

        if agent.log['losses']:
            print(f"  Final loss (avg 100): {np.mean(agent.log['losses'][-100:]):.4f}")
        if agent.log['q_values']:
            print(f"  Final Q-value (avg 100): {np.mean(agent.log['q_values'][-100:]):.2f}")

        # --- 중간 갈무리 ---
        save_path = 'dqn_cartpole.pt'
        agent.save(save_path)
        print(f"\n  Checkpoint saved to {save_path}")

        # --- 불러오기 확인 ---
        agent2 = DQNAgent(4, 2)
        agent2.load(save_path)
        eval_env = gym.make('CartPole-v1')
        loaded_eval = agent2.evaluate(eval_env, n_episodes=5)
        eval_env.close()
        print(f"  Loaded checkpoint eval: {loaded_eval['mean_return']:.1f}")

        print("\nDQN implementation demo complete!")


    if __name__ == "__main__":
        demo_dqn_implementation()
    ```


## 2. 논의

이 구현은 `ReplayBuffer`(NumPy 원형 버퍼), `QNetwork`(사비에르 초기화를 한 MLP), `DQNAgent` 세 클래스와 학습 루프 `train_dqn` 으로 이루어진다. `DQNAgent.train_step` 한 번이 DQN 업데이트 한 번이다. 미니배치를 뽑고, 타깃 네트워크로 $r + \gamma(1-d)\max_{a'} Q_{\theta^-}(s', a')$ 을 계산하고, 후버 로스(`SmoothL1Loss`)로 온라인 네트워크를 한 스텝 갱신한 뒤, 업데이트 200번마다 타깃 네트워크를 덮어쓴다.

데모 함수는 CartPole-v1 에서 300 에피소드를 학습한다. 50 에피소드마다 그리디 폴리시로 10 에피소드를 평가해 최근 50 에피소드 평균 리워드, 평가 리턴, $\varepsilon$, 로스, 평균 Q 값을 한 줄씩 찍는다. 끝나면 20 에피소드로 최종 평가를 하고, 체크포인트를 저장했다가 새 에이전트에 불러와 다시 평가해 저장과 불러오기가 제대로 되는지 확인한다. 시드 42 로 한 번 돌린 결과이므로 학습 곡선의 세부 모양은 시드에 따라 달라진다.

로그에 남기는 평균 Q 값은 학습이 건강한지 보는 좋은 신호다. CartPole 에서 매 스텝 리워드가 1 이고 $\gamma = 0.99$ 이면 $Q$ 의 참값은 $1/(1-\gamma) = 100$ 을 넘을 수 없으므로, 로그의 Q̄ 가 이 값을 크게 넘어 계속 오르면 과대추정이 일어나고 있다는 뜻이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
데모의 $\varepsilon$ 스케줄(`eps_start=1.0`, `eps_end=0.01`, `eps_decay_steps=5000`)에서 전체 스텝 $k$ 일 때의 $\varepsilon$ 을 식으로 적고, 처음 5,000 스텝 동안 무작위 행동의 비율이 평균 얼마인지 구하라. 그 가운데 처음 몇 스텝 동안은 학습이 아예 일어나지 않는가?

</div>

??? success "풀이"
    `epsilon` 속성은 $\varepsilon(k) = 1.0 + \min(1, k/5000)\,(0.01 - 1.0) = 1 - 0.99\min(1, k/5000)$ 이다. 처음 5,000 스텝에서 $\varepsilon$ 은 1 에서 0.01 로 일차함수로 내려가므로 평균은 $(1 + 0.01)/2 = 0.505$, 곧 행동의 절반가량이 무작위다. 그 뒤로는 1% 만 무작위다. `train_step` 은 버퍼에 전이가 `min_buffer_size=1000` 개 쌓일 때까지 `None` 을 돌려주므로 처음 1,000 스텝은 데이터만 모으고 학습하지 않는다. 이 1,000 스텝의 평균 $\varepsilon$ 은 약 0.90 이라, 버퍼의 첫 내용은 거의 무작위 폴리시가 모은 것이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`loss_fn='huber'` 일 때 쓰는 `nn.SmoothL1Loss()`(기본값 `beta=1`)의 TD 오차 $\delta$ 에 대한 그래디언트를 적고, MSE 와 견주어 DQN 에서 후버 로스를 쓰는 까닭을 설명하라. 그래디언트 클리핑(`grad_clip=10.0`)과는 무엇이 다른가?

</div>

??? success "풀이"
    `SmoothL1Loss` 는 $|\delta| < 1$ 에서 $\tfrac12\delta^2$, 그 밖에서 $|\delta| - \tfrac12$ 이다. 그래디언트는 $|\delta| < 1$ 에서 $\delta$, 그 밖에서 $\operatorname{sign}(\delta)$ 이므로 크기가 1 을 넘지 않는다. MSE 의 그래디언트 $\delta$ 는 오차에 비례해 커지므로, 학습 초기나 타깃 네트워크를 덮어쓴 직후처럼 TD 오차가 큰 표본 하나가 업데이트를 좌우할 수 있다. 후버 로스는 **표본마다** 그래디언트 크기를 제한하고, `clip_grad_norm_` 은 배치 전체를 합친 **파라메터 그래디언트 벡터의 노름**을 제한한다. 앞의 것은 이상한 표본의 영향을 줄이고, 뒤의 것은 한 업데이트의 걸음 크기를 제한한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
`train_dqn` 은 `done = terminated or truncated` 를 버퍼에 저장하고 타깃에서 `(1 - dones)` 로 부트스트랩을 끊는다. CartPole-v1 은 500 스텝에서 에피소드를 잘라(truncate) 끝낸다. 이 처리가 500 스텝 근처 상태의 Q 값 타깃을 어떻게 왜곡하는지 설명하고, 바르게 고치는 방법을 적어라.

</div>

??? success "풀이"
    `terminated` 는 막대가 쓰러져 MDP 가 정말 끝난 것이고, `truncated` 는 시간 제한 때문에 관찰을 멈춘 것일 뿐 상태 자체는 계속 이어질 수 있다. 잘린 스텝을 종료로 저장하면 그 전이의 타깃이 $r = 1$ 로 줄어 $1 + \gamma\max Q(s', \cdot) \approx 100$ 이어야 할 값이 1 이 된다. 같은 상태가 다른 에피소드에서는 잘리지 않고 이어지므로, 네트워크는 똑같이 생긴 상태에 대해 서로 다른 타깃을 받는다. 시간 정보가 관찰에 없으니 이 차이를 설명할 수 없고, 잘 버티는 폴리시일수록 이런 전이가 많아져 Q 값이 흔들린다. 고치려면 버퍼에는 `terminated` 만 종료로 저장해 부트스트랩을 끊고, `truncated` 는 에피소드 루프를 끝내는 데만 쓴다.

## 정리하며

**다룬 것** — DQN 구현

이 구현은 경험 리플레이, 타깃 네트워크, 후버 로스, 그래디언트 클리핑, 주기적 평가, 체크포인트를 `ReplayBuffer`, `QNetwork`, `DQNAgent` 세 클래스에 모았다.

핵심 클래스는 `ReplayBuffer`, `QNetwork`, `DQNAgent`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
