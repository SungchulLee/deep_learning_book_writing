# 비동기 어드밴티지 Actor-Critic(A3C)

**비동기 어드밴티지 Actor-Critic**(A3C)은 워커 여럿이 저마다 환경을 하나씩 돌리며, 서로 기다리지 않고(비동기로) 공유 전역 모델을 갱신하는 Actor-Critic 이다. 워커는 전역 모델의 파라미터를 제 로컬 모델에 복사하고, n 스텝을 굴려 n-step 리턴과 어드밴티지로 그래디언트를 계산한 뒤, 그 그래디언트를 전역 모델에 넘겨 공유 옵티마이저로 한 스텝 간다. 잠금 없이 여러 프로세스가 같은 파라미터를 고치는 이 방식을 **Hogwild** 갱신이라 한다. 이 쪽은 공유 메모리를 쓰는 다중 프로세스 구현(`A3CTrainer`, `SharedAdam`)과, 같은 흐름을 프로세스 하나에서 차례로 흉내 내는 `SimulatedA3C` 를 담는다. 데모는 뒤엣것만 돌린다.

## 1. 코드

??? note "코드 (514줄)"

    ```python
    """
    34.2.3장: 발 안 맞춘 이점 액터-크리틱(A3C)
    ==========================================================
    온 세상 모형과 가장 좋게 하는 개를 위한 함께 쓰는 기억을
    쓰는, 파이토치 여러 프로세스 A3C 구현.
    """

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.multiprocessing as mp
    from torch.distributions import Categorical
    import numpy as np
    import gymnasium as gym
    from typing import List
    import os

    # ========================================================================
    # 메인
    # ========================================================================


    # ---------------------------------------------------------------------------
    # 액터-크리틱 네트워크
    # ---------------------------------------------------------------------------

    class A3CNetwork(nn.Module):
        """등뼈를 함께 쓰는 A3C용 액터-크리틱 네트워크."""

        def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128):
            super().__init__()
            self.features = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.actor = nn.Linear(hidden_dim, act_dim)
            self.critic = nn.Linear(hidden_dim, 1)

            # 무게 첫 값 매기기
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, np.sqrt(2))
                    nn.init.constant_(m.bias, 0)
            nn.init.orthogonal_(self.actor.weight, 0.01)
            nn.init.orthogonal_(self.critic.weight, 1.0)

        def forward(self, obs):
            features = self.features(obs)
            logits = self.actor(features)
            value = self.critic(features).squeeze(-1)
            return logits, value

        def get_action_and_value(self, obs, action=None):
            logits, value = self.forward(obs)
            dist = Categorical(logits=logits)
            if action is None:
                action = dist.sample()
            return action, dist.log_prob(action), dist.entropy(), value


    # ---------------------------------------------------------------------------
    # 함께 쓰는 Adam 가장 좋게 하는 개
    # ---------------------------------------------------------------------------

    class SharedAdam(torch.optim.Adam):
        """
        여러 프로세스를 위해 상태를 함께 쓰는 Adam.

        함께 쓰는 기억으로 가장 좋게 하는 개의 상태 텐서를 프로세스
        사이에서 함께 써 Hogwild 형태 고침을 이루게 한다.
        """

        def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
            super().__init__(params, lr=lr, betas=betas, eps=eps)
            # 상태에 첫 값을 매기고 기억을 함께 쓴다
            for group in self.param_groups:
                for p in group["params"]:
                    state = self.state[p]
                    state["step"] = torch.zeros(1)
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                    # 기억 함께 쓰기
                    state["step"].share_memory_()
                    state["exp_avg"].share_memory_()
                    state["exp_avg_sq"].share_memory_()


    # ---------------------------------------------------------------------------
    # A3C 일꾼
    # ---------------------------------------------------------------------------

    def a3c_worker(
        rank: int,
        global_model: A3CNetwork,
        optimizer: SharedAdam,
        env_id: str,
        global_episode_counter: mp.Value,
        global_rewards: mp.Manager,
        max_episodes: int,
        gamma: float = 0.99,
        n_steps: int = 20,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 40.0,
    ):
        """
        A3C 일꾼 프로세스.

        일꾼마다:
        1. 제 모형을 온 세상 모형에 맞춘다
        2. n_steps만큼 겪음을 모은다
        3. 제자리에서 그래디언트를 셈한다
        4. 그 그래디언트를 온 세상 모형에 매긴다
        """
        torch.manual_seed(rank + 42)

        env = gym.make(env_id)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n

        # 제 모형(함께 쓰지 않는다)
        local_model = A3CNetwork(obs_dim, act_dim)

        obs, _ = env.reset(seed=rank)
        episode_reward = 0.0

        while True:
            # 익힘이 끝났는지 살핀다
            with global_episode_counter.get_lock():
                if global_episode_counter.value >= max_episodes:
                    break

            # 제 모형을 온 세상 모형에 맞춘다
            local_model.load_state_dict(global_model.state_dict())

            # n스텝 굴림을 모은다
            states, actions, rewards, dones = [], [], [], []
            log_probs, values, entropies = [], [], []

            for _ in range(n_steps):
                obs_t = torch.FloatTensor(obs).unsqueeze(0)

                with torch.no_grad():
                    action, log_prob, entropy, value = local_model.get_action_and_value(obs_t)

                next_obs, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated

                states.append(obs)
                actions.append(action.item())
                rewards.append(reward)
                dones.append(done)

                episode_reward += reward
                obs = next_obs

                if done:
                    with global_episode_counter.get_lock():
                        global_episode_counter.value += 1
                        ep_num = global_episode_counter.value

                    global_rewards.append(episode_reward)

                    if ep_num % 100 == 0:
                        recent = list(global_rewards)[-100:]
                        print(
                            f"Worker {rank} | Episode {ep_num} | "
                            f"Reward: {episode_reward:.1f} | "
                            f"Avg(100): {np.mean(recent):.1f}"
                        )

                    episode_reward = 0.0
                    obs, _ = env.reset()

                    if ep_num >= max_episodes:
                        break

            # 돌아옴과 이점을 셈한다
            states_t = torch.FloatTensor(np.array(states))
            actions_t = torch.LongTensor(actions)

            # 마지막 상태의 부트스트랩 값
            with torch.no_grad():
                if dones[-1]:
                    R = 0.0
                else:
                    _, last_value = local_model(torch.FloatTensor(obs).unsqueeze(0))
                    R = last_value.item()

            returns = []
            for t in reversed(range(len(rewards))):
                R = rewards[t] + gamma * R * (1 - dones[t])
                returns.insert(0, R)
            returns = torch.FloatTensor(returns)

            # 그래디언트를 켜고 앞으로 지나가기
            _, log_probs_t, entropies_t, values_t = local_model.get_action_and_value(
                states_t, actions_t
            )

            # 이점
            advantages = returns - values_t.detach()

            # 로스
            policy_loss = -(log_probs_t * advantages).mean()
            value_loss = F.mse_loss(values_t, returns)
            entropy_loss = -entropies_t.mean()

            total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

            # 제 모형에서 그래디언트를 셈한다
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(local_model.parameters(), max_grad_norm)

            # 그래디언트를 온 세상 모형으로 옮긴다
            for local_param, global_param in zip(
                local_model.parameters(), global_model.parameters()
            ):
                if global_param.grad is None:
                    global_param.grad = local_param.grad.clone()
                else:
                    global_param.grad.copy_(local_param.grad)

            # 온 세상 모형에 그래디언트를 매긴다(Hogwild 형태)
            optimizer.step()

        env.close()


    # ---------------------------------------------------------------------------
    # A3C 익힘 이끄개
    # ---------------------------------------------------------------------------

    class A3CTrainer:
        """
        A3C 익힘을 이끄는 것.

        일꾼 프로세스를 돋아나게 하고 온 세상 모형과 가장 좋게 하는
        개를 다룬다.
        """

        def __init__(
            self,
            env_id: str = "CartPole-v1",
            n_workers: int = 4,
            n_steps: int = 20,
            lr: float = 1e-3,
            gamma: float = 0.99,
            hidden_dim: int = 128,
            entropy_coef: float = 0.01,
            value_coef: float = 0.5,
        ):
            self.env_id = env_id
            self.n_workers = n_workers
            self.n_steps = n_steps
            self.gamma = gamma
            self.entropy_coef = entropy_coef
            self.value_coef = value_coef

            # 차원을 알아낸다
            env = gym.make(env_id)
            obs_dim = env.observation_space.shape[0]
            act_dim = env.action_space.n
            env.close()

            # 온 세상 모형(함께 쓰는 기억)
            self.global_model = A3CNetwork(obs_dim, act_dim, hidden_dim)
            self.global_model.share_memory()

            # 함께 쓰는 가장 좋게 하는 개
            self.optimizer = SharedAdam(self.global_model.parameters(), lr=lr)

        def train(self, max_episodes: int = 2000) -> List[float]:
            """일꾼 프로세스 여럿으로 익힌다."""
            mp.set_start_method("spawn", force=True)

            # 함께 쓰는 세개
            global_episode_counter = mp.Value("i", 0)
            manager = mp.Manager()
            global_rewards = manager.list()

            # 일꾼을 돋아나게 한다
            processes = []
            for rank in range(self.n_workers):
                p = mp.Process(
                    target=a3c_worker,
                    args=(
                        rank,
                        self.global_model,
                        self.optimizer,
                        self.env_id,
                        global_episode_counter,
                        global_rewards,
                        max_episodes,
                        self.gamma,
                        self.n_steps,
                        self.entropy_coef,
                        self.value_coef,
                    ),
                )
                p.start()
                processes.append(p)

            # 온 일꾼을 기다린다
            for p in processes:
                p.join()

            return list(global_rewards)

        def evaluate(self, n_episodes: int = 10) -> float:
            """온 세상 모형을 따진다."""
            env = gym.make(self.env_id)
            rewards = []

            for _ in range(n_episodes):
                obs, _ = env.reset()
                total_reward = 0.0
                done = False

                while not done:
                    with torch.no_grad():
                        obs_t = torch.FloatTensor(obs).unsqueeze(0)
                        logits, _ = self.global_model(obs_t)
                        action = logits.argmax(dim=-1).item()

                    obs, reward, terminated, truncated, _ = env.step(action)
                    total_reward += reward
                    done = terminated or truncated

                rewards.append(total_reward)

            env.close()
            return np.mean(rewards)


    # ---------------------------------------------------------------------------
    # 프로세스 하나로 쉽게 흉내 낸 A3C
    # ---------------------------------------------------------------------------

    class SimulatedA3C:
        """
        프로세스 하나에서 쉽게 흉내 낸 A3C.

        둘레를 여럿 지니고 차례대로 고쳐 A3C의 거동을 흉내 낸다.
        여러 프로세스의 얽힘 없이 보여 주기에 쓸모 있다.
        """

        def __init__(
            self,
            env_id: str = "CartPole-v1",
            n_workers: int = 4,
            n_steps: int = 20,
            lr: float = 1e-3,
            gamma: float = 0.99,
            hidden_dim: int = 128,
            entropy_coef: float = 0.01,
            value_coef: float = 0.5,
        ):
            self.n_workers = n_workers
            self.n_steps = n_steps
            self.gamma = gamma
            self.entropy_coef = entropy_coef
            self.value_coef = value_coef

            # 둘레를 만든다
            self.envs = [gym.make(env_id) for _ in range(n_workers)]
            obs_dim = self.envs[0].observation_space.shape[0]
            act_dim = self.envs[0].action_space.n

            # 온 세상 모형
            self.global_model = A3CNetwork(obs_dim, act_dim, hidden_dim)
            self.optimizer = torch.optim.Adam(self.global_model.parameters(), lr=lr)

            # 제 모형들
            self.local_models = [
                A3CNetwork(obs_dim, act_dim, hidden_dim) for _ in range(n_workers)
            ]

        def train(self, max_episodes: int = 1000, print_interval: int = 100):
            all_rewards = []
            recent_rewards = deque(maxlen=100)
            episode_count = 0

            # 봄에 첫 값을 매긴다
            obs_list = [env.reset(seed=i)[0] for i, env in enumerate(self.envs)]
            episode_rewards = [0.0] * self.n_workers

            while episode_count < max_episodes:
                # 일꾼마다 모으고 고친다
                for w in range(self.n_workers):
                    # 제 모형을 온 세상 모형에 맞춘다
                    self.local_models[w].load_state_dict(self.global_model.state_dict())

                    # 굴림을 모은다
                    states, actions, rewards, dones = [], [], [], []

                    for _ in range(self.n_steps):
                        obs_t = torch.FloatTensor(obs_list[w]).unsqueeze(0)
                        with torch.no_grad():
                            action, _, _, _ = self.local_models[w].get_action_and_value(obs_t)

                        next_obs, reward, terminated, truncated, _ = self.envs[w].step(action.item())
                        done = terminated or truncated

                        states.append(obs_list[w])
                        actions.append(action.item())
                        rewards.append(reward)
                        dones.append(done)
                        episode_rewards[w] += reward

                        obs_list[w] = next_obs

                        if done:
                            all_rewards.append(episode_rewards[w])
                            recent_rewards.append(episode_rewards[w])
                            episode_count += 1
                            episode_rewards[w] = 0.0
                            obs_list[w], _ = self.envs[w].reset()

                            if episode_count % print_interval == 0:
                                print(
                                    f"Episode {episode_count} | "
                                    f"Avg(100): {np.mean(recent_rewards):.1f}"
                                )

                            if episode_count >= max_episodes:
                                break

                    if episode_count >= max_episodes:
                        break

                    # 돌아옴을 셈한다
                    states_t = torch.FloatTensor(np.array(states))
                    actions_t = torch.LongTensor(actions)

                    with torch.no_grad():
                        if dones[-1]:
                            R = 0.0
                        else:
                            _, last_v = self.local_models[w](
                                torch.FloatTensor(obs_list[w]).unsqueeze(0)
                            )
                            R = last_v.item()

                    returns_list = []
                    for t in reversed(range(len(rewards))):
                        R = rewards[t] + self.gamma * R * (1 - dones[t])
                        returns_list.insert(0, R)
                    returns_t = torch.FloatTensor(returns_list)

                    # 앞으로 지나가기 + 로스
                    _, lp, ent, val = self.local_models[w].get_action_and_value(states_t, actions_t)
                    adv = returns_t - val.detach()

                    loss = (
                        -(lp * adv).mean()
                        + self.value_coef * F.mse_loss(val, returns_t)
                        - self.entropy_coef * ent.mean()
                    )

                    # 온 세상 모형에 그래디언트를 매긴다
                    self.optimizer.zero_grad()
                    loss.backward()

                    for local_p, global_p in zip(
                        self.local_models[w].parameters(),
                        self.global_model.parameters()
                    ):
                        if global_p.grad is None:
                            global_p.grad = local_p.grad.clone()
                        else:
                            global_p.grad.copy_(local_p.grad)

                    nn.utils.clip_grad_norm_(self.global_model.parameters(), 40.0)
                    self.optimizer.step()

            for env in self.envs:
                env.close()

            return all_rewards


    from collections import deque


    def demo_simulated_a3c():
        """프로세스 하나로 흉내 낸 A3C를 보여 준다."""
        print("=" * 60)
        print("Simulated A3C on CartPole-v1")
        print("=" * 60)

        agent = SimulatedA3C(
            env_id="CartPole-v1",
            n_workers=4,
            n_steps=20,
            lr=1e-3,
            gamma=0.99,
            hidden_dim=128,
            entropy_coef=0.01,
            value_coef=0.5,
        )

        rewards = agent.train(max_episodes=1000, print_interval=200)

        if len(rewards) >= 100:
            print(f"\nFinal avg reward (last 100): {np.mean(rewards[-100:]):.1f}")


    if __name__ == "__main__":
        demo_simulated_a3c()
    ```


## 2. 논의

다중 프로세스 쪽에서 `A3CTrainer` 는 전역 모델을 `share_memory()` 로 공유 메모리에 올리고, `SharedAdam` 은 Adam 의 상태(스텝 수, 1차·2차 모멘트)까지 공유 메모리에 두어 모든 워커가 같은 옵티마이저 상태를 쓰게 한다. 각 `a3c_worker` 는 20 스텝 롤아웃마다 전역 파라미터를 로컬 모델에 복사하고, 롤아웃 끝 상태의 가치로 부트스트랩한 n-step 리턴에서 어드밴티지 `returns - values.detach()` 를 만든다. 로스는 폴리시 로스, 값 로스(계수 0.5), 엔트로피 보너스(계수 0.01)의 합이고, 그래디언트 노름을 40 으로 자른 뒤 로컬 그래디언트를 전역 파라미터의 `.grad` 에 복사해 `optimizer.step()` 한다. 워커가 롤아웃을 도는 동안 다른 워커가 전역 모델을 이미 고쳤을 수 있으므로, 그래디언트는 조금 낡은 파라미터에서 계산된 채 새 파라미터에 적용된다.

데모가 돌리는 `SimulatedA3C` 는 이 비동기성을 흉내 내지 않는다. 워커들이 차례로 돌면서 롤아웃 직전에 전역 모델을 복사하므로 그래디언트가 낡는 일이 없고, 사실상 환경 하나짜리 롤아웃으로 한 번씩 갱신하는 동기식 Actor-Critic 이다(연습문제 2).

두 구현에 공통된 결함이 하나 있다. `optimizer.zero_grad()` 는 옵티마이저가 가진 **전역** 파라미터의 그래디언트만 지우고, 로컬 모델의 그래디언트는 아무도 지우지 않는다. 그래서 `loss.backward()` 가 부를 때마다 로컬 그래디언트에 더해 쌓이고, 전역 모델에는 그 워커의 지난 그래디언트 전부를 합한 값이 넘어간다(연습문제 1). 데모의 학습 곡선은 이 결함을 안은 채의 결과다. 또 `SimulatedA3C` 에는 `torch.manual_seed` 가 없어 네트워크 초기화와 액션 샘플링이 돌릴 때마다 다르다. 이 쪽에는 출력 블록이 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
`SimulatedA3C.train` 에서 워커 $w$ 가 $k$ 번째로 갱신할 때 전역 모델에 넘어가는 그래디언트를, 그 워커가 매번 계산한 로스의 그래디언트 $g_w^{(1)}, \dots, g_w^{(k)}$ 로 나타내라. `load_state_dict` 가 그래디언트를 지워 주지 않는다는 점에 주의하라. 무엇이 문제이며 어떻게 고치는가?

</div>

??? success "풀이"
    `self.optimizer` 는 전역 모델의 파라미터만 가지므로 `self.optimizer.zero_grad()` 는 전역 그래디언트만 비운다. `load_state_dict` 는 로컬 모델의 웨이트만 덮어쓰고 `.grad` 는 그대로 두므로, `loss.backward()` 는 로컬 그래디언트에 계속 더한다. 따라서 $k$ 번째 갱신에 넘어가는 것은 $g_w^{(1)} + \dots + g_w^{(k)}$ 이고, 학습이 진행될수록 갱신의 크기가 커지며 오래전 파라미터에서 계산한 그래디언트가 영영 남는다. `SimulatedA3C` 에서는 전역 노름을 40 으로 자르므로 크기는 묶이지만 방향은 지난 그래디언트의 합 쪽으로 끌린다(`a3c_worker` 는 자르는 것도 쌓인 로컬 그래디언트에 하므로 사정이 같다). 고치려면 `backward()` 앞에서 `self.local_models[w].zero_grad()`(워커 함수에서는 `local_model.zero_grad()`)를 부른다. 고친 뒤 학습 곡선이 달라지는지는 시드를 여럿 돌려 확인한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`SimulatedA3C` 는 진짜 A3C 의 무엇을 흉내 내고 무엇을 흉내 내지 못하는가? 진짜 A3C 에서 워커가 계산한 그래디언트가 「낡는」 까닭을 설명하고, 낡은 그래디언트가 학습에 주는 영향과 A3C 가 그래도 잘 작동하는 까닭을 논하라. 같은 데이터를 쓴다면 A2C 와 무엇이 다른가?

</div>

??? success "풀이"
    `SimulatedA3C` 는 워커마다 환경과 로컬 모델을 따로 두고, 롤아웃 직전에 전역 모델을 복사하며, 한 워커의 롤아웃 하나로 한 번 갱신한다는 흐름을 흉내 낸다. 그러나 워커가 차례로 돌기 때문에 롤아웃을 모으는 동안 전역 모델이 바뀌는 일이 없다. 진짜 A3C 에서는 워커가 $\theta_{\text{old}}$ 를 복사해 20 스텝을 굴리는 사이에 다른 워커들이 전역 모델을 여러 번 고치므로, $\theta_{\text{old}}$ 에서 계산한 그래디언트가 $\theta_{\text{new}}$ 에 적용된다. 낡음이 크면 그래디언트가 더는 지금 파라미터에서의 오르막이 아니어서 학습이 흔들린다. 그래도 워커 수가 적고 러닝레이트가 작으면 파라미터가 그사이 조금밖에 안 움직여 낡음의 영향이 작고, 워커들이 서로 다른 상태를 탐색해 데이터의 상관이 줄어드는 이점이 이를 메운다. A2C 는 모든 워커의 롤아웃을 기다렸다가 한 배치로 한 번 갱신하므로 낡은 그래디언트가 없고, GPU 에서 배치로 계산하기 좋다. 그래서 오늘날에는 A2C 가 같은 성능을 더 단순하게 내는 경우가 많다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
연습문제 1의 결함을 고친 뒤 `A3CTrainer` 로 진짜 다중 프로세스 학습을 돌려라. `spawn` 시작 방식에서 메인 코드를 `if __name__ == "__main__":` 안에 두어야 하는 까닭을 설명하고, 워커 수를 1, 2, 4, 8 로 바꾸며 (가) 같은 에피소드 수에서의 평가 리워드와 (나) 걸린 벽시계 시간을 시드 여럿으로 견주어라. 또 `SharedAdam` 의 상태를 여러 프로세스가 잠금 없이 고칠 때 생길 수 있는 일을 논하라.

</div>

??? success "풀이"
    `spawn` 은 자식 프로세스에서 모듈을 처음부터 다시 임포트하므로, 학습을 시작하는 코드가 가드 밖에 있으면 자식마다 다시 프로세스를 띄우려 들어 실패한다. 워커를 늘리면 같은 시간에 모으는 전이가 늘어 벽시계 시간은 대개 줄지만, 에피소드 수로 재면 낡은 그래디언트 때문에 효율이 같거나 조금 나빠질 수 있다. CPU 코어 수를 넘기면 시간도 더는 줄지 않는다. `SharedAdam` 의 모멘트와 스텝 수는 잠금 없이 고쳐지므로 두 워커가 동시에 `step()` 하면 한쪽의 갱신이 덮어써지거나 스텝 수가 실제보다 적게 셈해질 수 있다. 이것이 Adam 의 편향 보정을 조금 어긋나게 할 뿐 학습을 깨뜨리지는 않는다는 것이 Hogwild 류 방법의 경험적 전제다. `torch.manual_seed(rank + 42)` 로 워커마다 시드를 박지만, 프로세스가 갱신하는 순서는 운영체제가 정하므로 같은 시드로도 결과가 달라진다. 그래서 시드를 여럿 돌려 흩어짐을 함께 보고한다.

## 정리하며

**다룬 것** — 비동기 어드밴티지 Actor-Critic(A3C)

A3C 는 워커들이 로컬 모델로 계산한 그래디언트를 공유 전역 모델에 잠금 없이 적용해, 기다림 없이 병렬로 학습한다. 그 대가는 낡은 그래디언트다. 데모의 `SimulatedA3C` 는 그 비동기성 없이 흐름만 흉내 내며, 두 구현 모두 로컬 그래디언트를 지우지 않아 갱신이 쌓이는 결함을 안고 있다.

핵심 클래스는 `A3CNetwork`, `SharedAdam`, `A3CTrainer`, `SimulatedA3C`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
