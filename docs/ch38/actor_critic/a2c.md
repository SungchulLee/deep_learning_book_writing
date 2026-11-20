# 어드밴티지 Actor-Critic(A2C)

**어드밴티지 Actor-Critic**(A2C)은 환경 여러 개를 나란히, 발맞추어 한 스텝씩 돌려 모은 짧은 롤아웃으로 액터와 크리틱을 함께 갱신하는 방법이다. 롤아웃은 지금 폴리시로 환경을 몇 스텝 굴려 모은 전이의 묶음이다. 액터는 **어드밴티지** $\hat A_t = \hat R_t - V(s_t)$ — 그 액션의 결과가 크리틱의 예상보다 얼마나 나았는가 — 를 로그 확률에 곱해 배우고, 크리틱은 n-step 리턴 $\hat R_t$ 에 회귀한다. 이 쪽은 공유 네트워크 `A2CNetwork`, 환경 여러 개를 묶는 `VecEnv`, 그리고 n-step 리턴과 GAE 를 골라 쓸 수 있는 `A2C` 를 구현해 CartPole 에서 돌린다.

## 1. 코드

??? note "코드 (437줄)"

    ```python
    """
    34.2.2장: 이점 액터-크리틱(A2C)
    =============================================
    벡터 둘레, n스텝 돌아옴, 액터-크리틱을 함께 익히는
    온전한 A2C 구현.
    """

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
    import numpy as np
    import gymnasium as gym
    from typing import List
    from collections import deque

    # ========================================================================
    # 메인
    # ========================================================================


    # ---------------------------------------------------------------------------
    # 액터-크리틱 네트워크
    # ---------------------------------------------------------------------------

    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer


    class A2CNetwork(nn.Module):
        """A2C를 위한, 함께 쓰는 액터-크리틱 네트워크."""

        def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64):
            super().__init__()
            self.features = nn.Sequential(
                layer_init(nn.Linear(obs_dim, hidden_dim)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden_dim, hidden_dim)),
                nn.Tanh(),
            )
            self.actor = layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)
            self.critic = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

        def forward(self, obs):
            features = self.features(obs)
            return self.actor(features), self.critic(features).squeeze(-1)

        def get_action_and_value(self, obs, action=None):
            logits, value = self.forward(obs)
            dist = Categorical(logits=logits)
            if action is None:
                action = dist.sample()
            return action, dist.log_prob(action), dist.entropy(), value

        def get_value(self, obs):
            return self.forward(obs)[1]


    # ---------------------------------------------------------------------------
    # 벡터 둘레 감싸개
    # ---------------------------------------------------------------------------

    class VecEnv:
        """
        쉬운 발맞춘 벡터 둘레.

        둘레 N개를 나란히 돌리며 발맞추어 스텝을 옮긴다.
        """

        def __init__(self, env_id: str, n_envs: int, seed: int = 0):
            self.envs = [gym.make(env_id) for _ in range(n_envs)]
            self.n_envs = n_envs
            for i, env in enumerate(self.envs):
                env.reset(seed=seed + i)

        def reset(self):
            obs_list = []
            for env in self.envs:
                obs, _ = env.reset()
                obs_list.append(obs)
            return np.array(obs_list)

        def step(self, actions):
            obs_list, rewards, dones, infos = [], [], [], []
            for env, action in zip(self.envs, actions):
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # 끝나면 저절로 되돌린다
                if done:
                    info["terminal_obs"] = obs
                    info["episode_reward"] = info.get("episode", {}).get("r", None)
                    obs, _ = env.reset()

                obs_list.append(obs)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)

            return np.array(obs_list), np.array(rewards), np.array(dones), infos

        @property
        def observation_space(self):
            return self.envs[0].observation_space

        @property
        def action_space(self):
            return self.envs[0].action_space

        def close(self):
            for env in self.envs:
                env.close()


    # ---------------------------------------------------------------------------
    # A2C 에이전트
    # ---------------------------------------------------------------------------

    class A2C:
        """
        발맞춘 벡터 둘레를 쓰는 이점 액터-크리틱(A2C).

        파라메터
        ----------
        env_id : str
            Gymnasium 둘레 아이디.
        n_envs : int
            나란한 둘레의 개수.
        n_steps : int
            둘레마다 굴림 길이.
        lr : float
            러닝레이트.
        gamma : float
            깎기 인자.
        value_coef : float
            값 로스 계수.
        entropy_coef : float
            엔트로피 덤 계수.
        max_grad_norm : float
            자르기를 위한 그래디언트 노름의 최댓값.
        use_gae : bool
            GAE를 쓸지(True) n스텝 돌아옴을 쓸지(False).
        gae_lambda : float
            GAE 람다 파라메터.
        """

        def __init__(
            self,
            env_id: str = "CartPole-v1",
            n_envs: int = 8,
            n_steps: int = 5,
            lr: float = 7e-4,
            gamma: float = 0.99,
            hidden_dim: int = 64,
            value_coef: float = 0.5,
            entropy_coef: float = 0.01,
            max_grad_norm: float = 0.5,
            use_gae: bool = False,
            gae_lambda: float = 0.95,
            seed: int = 0,
        ):
            self.gamma = gamma
            self.n_steps = n_steps
            self.n_envs = n_envs
            self.value_coef = value_coef
            self.entropy_coef = entropy_coef
            self.max_grad_norm = max_grad_norm
            self.use_gae = use_gae
            self.gae_lambda = gae_lambda

            # 둘레 갖추기
            self.envs = VecEnv(env_id, n_envs, seed=seed)
            obs_dim = self.envs.observation_space.shape[0]
            act_dim = self.envs.action_space.n

            # 네트워크 갖추기
            self.network = A2CNetwork(obs_dim, act_dim, hidden_dim)
            self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

        def collect_rollout(self, obs):
            """
            온 둘레에서 n_steps만큼의 겪음을 모은다.

            돌려주는 값
            -------
            obs : ndarray (이어 가기 위한 다음 봄)
            rollout : 텐서 사전
            episode_rewards : 마친 에피소드 리워드의 목록
            """
            # 곳간
            mb_obs = np.zeros((self.n_steps, self.n_envs) + self.envs.observation_space.shape)
            mb_actions = np.zeros((self.n_steps, self.n_envs), dtype=np.int64)
            mb_rewards = np.zeros((self.n_steps, self.n_envs))
            mb_dones = np.zeros((self.n_steps, self.n_envs))
            mb_values = np.zeros((self.n_steps, self.n_envs))
            mb_log_probs = np.zeros((self.n_steps, self.n_envs))

            episode_rewards = []

            for step in range(self.n_steps):
                mb_obs[step] = obs

                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs)
                    action, log_prob, _, value = self.network.get_action_and_value(obs_t)

                mb_actions[step] = action.numpy()
                mb_values[step] = value.numpy()
                mb_log_probs[step] = log_prob.numpy()

                obs, rewards, dones, infos = self.envs.step(action.numpy())
                mb_rewards[step] = rewards
                mb_dones[step] = dones

                # 에피소드가 마친 것을 좇는다
                for i, done in enumerate(dones):
                    if done:
                        # 갈무리한 리워드로 에피소드 리워드를 셈한다
                        ep_reward = sum(mb_rewards[s, i] for s in range(step + 1))
                        episode_rewards.append(ep_reward)

            rollout = {
                "obs": torch.FloatTensor(mb_obs.reshape(-1, *self.envs.observation_space.shape)),
                "actions": torch.LongTensor(mb_actions.reshape(-1)),
                "rewards": mb_rewards,
                "dones": mb_dones,
                "values": mb_values,
                "log_probs": mb_log_probs,
            }

            return obs, rollout, episode_rewards

        def compute_returns_and_advantages(self, rollout, last_obs):
            """돌아옴 타깃과 이점을 셈한다."""
            rewards = rollout["rewards"]
            dones = rollout["dones"]
            values = rollout["values"]

            with torch.no_grad():
                last_value = self.network.get_value(
                    torch.FloatTensor(last_obs)
                ).numpy()

            if self.use_gae:
                # GAE 셈하기
                advantages = np.zeros_like(rewards)
                last_gae = 0
                for t in reversed(range(self.n_steps)):
                    if t == self.n_steps - 1:
                        next_non_terminal = 1.0 - dones[t]
                        next_values = last_value
                    else:
                        next_non_terminal = 1.0 - dones[t]
                        next_values = values[t + 1]

                    delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
                    advantages[t] = last_gae = \
                        delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae

                returns = advantages + values
            else:
                # n스텝 돌아옴(여느 A2C)
                returns = np.zeros_like(rewards)
                R = last_value
                for t in reversed(range(self.n_steps)):
                    R = rewards[t] + self.gamma * R * (1.0 - dones[t])
                    returns[t] = R
                advantages = returns - values

            returns = torch.FloatTensor(returns.reshape(-1))
            advantages = torch.FloatTensor(advantages.reshape(-1))

            # 이점을 고르게 한다
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            return returns, advantages

        def update(self, rollout, returns, advantages):
            """A2C 고침을 벌인다."""
            obs = rollout["obs"]
            actions = rollout["actions"]

            # 앞으로 지나가기
            _, new_log_probs, entropy, new_values = \
                self.network.get_action_and_value(obs, actions)

            # 폴리시 로스
            policy_loss = -(new_log_probs * advantages).mean()

            # 값 로스
            value_loss = nn.functional.mse_loss(new_values, returns)

            # 엔트로피 로스
            entropy_loss = -entropy.mean()

            # 온 로스
            total_loss = (
                policy_loss
                + self.value_coef * value_loss
                + self.entropy_coef * entropy_loss
            )

            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()

            return {
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "entropy": entropy.mean().item(),
                "total_loss": total_loss.item(),
            }

        def train(self, total_steps: int = 200000, print_interval: int = 10000) -> List[float]:
            """A2C 에이전트를 익힌다."""
            obs = self.envs.reset()
            all_rewards = []
            recent_rewards = deque(maxlen=100)
            steps = 0
            updates = 0

            while steps < total_steps:
                obs, rollout, episode_rewards = self.collect_rollout(obs)
                returns, advantages = self.compute_returns_and_advantages(rollout, obs)
                metrics = self.update(rollout, returns, advantages)

                steps += self.n_steps * self.n_envs
                updates += 1

                for r in episode_rewards:
                    all_rewards.append(r)
                    recent_rewards.append(r)

                if steps % print_interval < self.n_steps * self.n_envs and len(recent_rewards) > 0:
                    print(
                        f"Step {steps:>8d} | "
                        f"Avg(100): {np.mean(recent_rewards):>7.1f} | "
                        f"Policy: {metrics['policy_loss']:>7.4f} | "
                        f"Value: {metrics['value_loss']:>7.4f} | "
                        f"H: {metrics['entropy']:>5.3f}"
                    )

            self.envs.close()
            return all_rewards

        def evaluate(self, env_id: str, n_episodes: int = 10) -> float:
            """익힌 폴리시를 따진다."""
            env = gym.make(env_id)
            rewards = []

            for _ in range(n_episodes):
                obs, _ = env.reset()
                total_reward = 0.0
                done = False

                while not done:
                    with torch.no_grad():
                        obs_t = torch.FloatTensor(obs).unsqueeze(0)
                        logits, _ = self.network(obs_t)
                        action = logits.argmax(dim=-1).item()

                    obs, reward, terminated, truncated, _ = env.step(action)
                    total_reward += reward
                    done = terminated or truncated

                rewards.append(total_reward)

            env.close()
            return np.mean(rewards)


    # ---------------------------------------------------------------------------
    # 보여 주기
    # ---------------------------------------------------------------------------

    def demo_a2c():
        """CartPole에서 A2C를 익힌다."""
        print("=" * 60)
        print("A2C on CartPole-v1")
        print("=" * 60)

        agent = A2C(
            env_id="CartPole-v1",
            n_envs=8,
            n_steps=5,
            lr=7e-4,
            gamma=0.99,
            hidden_dim=64,
            value_coef=0.5,
            entropy_coef=0.01,
            use_gae=False,
            seed=42,
        )

        rewards = agent.train(total_steps=200000, print_interval=20000)

        # 따지기
        eval_reward = agent.evaluate("CartPole-v1", n_episodes=20)
        print(f"\nEvaluation reward (20 episodes): {eval_reward:.1f}")

        return rewards


    def demo_a2c_with_gae():
        """CartPole에서 GAE를 쓰는 A2C를 익힌다."""
        print("\n" + "=" * 60)
        print("A2C with GAE on CartPole-v1")
        print("=" * 60)

        agent = A2C(
            env_id="CartPole-v1",
            n_envs=8,
            n_steps=128,
            lr=2.5e-4,
            gamma=0.99,
            hidden_dim=64,
            value_coef=0.5,
            entropy_coef=0.01,
            use_gae=True,
            gae_lambda=0.95,
            seed=42,
        )

        rewards = agent.train(total_steps=200000, print_interval=20000)

        eval_reward = agent.evaluate("CartPole-v1", n_episodes=20)
        print(f"\nEvaluation reward (20 episodes): {eval_reward:.1f}")

        return rewards


    if __name__ == "__main__":
        demo_a2c()
        demo_a2c_with_gae()
    ```


## 2. 논의

한 번의 갱신은 세 단계다. `collect_rollout` 이 환경 `n_envs` 개에서 `n_steps` 스텝씩 전이를 모으고, `compute_returns_and_advantages` 가 롤아웃 끝의 상태 가치 $V(s_n)$ 에서 거꾸로 올라오며 $\hat R_t = r_t + \gamma \hat R_{t+1}(1-d_t)$ 로 리턴을 만든다. 그래서 롤아웃의 앞쪽 스텝은 $n$ 스텝짜리, 마지막 스텝은 1 스텝짜리 리턴을 받는다. `use_gae=True` 면 같은 자리에서 GAE 를 계산한다. 마지막으로 `update` 가 폴리시 로스, 값 로스(계수 0.5), 엔트로피 보너스(계수 0.01)를 더한 하나의 로스로 공유 네트워크를 한 번 갱신하고 그래디언트 노름을 0.5 로 자른다. 첫 데모는 환경 8개 × 5 스텝 = 40 전이마다, GAE 데모는 8 × 128 = 1,024 전이마다 갱신하므로 같은 20만 스텝에서 갱신 횟수가 약 5,000 번과 약 200 번으로 크게 다르다.

출력을 읽을 때 조심할 점이 셋 있다. 첫째, `collect_rollout` 이 기록하는 에피소드 리워드는 **그 롤아웃 안의** 리워드만 더한 값이다. `n_steps=5` 이면 CartPole 에피소드가 몇백 스텝을 버텨도 기록은 5 를 넘지 못하므로, 학습 중 찍히는 `Avg(100)` 은 에피소드 리턴이 아니다(연습문제 1). 학습이 됐는지는 끝의 `evaluate` 가 찍는 평가 리워드로 판단한다. `evaluate` 는 확률을 뽑지 않고 로짓의 argmax 로 액션을 고른다. 둘째, `VecEnv` 는 `terminated` 와 `truncated` 를 구별하지 않고 둘 다 `done` 으로 넘기므로, 500 스텝 시간 제한에 걸려 잘린 에피소드에서도 다음 가치를 0 으로 둔다(연습문제 3). 셋째, 환경은 `seed` 로 고정되지만 `torch.manual_seed` 가 없어 네트워크 초기화와 액션 샘플링은 돌릴 때마다 다르다. 이 쪽에는 출력 블록이 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
`collect_rollout` 의 `ep_reward = sum(mb_rewards[s, i] for s in range(step + 1))` 가 실제로 무엇을 더하는지 설명하라. `n_steps=5` 와 `n_steps=128` 에서 CartPole 의 `Avg(100)` 이 가질 수 있는 최댓값은 각각 얼마인가? 에피소드 리턴을 제대로 기록하도록 고쳐라.

</div>

??? success "풀이"
    이 합은 지금 롤아웃의 처음부터 에피소드가 끝난 스텝까지, 환경 $i$ 의 리워드를 더한다. 에피소드가 이전 롤아웃에서 시작했다면 그 앞부분은 빠지고, 같은 롤아웃 안에서 앞 에피소드가 끝났다면 그 리워드가 섞여 들어간다. CartPole 은 매 스텝 리워드가 1 이므로 기록은 많아야 롤아웃 길이, 곧 `n_steps=5` 에서 5, `n_steps=128` 에서 128 이다. 고치려면 환경마다 누적 리워드 `running[i]` 를 롤아웃을 넘어 들고 다니며 매 스텝 `running[i] += rewards[i]` 를 하고, `done` 이면 그 값을 기록한 뒤 0 으로 되돌린다. `gymnasium.wrappers.RecordEpisodeStatistics` 로 환경을 감싸고 `info["episode"]["r"]` 를 읽어도 된다 — 코드가 이미 `info.get("episode", ...)` 를 찾고 있지만 그 래퍼를 씌우지 않아 늘 `None` 이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`n_steps=5` 인 롤아웃에서 에피소드가 끝나지 않았다고 하자. `compute_returns_and_advantages` 의 n-step 갈래가 $t=0,\dots,4$ 에 만드는 $\hat R_t$ 를 리워드와 $V(s_5)$ 로 적어라. 같은 롤아웃의 스텝들이 서로 다른 길이의 리턴을 받는 것이 편향과 분산에 어떤 뜻을 갖는가? 또 `A2CNetwork` 가 액터 헤드를 `std=0.01`, 크리틱 헤드를 `std=1.0` 으로 초기화하는 까닭은 무엇인가?

</div>

??? success "풀이"
    $\hat R_t = \sum_{k=t}^{4}\gamma^{k-t} r_k + \gamma^{5-t} V(s_5)$ 이다. $t=0$ 은 실제 리워드 다섯 개와 부트스트랩 하나, $t=4$ 는 리워드 하나와 부트스트랩 하나를 쓴다. 실제 리워드가 많을수록 $V$ 의 오차에서 오는 편향은 작고 리워드의 무작위성에서 오는 분산은 크다. 그래서 롤아웃의 끝쪽 스텝일수록 크리틱에 더 기대는, 편향이 큰 타깃을 받는다. 중간에 `done` 이 있으면 $(1-d_t)$ 가 그 뒤를 끊어 에피소드 경계를 넘지 않는다. 액터 헤드를 아주 작게 초기화하면 처음 로짓이 거의 0 이라 폴리시가 균등에 가깝게 출발해 탐색이 고르다. 크리틱 헤드는 리턴 크기의 값을 내야 하므로 출력을 억누르지 않는 `std=1.0` 을 쓴다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
`VecEnv.step` 은 `terminated or truncated` 를 하나의 `done` 으로 합친다. CartPole-v1 이 500 스텝 시간 제한으로 잘렸을 때 이것이 크리틱의 타깃을 어떻게 틀리게 만드는지 설명하고, 이미 저장해 둔 `info["terminal_obs"]` 를 써서 바로잡아라. 고치기 전후를 시드 여럿으로 견주어라.

</div>

??? success "풀이"
    잘림(truncation)은 막대가 쓰러진 것이 아니라 시간이 다 된 것이므로 그 상태의 참 가치는 0 이 아니다. 그런데 `done` 으로 합치면 $\hat R_t = r_t + \gamma\cdot 0$ 이 되어, 잘 버틴 상태일수록 낮은 가치를 배우게 된다. 크리틱은 같은 상태가 어떤 때는 계속 이어지고 어떤 때는 0 으로 끝난다고 보게 되어 타깃이 모순된다. 바로잡으려면 `terminated` 와 `truncated` 를 따로 돌려주고, 잘렸을 때는 리워드에 $\gamma V(s_{\text{terminal}})$ 을 더한 뒤 `done` 으로 끊는다. 여기서 $s_{\text{terminal}}$ 이 `info["terminal_obs"]` 다 — 자동 리셋 뒤의 `obs` 는 새 에피소드의 첫 상태이므로 쓸 수 없다. 좋은 폴리시일수록 잘림을 자주 만나므로 이 차이는 학습 후반에 커진다. 효과를 말하려면 시드를 다섯 개 이상 돌려 평가 리워드의 차이가 시드에 따른 흔들림보다 큰지 본다.

## 정리하며

**다룬 것** — 어드밴티지 Actor-Critic(A2C)

A2C 는 환경 여러 개에서 짧은 롤아웃을 모아 n-step 리턴(또는 GAE)으로 어드밴티지를 만들고, 폴리시·값·엔트로피 항을 더한 하나의 로스로 공유 네트워크를 갱신한다. 이 구현의 학습 로그는 에피소드 리턴을 롤아웃 길이에서 잘라 기록하고 시간 제한을 종료로 다루므로, 성능은 평가 리워드로 읽는다.

핵심 클래스는 `A2CNetwork`, `VecEnv`, `A2C`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
