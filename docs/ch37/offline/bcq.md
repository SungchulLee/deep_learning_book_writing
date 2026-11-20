# 배치 제약 Q 학습(BCQ)

배치 제약 Q 학습은 오프라인 데이터셋이 잘 뒷받침하는 액션만 고르도록 폴리시를 제약해 오프라인 강화학습의 외삽 오차 문제를 다룬다. 행동 복제 모델을 학습해 데이터 분포를 추정하고 Q 학습 동안 분포 밖 액션을 걸러, Q 함수가 본 적 없는 상태-액션 쌍에 잘못 높은 값을 매기는 것을 막는다. 이 구현은 CartPole에서 제약 임계값을 튜닝할 수 있는 이산 BCQ를 보여 준다.

## 1. 코드

```python
"""
35.5.3 배치로 옭아맨 Q 배우기(BCQ)
===========================================

액션 본뜨기 매임을 지닌 조각난 BCQ.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from typing import Dict
import random

# ========================================================================
# 메인
# ========================================================================


class QNetwork(nn.Module):
    def __init__(self, sd, ad, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(sd, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, ad))
    def forward(self, x): return self.net(x)


class BehaviorModel(nn.Module):
    """액션 본뜨기 모델: 자료 묶음에서 P(a|s)를 어림한다."""
    def __init__(self, sd, ad, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(sd, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, ad))
    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

    def log_probs(self, x):
        return torch.log_softmax(self.net(x), dim=-1)


class DiscreteBCQAgent:
    """조각난 액션 자리를 위한 BCQ."""

    def __init__(self, state_dim, action_dim, lr_q=3e-4, lr_bc=1e-3,
                 gamma=0.99, threshold=0.3, target_freq=200):
        self.gamma = gamma
        self.threshold = threshold
        self.action_dim = action_dim
        self.target_freq = target_freq

        # Q 네트워크
        self.online = QNetwork(state_dim, action_dim)
        self.target = QNetwork(state_dim, action_dim)
        self.target.load_state_dict(self.online.state_dict())
        self.q_opt = optim.Adam(self.online.parameters(), lr=lr_q)

        # 액션 본
        self.bc_model = BehaviorModel(state_dim, action_dim)
        self.bc_opt = optim.Adam(self.bc_model.parameters(), lr=lr_bc)

        self.update_count = 0

    def train_behavior_model(self, states, actions, n_steps=3000, batch_size=256):
        """액션 본뜨기 본을 미리 익힌다."""
        n = len(actions)
        losses = []
        for step in range(n_steps):
            idx = np.random.randint(0, n, batch_size)
            s = torch.FloatTensor(states[idx])
            a = torch.LongTensor(actions[idx])
            log_p = self.bc_model.log_probs(s)
            loss = nn.functional.nll_loss(log_p, a)
            self.bc_opt.zero_grad(); loss.backward(); self.bc_opt.step()
            losses.append(loss.item())
        return losses

    def _filter_actions(self, states: torch.Tensor) -> torch.Tensor:
        """액션 본의 임계를 바탕으로 액션 마스크를 만든다."""
        with torch.no_grad():
            probs = self.bc_model(states)  # (batch, action_dim)
            max_probs = probs.max(dim=1, keepdim=True)[0]
            mask = (probs / max_probs >= self.threshold).float()
            # 적어도 액션 하나는 허락되게 함
            mask[mask.sum(1) == 0] = 1.0
        return mask

    def train_q_step(self, states, actions, rewards, next_states, dones,
                     batch_size=256) -> Dict[str, float]:
        """BCQ 액션 거르기를 하는 Q 러닝 한 스텝."""
        n = len(rewards)
        idx = np.random.randint(0, n, batch_size)
        s = torch.FloatTensor(states[idx])
        a = torch.LongTensor(actions[idx])
        r = torch.FloatTensor(rewards[idx])
        ns = torch.FloatTensor(next_states[idx])
        d = torch.FloatTensor(dones[idx])

        # 지금 Q
        q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # BCQ 타깃: 분포 안의 액션만 살핌
        with torch.no_grad():
            next_q = self.target(ns)
            mask = self._filter_actions(ns)
            # 걸러진 액션의 Q를 -inf로 둠
            next_q_masked = next_q * mask + (1 - mask) * (-1e8)
            best_next_q = next_q_masked.max(dim=1)[0]
            targets = r + (1 - d) * self.gamma * best_next_q

        loss = nn.functional.mse_loss(q, targets)
        self.q_opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.q_opt.step()

        self.update_count += 1
        if self.update_count % self.target_freq == 0:
            self.target.load_state_dict(self.online.state_dict())

        # 통계
        n_allowed = mask.sum(1).mean().item()
        return {'loss': loss.item(), 'avg_allowed_actions': n_allowed}

    def select_action(self, state: np.ndarray) -> int:
        s = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q = self.online(s)
            mask = self._filter_actions(s)
            q_masked = q * mask + (1 - mask) * (-1e8)
        return q_masked.argmax(1).item()


def collect_dataset(env_name='CartPole-v1', n=5000, seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = gym.make(env_name)
    sd = env.observation_space.shape[0]; ad = env.action_space.n
    q = nn.Sequential(nn.Linear(sd,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,ad))
    tgt = nn.Sequential(nn.Linear(sd,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,ad))
    tgt.load_state_dict(q.state_dict()); o = optim.Adam(q.parameters(),lr=1e-3)
    buf={'s':[],'a':[],'r':[],'ns':[],'d':[]}; step=0
    for _ in range(200):
        s,_=env.reset();done=False
        while not done:
            step+=1;eps=max(0.05,1.0-step/3000)
            a=env.action_space.sample() if random.random()<eps else q(torch.FloatTensor(s).unsqueeze(0)).argmax(1).item()
            ns,r,term,trunc,_=env.step(a);done=term or trunc
            for k,v in zip(['s','a','r','ns','d'],[s,a,r,ns,float(done)]):buf[k].append(v)
            if len(buf['s'])>64:
                idx=np.random.randint(0,len(buf['s']),64)
                st=torch.FloatTensor(np.array(buf['s'])[idx]);at=torch.LongTensor(np.array(buf['a'])[idx])
                rt=torch.FloatTensor(np.array(buf['r'])[idx]);nst=torch.FloatTensor(np.array(buf['ns'])[idx])
                dt=torch.FloatTensor(np.array(buf['d'])[idx])
                qv=q(st).gather(1,at.unsqueeze(1)).squeeze(1)
                with torch.no_grad():t_=rt+(1-dt)*0.99*tgt(nst).max(1)[0]
                lo=nn.functional.mse_loss(qv,t_);o.zero_grad();lo.backward();o.step()
                if step%200==0:tgt.load_state_dict(q.state_dict())
            s=ns
    S,A,R,NS,D=[],[],[],[],[];c=0
    while c<n:
        s,_=env.reset();done=False
        while not done and c<n:
            a=env.action_space.sample() if random.random()<0.3 else q(torch.FloatTensor(s).unsqueeze(0)).argmax(1).item()
            ns,r,term,trunc,_=env.step(a);done=term or trunc
            S.append(s);A.append(a);R.append(r);NS.append(ns);D.append(float(done));s=ns;c+=1
    env.close()
    return {k:np.array(v,dt) for k,v,dt in zip(['states','actions','rewards','next_states','dones'],
            [S,A,R,NS,D],[np.float32,np.int64,np.float32,np.float32,np.float32])}


def demo_bcq():
    print("=" * 60)
    print("BCQ Demo")
    print("=" * 60)

    data = collect_dataset(n=5000)
    sd = data['states'].shape[1]; ad = int(data['actions'].max())+1
    print(f"\nDataset: {len(data['rewards'])} transitions, {ad} actions")

    for tau in [0.0, 0.1, 0.3, 0.5]:
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        agent = DiscreteBCQAgent(sd, ad, threshold=tau)

        # 액션 본 미리 익히기
        bc_losses = agent.train_behavior_model(data['states'], data['actions'], n_steps=2000)
        print(f"\n  τ={tau}: BC loss = {np.mean(bc_losses[-100:]):.4f}")

        # Q 네트워크 익히기
        for step in range(5000):
            agent.train_q_step(data['states'], data['actions'], data['rewards'],
                               data['next_states'], data['dones'])

        # 평가한다
        env = gym.make('CartPole-v1'); rets = []
        for _ in range(20):
            s,_=env.reset();t=0;done=False
            while not done:
                a=agent.select_action(s);s,r,term,trunc,_=env.step(a);done=term or trunc;t+=r
            rets.append(t)
        env.close()
        print(f"    Eval: {np.mean(rets):.1f} ± {np.std(rets):.1f}")

    print("\nBCQ demo complete!")


if __name__ == "__main__":
    demo_bcq()
```

## 2. 논의

이 구현은 세 클래스로 되어 있다. `BehaviorModel` 은 데이터셋의 (상태, 행동) 쌍으로 행동 폴리시 $\hat\pi_\beta(a|s)$ 를 배우는 분류기, 곧 행동 복제(behavior cloning) 모델이다. `DiscreteBCQAgent._filter_actions` 는 이 모델의 확률로 $\hat\pi_\beta(a|s)/\max_{a'}\hat\pi_\beta(a'|s) \ge \tau$ 인 행동만 남기는 마스크를 만들고, TD 타깃의 $\max_{a'}$ 과 행동 선택의 $\arg\max$ 를 모두 이 마스크 안에서만 한다. 데이터에서 드물게 나온 행동은 Q 값이 아무리 커도 고르지 않으므로, 외삽된 Q 값이 타깃으로 번지지 않는다. $\tau = 0$ 이면 모든 행동이 남아 제약 없는 오프라인 DQN 과 같아진다.

데모 함수는 DQN 을 200 에피소드 온라인으로 학습한 뒤 그 그리디 폴리시에 30% 무작위 행동을 섞어 전이 5,000 개의 데이터셋을 만든다. 그다음 $\tau$ 를 0, 0.1, 0.3, 0.5 로 바꿔 가며, 매번 같은 시드 42 로 행동 복제 모델을 2,000 스텝, Q 네트워크를 5,000 스텝 학습하고, 실제 둘레에서 20 에피소드 평가한다. 평가 둘레는 시드를 받지 않으므로 평가 리턴은 돌릴 때마다 조금씩 다를 수 있고, 설정마다 한 번만 돌리므로 작은 차이는 우열로 읽을 수 없다. 행동이 둘뿐인 CartPole 에서는 $\tau$ 가 어느 값을 넘으면 Q 네트워크가 행동 선택에 아무 역할도 하지 못하게 된다는 점도 눈여겨보아야 한다(연습문제 2).

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
데모의 데이터셋은 그리디 행동을 70%, 무작위 행동을 30% 의 비율로 고른다(무작위 행동이 그리디 행동과 같을 수도 있다). 행동 복제 모델이 이 비율을 완벽히 배웠다고 할 때, 한 상태에서 그리디 행동과 다른 행동의 확률은 각각 얼마인가? `_filter_actions` 의 비 $\hat\pi_\beta(a|s)/\max_{a'}\hat\pi_\beta(a'|s)$ 로 보면, 데모의 $\tau$ 네 값 가운데 어느 것이 두 행동을 모두 남기는가?

</div>

??? success "풀이"
    무작위 행동은 두 행동을 반반으로 고르므로 그리디 행동의 확률은 $0.7 + 0.3 \times \tfrac12 = 0.85$, 다른 행동은 $0.3 \times \tfrac12 = 0.15$ 다. 비로 보면 그리디 행동은 1, 다른 행동은 $0.15/0.85 \approx 0.18$ 이다. 따라서 $\tau = 0$ 과 $0.1$ 은 두 행동을 모두 남기고, $\tau = 0.3$ 과 $0.5$ 는 그리디 행동 하나만 남긴다. 실제로는 행동 복제 모델이 이 비율을 상태마다 정확히 맞히지 못하고, 데이터를 모은 Q 네트워크의 그리디 행동이 상태 공간의 경계 근처에서 바뀌므로 비가 0.18 근처에서 흩어진다. 그래서 $\tau = 0.1$ 에서도 일부 상태에서는 한 행동만 남을 수 있다. 학습 중 `train_q_step` 이 돌려주는 `avg_allowed_actions` 를 찍어 보면 이것을 직접 확인할 수 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`_filter_actions` 에는 「적어도 액션 하나는 허락되게 함」이라는 안전장치 `mask[mask.sum(1) == 0] = 1.0` 이 있다. $0 \le \tau \le 1$ 에서 이 줄이 실제로 쓰이는 일이 있는가? 비 대신 절대 확률 $\hat\pi_\beta(a|s) \ge \tau$ 로 거르면 무엇이 달라지는가?

</div>

??? success "풀이"
    가장 확률이 큰 행동은 비가 정확히 1 이므로 $\tau \le 1$ 이면 늘 남는다. 따라서 마스크의 합이 0 이 되는 일은 없고, 이 안전장치는 $\tau > 1$ 같은 잘못된 설정에서만 작동한다. 비로 거르는 까닭이 바로 이것이다. 절대 확률로 거르면 행동이 여럿이고 행동 폴리시가 고르게 퍼진 상태에서는 모든 행동의 확률이 $\tau$ 보다 작아 아무 행동도 남지 않을 수 있고, 그때마다 안전장치가 모든 행동을 풀어 제약이 사라진다. 비는 상태마다 「가장 흔한 행동에 견주어 얼마나 드문가」를 재므로, 행동 폴리시가 고르게 퍼진 상태에서는 많은 행동을, 한쪽으로 쏠린 상태에서는 적은 행동을 남긴다. Fujimoto 등이 이산 BCQ 에서 비를 쓴 것도 이 때문이다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
연습문제 1 의 결과대로 $\tau = 0.3$ 이나 $0.5$ 에서 대부분의 상태에 행동이 하나만 남는다면, `select_action` 이 고르는 행동은 Q 네트워크와 어떤 관계가 있는가? 데모의 네 줄은 각각 무엇을 평가한 것인지 다시 해석하고, 이 비교를 공정하게 만들려면 무엇을 함께 돌려야 하는지 적어라.

</div>

??? success "풀이"
    행동이 하나만 남으면 `q_masked.argmax` 는 Q 값과 상관없이 그 행동을 고른다. 그 행동은 행동 복제 모델이 가장 확률이 높다고 본 행동이므로, 에이전트는 사실상 **행동 복제 폴리시**다. Q 네트워크는 학습되지만 행동 선택에 쓰이지 않는다(타깃 계산에서도 $\max$ 가 한 행동에 대해서만 이루어지므로 SARSA 처럼 행동 복제 폴리시의 가치를 배운다). 그러니 $\tau = 0$ 은 제약 없는 오프라인 DQN, $\tau = 0.1$ 은 대체로 제약 없는 DQN 에 가까운 BCQ, $\tau = 0.3$·$0.5$ 는 데이터를 모은 DQN 의 그리디 폴리시를 따라 하는 행동 복제를 평가한 것이다. 이 데이터셋은 그리디 행동이 85% 나 되므로 행동 복제가 꽤 잘할 수 있고, 그 결과를 「BCQ 가 잘한다」로 읽으면 안 된다. 공정하게 견주려면 (1) 행동 복제 모델의 argmax 만으로 평가한 순수 행동 복제, (2) 데이터를 모은 폴리시 자체의 평가 리턴을 기준선으로 함께 돌리고, 시드 여러 개로 흩어짐을 잰다. BCQ 의 제약이 의미 있게 작동하려면 행동이 많아 「드문 행동」과 「흔한 행동」 사이에 중간 지대가 있어야 하는데, Fujimoto 등이 이산 BCQ 를 아타리(행동 최대 18 개)에서 $\tau = 0.3$ 으로 시험한 것도 그 때문이다.

## 정리하며

**다룬 것** — 배치 제약 Q 학습(BCQ)

이 구현은 행동 폴리시를 배우는 `BehaviorModel` 과, 그 확률의 비로 드문 행동을 걸러 낸 뒤 Q 러닝을 하는 `DiscreteBCQAgent` 를 중심에 둔다.

핵심 클래스는 `QNetwork`, `BehaviorModel`, `DiscreteBCQAgent`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
