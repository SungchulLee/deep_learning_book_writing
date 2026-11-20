# 배치 제약 Q 학습(BCQ)

배치 제약 Q 학습은 오프라인 데이터셋이 잘 뒷받침하는 액션만 고르도록 폴리시를 제약해 오프라인 강화학습의 외삽 오차 문제를 다룬다. 행동 복제 모델을 학습해 데이터 분포를 추정하고 Q 학습 동안 분포 밖 액션을 걸러, Q 함수가 본 적 없는 상태-액션 쌍에 잘못 높은 값을 매기는 것을 막는다. 이 구현은 CartPole에서 제약 임계값을 튜닝할 수 있는 이산 BCQ을 보여 준다.

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

이 구현은 BCQ의 핵심 논리를 감싼 `QNetwork`, `BehaviorModel`, `DiscreteBCQAgent` 클래스를 중심에 둔다. 코드는 알고리즘 구성 요소를 데모·평가 논리와 분리하는 모듈식 설계를 따른다.

데모 함수는 이 구성 요소들을 전형적인 강화학습 벤치마크에 실제로 적용하는 모습을 보인다. 출력을 살피면 하이퍼파라미터 선택과 문제 구조에 따라 알고리즘의 성능이 어떻게 달라지는지 볼 수 있다.

실용적 관점에서 이 구현은 순수한 성능보다 명료함을 앞세운다. 실제로 쓰는 시스템은 보통 배치 계산, GPU 가속, 더 정교한 하이퍼파라미터 튜닝 같은 개선을 더한다. 그럼에도 여기 보인 핵심 알고리즘 아이디어는 대규모 응용으로 그대로 옮겨 간다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
데모 코드를 돌려 핵심 출력 지표를 기록하라. 하이퍼파라미터 하나(러닝레이트, 은닉 차원, 레이어 개수 같은 것)를 바꾸고 결과가 어떻게 바뀌는지 기록하라.

</div>

??? success "풀이"
    데모를 돌린 뒤 나머지를 고정하고 고른 하이퍼파라미터를 차례로 바꾼다. 예를 들어 은닉 차원을 두 배로 하면 보통 표현력이 늘지만 계산 시간이 커진다. 러닝레이트는 단조롭지 않은 영향을 준다. 너무 작으면 느리게 수렴하고 너무 크면 진동한다. 고른 하이퍼파라미터의 서로 다른 값을 적어도 셋 잡아 구체적인 수를 기록해 두라.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 구현에서 핵심 구조 선택의 역할을 설명하라. 왜 그 활성화 함수, 정규화 전략, 로스 함수를 쓰는가? 다른 것으로 바꾸면 어떻게 되는가?

</div>

??? success "풀이"
    이 구조 선택은 오프라인 강화학습에서 자리 잡은 모범 사례를 따른다. 예를 들어 ReLU 활성화는 비선형성을 주면서 양의 입력에서 그래디언트가 사라지는 것을 피한다. 로스 함수는 과제 유형에 맞게 고른다(분류에는 교차 엔트로피, 회귀에는 평균 제곱 오차). 다른 것으로 바꾸면(예를 들어 시그모이드 활성화, L1 로스) 최적화 지형이 바뀌어 성능이 나빠질 수 있지만 어떤 상황에서는 바꾸는 것이 이로울 수도 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
더 어려운 상황을 다루도록 구현을 확장하라. 더 큰 데이터셋, 다른 문제 변형, 피처 추가 가운데 하나를 고른다. 수정 사항을 적고 성능에 미친 영향을 분석하라.

</div>

??? success "풀이"
    자연스러운 확장 하나는 정칙화(드롭아웃, 웨이트 감쇠)나 더 정교한 구조(레이어 추가, 스킵 연결)를 더하는 것이다. 고른 확장을 구현하고 같은 데이터로 학습한 뒤 전후의 지표를 비교하여라. 이 확장은 원래 알고리즘과 수정 사항의 이론적 근거를 모두 이해했음을 보여야 한다.

## 정리하며

**다룬 것** — 배치 제약 Q 학습(BCQ)

이 구현은 BCQ의 핵심 논리를 감싼 `QNetwork`, `BehaviorModel`, `DiscreteBCQAgent` 클래스를 중심에 둔다.

핵심 클래스는 `QNetwork`, `BehaviorModel`, `DiscreteBCQAgent`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
