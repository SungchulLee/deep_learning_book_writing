# 묶음에 매인 Q 배우기(BCQ)

묶음에 매인 Q 배우기는 오프라인 자료 뭉치가 잘 받쳐 주는 움직임만 고르도록 방침을 매어 오프라인 힘 북돋우는 배움의 바깥으로 늘려 짚는 어긋남 문제를 다룬다. 움직임 본뜨기 모델을 익혀 자료 분포를 어림하고 Q 배우기 동안 분포 밖 움직임을 걸러, Q 함수가 본 적 없는 상태-움직임 짝에 잘못 높은 값을 매기는 것을 막는다. 이 짜기는 CartPole에서 매임 문턱을 손볼 수 있는 띄엄띄엄 BCQ을 보여 준다.

## 1. 코드

```python
"""
33.5.3 묶음으로 옭아맨 Q 배우기(BCQ)
===========================================

움직임 본뜨기 매임을 지닌 조각난 BCQ.
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
    """행동 본뜨기 모델: 자료 묶음에서 P(a|s)를 어림한다."""
    def __init__(self, sd, ad, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(sd, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, ad))
    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

    def log_probs(self, x):
        return torch.log_softmax(self.net(x), dim=-1)


class DiscreteBCQAgent:
    """조각난 움직임 자리를 위한 BCQ."""

    def __init__(self, state_dim, action_dim, lr_q=3e-4, lr_bc=1e-3,
                 gamma=0.99, threshold=0.3, target_freq=200):
        self.gamma = gamma
        self.threshold = threshold
        self.action_dim = action_dim
        self.target_freq = target_freq

        # Q 그물
        self.online = QNetwork(state_dim, action_dim)
        self.target = QNetwork(state_dim, action_dim)
        self.target.load_state_dict(self.online.state_dict())
        self.q_opt = optim.Adam(self.online.parameters(), lr=lr_q)

        # 움직임 본
        self.bc_model = BehaviorModel(state_dim, action_dim)
        self.bc_opt = optim.Adam(self.bc_model.parameters(), lr=lr_bc)

        self.update_count = 0

    def train_behavior_model(self, states, actions, n_steps=3000, batch_size=256):
        """움직임 본뜨기 본을 미리 익힌다."""
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
        """움직임 본의 문턱을 바탕으로 움직임 가리개를 만든다."""
        with torch.no_grad():
            probs = self.bc_model(states)  # (batch, action_dim)
            max_probs = probs.max(dim=1, keepdim=True)[0]
            mask = (probs / max_probs >= self.threshold).float()
            # 적어도 움직임 하나는 허락되게 함
            mask[mask.sum(1) == 0] = 1.0
        return mask

    def train_q_step(self, states, actions, rewards, next_states, dones,
                     batch_size=256) -> Dict[str, float]:
        """BCQ 움직임 거르기를 하는 Q 배움 한 걸음."""
        n = len(rewards)
        idx = np.random.randint(0, n, batch_size)
        s = torch.FloatTensor(states[idx])
        a = torch.LongTensor(actions[idx])
        r = torch.FloatTensor(rewards[idx])
        ns = torch.FloatTensor(next_states[idx])
        d = torch.FloatTensor(dones[idx])

        # 지금 Q
        q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # BCQ 과녁: 분포 안의 움직임만 살핌
        with torch.no_grad():
            next_q = self.target(ns)
            mask = self._filter_actions(ns)
            # 걸러진 움직임의 Q를 -inf로 둠
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

        # 움직임 본 미리 익히기
        bc_losses = agent.train_behavior_model(data['states'], data['actions'], n_steps=2000)
        print(f"\n  τ={tau}: BC loss = {np.mean(bc_losses[-100:]):.4f}")

        # Q 그물 익히기
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
    demo_bcq()```

## 2. 논의

이 짜기는 BCQ의 핵심 논리를 감싼 `QNetwork`, `BehaviorModel`, `DiscreteBCQAgent` 갈래를 한가운데 둔다. 코드는 알고리즘 조각을 보여 주기와 따지기 논리에서 떼어 놓는 조각 짜기를 따른다.

보여 주기 함수는 이 조각들을 여느 힘 북돋우는 배움 잣대에 실제로 쓰는 모습을 보인다. 내놓기를 살피면 웃잡 고름과 문제 짜임에 따라 알고리즘의 성능이 어떻게 달라지는지 볼 수 있다.

실제 관점에서 이 짜기는 순수한 성능보다 또렷함을 앞세운다. 실제로 쓰는 얼개는 보통 묶음 셈, GPU 빠르게 하기, 더 정교한 윗매개변수 맞추기 같은 개선을 더한다. 그럼에도 여기 보인 핵심 알고리즘 생각은 큰 규모의 쓰임새로 곧바로 옮겨 간다.

## 연습문제

**연습문제 1.**
보여 주기 코드를 돌려 핵심 내놓기 잣대를 적어라. 윗매개변수 하나(배움 빠르기, 숨은 차원, 층 개수 같은 것)를 고치고 결과가 어떻게 바뀌는지 적어라.

??? success "연습문제 1 풀이"
    보여 주기를 돌린 뒤 나머지를 붙박아 두고 고른 윗매개변수를 차근히 바꾼다. 보기로 숨은 차원을 두 배로 하면 보통 나타냄 담이가 늘지만 셈 시간이 커진다. 배움 빠르기는 단조롭지 않은 영향을 준다. 너무 작으면 느리게 모이고 너무 크면 흔들린다. 고른 윗매개변수의 서로 다른 값을 적어도 셋 잡아 구체적인 수를 적어 두라.

---

**연습문제 2.**
이 짜기에서 핵심 얼개 고르기의 몫을 밝혀라. 왜 그 깨움 함수, 고르게 맞추기 셈속, 손실 함수를 쓰는가? 다른 것으로 바꾸면 어떻게 되는가?

??? success "연습문제 2 풀이"
    이 얼개 고르기는 오프라인 힘 북돋우는 배움에서 자리 잡은 가장 좋은 방식을 따른다. 보기로 ReLU 깨움은 비선형을 주면서 양의 들임에서 기울기가 사라지는 것을 피한다. 손실 함수는 일의 갈래에 맞게 고른다(가름에는 교차 엔트로피, 되돌이 맞춤에는 평균 제곱 어긋남). 다른 것으로 바꾸면(보기로 시그모이드 깨움, L1 손실) 가장 좋게 하기의 풍경이 바뀌어 성능이 나빠질 수 있지만 어떤 상황에서는 바꿈이 이로울 수도 있다.

---

**연습문제 3.**
더 어려운 상황을 다루도록 짜기를 넓혀라. 더 큰 자료 묶음, 다른 문제 변형, 특징 하나 더하기 가운데 하나이다. 고침을 적고 성능에 미친 영향을 따져라.

??? success "연습문제 3 풀이"
    자연스러운 넓힘 하나는 규칙 세우기(떨구기, 무게 줄임)나 더 정교한 얼개(층 더하기, 건너뛰는 이음)를 더하는 것이다. 고른 넓힘을 짜고 같은 자료로 익힌 뒤 앞뒤의 잣대를 견주어라. 이 넓힘은 본디 알고리즘과 고침의 이론 까닭을 모두 이해했음을 보여야 한다.

## 정리하며

**다룬 것** — 묶음에 매인 Q 배우기(BCQ)

이 짜기는 BCQ의 핵심 논리를 감싼 `QNetwork`, `BehaviorModel`, `DiscreteBCQAgent` 갈래를 한가운데 둔다.

고갱이 갈래는 `QNetwork`, `BehaviorModel`, `DiscreteBCQAgent`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
