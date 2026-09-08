# MAML

모델 가리지 않는 메타 학습(MAML). 참고: Finn 외, "Model-Agnostic Meta-Learning for Fast Adaptation" (2017)

모자란 데이터나 서로 이어진 데이터에서 효율적으로 배우는 것은 오늘날 깊은 학습의 한가운데 놓인 어려움이다. 이 모듈은 모델이 앞선 앎을 살려 새 과제에 재빨리 맞추어 가게 하는 소수 예시 학습 기법을 보여 준다.

## 1. 코드

```python
"""
모델 가리지 않는 메타 학습(MAML)

참고: Finn 외, "Model-Agnostic Meta-Learning for Fast Adaptation" (2017)

핵심 생각: 기울기 걸음 몇 번만으로 새 과제에 재빨리 맞추어 갈 수 있게 하는
초기화를 배운다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# ========================================================================
# 메인
# ========================================================================


class SimpleClassifier(nn.Module):
    """
    MAML을 위한 단순한 신경망.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, params=None):
        """
        매개변수를 갈아 끼울 수 있는 앞먹임.
        
        인수:
            x: 입력 텐서
            params: self.parameters() 대신 쓸 매개변수의 OrderedDict(선택)
        """
        if params is None:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        else:
            x = F.relu(F.linear(x, params['fc1.weight'], params['fc1.bias']))
            x = F.relu(F.linear(x, params['fc2.weight'], params['fc2.bias']))
            x = F.linear(x, params['fc3.weight'], params['fc3.bias'])
        return x


class MAML:
    """
    모델 가리지 않는 메타 학습 알고리즘.
    
    인수:
        model: 신경망 모델
        inner_lr: 안쪽 되돌이(과제 맞춤)의 학습률
        meta_lr: 바깥 되돌이(메타 갱신)의 학습률
        num_inner_steps: 안쪽 되돌이의 기울기 걸음 수
    """
    def __init__(self, model, inner_lr=0.01, meta_lr=0.001, num_inner_steps=5):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.num_inner_steps = num_inner_steps
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=meta_lr)
    
    def inner_loop(self, support_x, support_y, query_x, query_y):
        """
        과제 하나에 대해 안쪽 되돌이 맞춤을 한다.
        
        반환값:
            query_loss: 맞춘 뒤 물음 집합에서의 손실
        """
        # 과제별 맞춤을 위해 모델 매개변수를 복사한다
        params = OrderedDict(self.model.named_parameters())
        
        # 안쪽 되돌이: 받침 집합에 맞춘다
        for step in range(self.num_inner_steps):
            # 지금 매개변수로 앞먹임한다
            support_logits = self.model(support_x, params)
            support_loss = F.cross_entropy(support_logits, support_y)
            
            # 매개변수에 대한 기울기를 셈한다
            grads = torch.autograd.grad(
                support_loss,
                params.values(),
                create_graph=True  # 이차 기울기에 중요하다
            )
            
            # 기울기 내려가기로 매개변수를 고친다
            params = OrderedDict(
                (name, param - self.inner_lr * grad)
                for ((name, param), grad) in zip(params.items(), grads)
            )
        
        # 맞춘 매개변수로 물음 집합에서 평가한다
        query_logits = self.model(query_x, params)
        query_loss = F.cross_entropy(query_logits, query_y)
        
        return query_loss
    
    def meta_train_step(self, tasks):
        """
        과제 배치에 대해 메타 학습 걸음을 한 번 밟는다.
        
        인수:
            tasks: (support_x, support_y, query_x, query_y) 짝의 목록
        
        반환값:
            meta_loss: 과제에 걸친 평균 물음 손실
            meta_accuracy: 물음 집합에서의 평균 정확도
        """
        self.meta_optimizer.zero_grad()
        
        meta_loss = 0.0
        meta_accuracy = 0.0
        
        # 과제마다 안쪽 되돌이를 돌고 기울기를 쌓는다
        for support_x, support_y, query_x, query_y in tasks:
            task_loss = self.inner_loop(support_x, support_y, query_x, query_y)
            meta_loss += task_loss
            
            # 이 과제의 정확도를 셈한다
            with torch.no_grad():
                # 맞춘 매개변수를 얻는다(그래프 없이 안쪽 되돌이를 다시 돈다)
                params = OrderedDict(self.model.named_parameters())
                for step in range(self.num_inner_steps):
                    support_logits = self.model(support_x, params)
                    support_loss = F.cross_entropy(support_logits, support_y)
                    grads = torch.autograd.grad(support_loss, params.values())
                    params = OrderedDict(
                        (name, param - self.inner_lr * grad)
                        for ((name, param), grad) in zip(params.items(), grads)
                    )
                
                query_logits = self.model(query_x, params)
                predictions = torch.argmax(query_logits, dim=1)
                accuracy = (predictions == query_y).float().mean()
                meta_accuracy += accuracy
        
        # 과제들에 걸쳐 평균 낸다
        meta_loss = meta_loss / len(tasks)
        meta_accuracy = meta_accuracy / len(tasks)
        
        # 메타 갱신: 처음 매개변수를 고친다
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item(), meta_accuracy.item()
    
    def adapt(self, support_x, support_y, num_steps=None):
        """
        받침 집합이 주어졌을 때 모델을 새 과제에 맞춘다.
        
        반환값:
            adapted_params: 과제에 맞춘 매개변수
        """
        if num_steps is None:
            num_steps = self.num_inner_steps
        
        params = OrderedDict(self.model.named_parameters())
        
        for step in range(num_steps):
            support_logits = self.model(support_x, params)
            support_loss = F.cross_entropy(support_logits, support_y)
            
            grads = torch.autograd.grad(support_loss, params.values())
            params = OrderedDict(
                (name, param - self.inner_lr * grad)
                for ((name, param), grad) in zip(params.items(), grads)
            )
        
        return params
    
    def predict(self, support_x, support_y, query_x):
        """
        받침 집합에 맞춘 뒤 물음 집합에서 맞힌다.
        """
        self.model.eval()
        with torch.no_grad():
            adapted_params = self.adapt(support_x, support_y)
            query_logits = self.model(query_x, adapted_params)
            predictions = torch.argmax(query_logits, dim=1)
        return predictions


# 사용 예
if __name__ == "__main__":
    # 모형 설정
    input_dim = 784  # 편 28x28 그림
    hidden_dim = 128
    output_dim = 5  # 5-갈래 분류
    
    # 모델과 MAML 학습기를 만든다
    model = SimpleClassifier(input_dim, hidden_dim, output_dim)
    maml = MAML(
        model,
        inner_lr=0.01,
        meta_lr=0.001,
        num_inner_steps=5
    )
    
    # 과제 배치를 만든다(5-갈래 5-예시)
    num_tasks = 4
    n_way = 5
    k_shot = 5
    n_query = 15
    
    tasks = []
    for _ in range(num_tasks):
        support_x = torch.randn(n_way * k_shot, input_dim)
        support_y = torch.arange(n_way).repeat_interleave(k_shot)
        query_x = torch.randn(n_query, input_dim)
        query_y = torch.randint(0, n_way, (n_query,))
        tasks.append((support_x, support_y, query_x, query_y))
    
    # 메타 학습 걸음
    meta_loss, meta_acc = maml.meta_train_step(tasks)
    print(f"Meta-Loss: {meta_loss:.4f}, Meta-Accuracy: {meta_acc:.4f}")
    
    # 새 과제로의 맞춤을 시험한다
    test_support_x = torch.randn(n_way * k_shot, input_dim)
    test_support_y = torch.arange(n_way).repeat_interleave(k_shot)
    test_query_x = torch.randn(n_query, input_dim)
    
    predictions = maml.predict(test_support_x, test_support_y, test_query_x)
    print(f"Predictions: {predictions}")```

## 2. 논의

이 구현은 함께 어울려 온전한 소수 예시 학습 구조를 이루는 클래스 2개(`SimpleClassifier`, `MAML`)를 정한다. 클래스마다 서로 다른 부품을 감싸 코드를 모듈 방식으로 만들고 넓히기 쉽게 한다. `forward` 메서드가 파이토치가 자동 미분에 쓰는 계산 그래프를 정한다.

손실 계산은 모델의 출력을 최적화 목표와 이어 준다. 알맞은 손실 함수를 고르는 일은 결정적으로 중요하다. 손실 함수가 모델이 무엇을 최적화하도록 배울지를 정하며, 학습된 표현과 결정 경계를 직접 빚어내기 때문이다.

여기서 보인 본새는 더 복잡한 상황으로도 자연스럽게 넓어진다. 초매개변수, 구조의 변형, 여러 데이터셋을 두고 실험해 보면 이해가 깊어지고 메타 학습 과제에 대한 실전 감각이 쌓인다.

## 연습문제

**연습문제 1.**
기본 초기화 상태의 `SimpleClassifier`에 든 학습 가능한 매개변수의 총 개수를 셈하라. 가중치와 편향을 모두 넣어 층별로 나누어 보여라.

??? success "연습문제 1 풀이"
    `nn.Linear(in_features, out_features)` 각각에는 `in_features * out_features`개의 가중치 매개변수와 (`bias=False`가 아닌 한) `out_features`개의 편향 매개변수가 있다. `nn.Conv2d(in_c, out_c, k)`에는 `in_c * out_c * k * k`개의 가중치와 `out_c`개의 편향이 있다. `nn.Embedding(num, dim)`에는 `num * dim`개의 매개변수가 있다. 모든 층에 대해 더하면 된다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다.

---

**연습문제 2.**
입력이 기대하는 모양과 자료형을 갖는지 확인하도록 주 함수나 클래스에 입력 검증을 추가하라. 잘못된 입력에는 유익한 오류 메시지를 내라.

??? success "연습문제 2 풀이"
    `forward` 메서드(또는 해당 함수)의 첫머리에 다음과 같은 검사를 추가한다. `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`와 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'`. 모양을 검증할 때는 중요한 차원을 확인한다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 유익한 오류 메시지는 디버깅 속도를 크게 높이고 코드를 재사용하기에도 더 견고하게 만든다.

---

**연습문제 3.**
이 구현에서 생길 수 있는 실패 양상 두 가지를 서술하고, 각각을 어떻게 진단하고 고칠지 설명하라.

??? success "연습문제 3 풀이"
    흔한 실패 양상은 다음과 같다. (1) **경사 소실/폭발** — 경사의 노름을 지켜보아 진단한다(`torch.nn.utils.clip_grad_norm_`을 쓰거나 층마다 `param.grad.norm()`을 기록한다). 경사 자르기, 더 나은 초기화(Xavier/Kaiming), 또는 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** — 학습 손실은 줄어드는데 검증 손실이 늘어나면 진단된다. 정칙화(드롭아웃, 가중치 감쇠, 데이터 증강)나 모델 용량 축소로 고친다. 이런 문제를 일찍 잡아내려면 언제나 학습 지표와 검증 지표를 함께 살펴라.

---

**연습문제 4.**
`SimpleClassifier`이 층이나 블록의 개수를 설정할 수 있도록 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`를 써서 깊이를 바꿀 수 있는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`처럼 되풀이하라. (그냥 파이썬 리스트가 아니라) `nn.ModuleList`를 써야 파이토치가 최적화 대상 매개변수를 모두 등록한다. `for n in [2, 4, 8]: model = SimpleClassifier(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`로 시험하라.

## 정리하며

**다룬 것** — MAML

이 구현은 함께 어울려 온전한 소수 예시 학습 구조를 이루는 클래스 2개(`SimpleClassifier`, `MAML`)를 정한다.

핵심 클래스는 `SimpleClassifier`, `MAML`이며 앞의 연습문제 4개로 직접 확인할 수 있다.
