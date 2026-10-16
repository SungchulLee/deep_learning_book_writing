# MAML

MAML(model-agnostic meta-learning)은 적은 서포트 데이터로 몇 번 갱신했을 때 새 과제에 잘 적응하는 초기 웨이트를 학습한다. Finn 등의 「Model-Agnostic Meta-Learning for Fast Adaptation」(2017)에 기반해, 함수형 웨이트 전달로 내부 갱신을 미분하는 예제다.

## 1. 코드

```python
"""
모델 가리지 않는 메타 학습(MAML)

참고: Finn 외, "Model-Agnostic Meta-Learning for Fast Adaptation" (2017)

핵심 생각: 그래디언트 걸음 몇 번만으로 새 과제에 재빨리 맞추어 갈 수 있게 하는
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
    MAML을 위한 단순한 뉴럴 넷트웍.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, params=None):
        """
        파라메터를 갈아 끼울 수 있는 피드포워드.
        
        인수:
            x: 입력 텐서
            params: self.parameters() 대신 쓸 파라메터의 OrderedDict(선택)
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
        model: 뉴럴 넷트웍 모델
        inner_lr: 안쪽 회귀(과제 맞춤)의 러닝레이트
        meta_lr: 바깥 회귀(메타 갱신)의 러닝레이트
        num_inner_steps: 안쪽 회귀의 그래디언트 걸음 수
    """
    def __init__(self, model, inner_lr=0.01, meta_lr=0.001, num_inner_steps=5):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.num_inner_steps = num_inner_steps
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=meta_lr)
    
    def inner_loop(self, support_x, support_y, query_x, query_y):
        """
        과제 하나에 대해 안쪽 회귀 맞춤을 한다.
        
        반환값:
            query_loss: 맞춘 뒤 물음 집합에서의 로스
        """
        # 과제별 맞춤을 위해 모델 파라메터를 복사한다
        params = OrderedDict(self.model.named_parameters())
        
        # 안쪽 회귀: 받침 집합에 맞춘다
        for step in range(self.num_inner_steps):
            # 지금 파라메터로 피드포워드한다
            support_logits = self.model(support_x, params)
            support_loss = F.cross_entropy(support_logits, support_y)
            
            # 파라메터에 대한 그래디언트를 셈한다
            grads = torch.autograd.grad(
                support_loss,
                params.values(),
                create_graph=True  # 이차 그래디언트에 중요하다
            )
            
            # 그래디언트 내려가기로 파라메터를 고친다
            params = OrderedDict(
                (name, param - self.inner_lr * grad)
                for ((name, param), grad) in zip(params.items(), grads)
            )
        
        # 맞춘 파라메터로 물음 집합에서 평가한다
        query_logits = self.model(query_x, params)
        query_loss = F.cross_entropy(query_logits, query_y)
        
        return query_loss
    
    def meta_train_step(self, tasks):
        """
        과제 배치에 대해 메타 학습 걸음을 한 번 밟는다.
        
        인수:
            tasks: (support_x, support_y, query_x, query_y) 짝의 목록
        
        반환값:
            meta_loss: 과제에 걸친 평균 물음 로스
            meta_accuracy: 물음 집합에서의 평균 정확도
        """
        self.meta_optimizer.zero_grad()
        
        meta_loss = 0.0
        meta_accuracy = 0.0
        
        # 과제마다 안쪽 회귀를 돌고 그래디언트를 쌓는다
        for support_x, support_y, query_x, query_y in tasks:
            task_loss = self.inner_loop(support_x, support_y, query_x, query_y)
            meta_loss += task_loss
            
            # 이 과제의 정확도를 셈한다.
            # 맞추는 일 자체에는 그래디언트가 있어야 하므로 no_grad 안에서 돌 수
            # 없다. 대신 걸음마다 결과를 떼어 내어 메타 그래프가 자라지 않게 한다.
            params = OrderedDict(self.model.named_parameters())
            for step in range(self.num_inner_steps):
                support_logits = self.model(support_x, params)
                support_loss = F.cross_entropy(support_logits, support_y)
                grads = torch.autograd.grad(support_loss, params.values())
                params = OrderedDict(
                    (name, (param - self.inner_lr * grad).detach().requires_grad_(True))
                    for ((name, param), grad) in zip(params.items(), grads)
                )

            # 재는 일에는 그래디언트가 필요 없다
            with torch.no_grad():
                query_logits = self.model(query_x, params)
                predictions = torch.argmax(query_logits, dim=1)
                accuracy = (predictions == query_y).float().mean()
                meta_accuracy += accuracy
        
        # 과제들에 걸쳐 평균 낸다
        meta_loss = meta_loss / len(tasks)
        meta_accuracy = meta_accuracy / len(tasks)
        
        # 메타 갱신: 처음 파라메터를 고친다
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item(), meta_accuracy.item()
    
    def adapt(self, support_x, support_y, num_steps=None):
        """
        받침 집합이 주어졌을 때 모델을 새 과제에 맞춘다.
        
        반환값:
            adapted_params: 과제에 맞춘 파라메터
        """
        if num_steps is None:
            num_steps = self.num_inner_steps
        
        params = OrderedDict(self.model.named_parameters())
        
        for step in range(num_steps):
            support_logits = self.model(support_x, params)
            support_loss = F.cross_entropy(support_logits, support_y)
            
            grads = torch.autograd.grad(support_loss, params.values())
            # 다음 걸음에서 또 그래디언트를 구해야 하므로 requires_grad를 살려 둔다
            params = OrderedDict(
                (name, (param - self.inner_lr * grad).detach().requires_grad_(True))
                for ((name, param), grad) in zip(params.items(), grads)
            )
        
        return params
    
    def predict(self, support_x, support_y, query_x):
        """
        받침 집합에 맞춘 뒤 물음 집합에서 맞힌다.
        """
        self.model.eval()
        # 맞추는 일은 그래디언트를 쓰므로 no_grad 밖에서 한다
        adapted_params = self.adapt(support_x, support_y)
        # 맞힌 뒤의 피드포워드에는 그래디언트가 필요 없다
        with torch.no_grad():
            query_logits = self.model(query_x, adapted_params)
            predictions = torch.argmax(query_logits, dim=1)
        return predictions


# 사용 예
if __name__ == "__main__":
    # 모형 설정
    input_dim = 784  # 편 28x28 그림
    hidden_dim = 128
    output_dim = 5  # 5-클래스 분류
    
    # 모델과 MAML 학습기를 만든다
    model = SimpleClassifier(input_dim, hidden_dim, output_dim)
    maml = MAML(
        model,
        inner_lr=0.01,
        meta_lr=0.001,
        num_inner_steps=5
    )
    
    # 과제 배치를 만든다(5-클래스 5-예시)
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
    print(f"Predictions: {predictions}")
```

**출력:**

```
Meta-Loss: 1.6146, Meta-Accuracy: 0.2000
Predictions: tensor([1, 4, 2, 0, 2, 1, 0, 0, 4, 4, 0, 2, 4, 0, 4])
```

## 2. 논의

과제 $\tau$에서 한 번 적응한 웨이트는 $\theta'_\tau=\theta-\alpha\nabla_\theta L_{S_\tau}(\theta)$다. 메타 목표는 적응 후 쿼리 로스 $\mathbb E_\tau L_{Q_\tau}(\theta'_\tau)$를 줄이는 것이다. 내부 루프는 서포트 로스로 과제별 웨이트를 만들고, 외부 루프는 여러 과제의 쿼리 로스를 평균 내어 원래 초기 웨이트를 갱신한다. 서포트와 쿼리는 같은 과제의 클래스에 속하지만 표본은 분리한다.

inner_loop의 create_graph=True는 경사 계산 자체의 그래프를 보존한다. 따라서 외부 역전파는 내부 갱신을 통과하며 2차 미분 항도 포함한다. 모델 웨이트를 제자리에서 덮어쓰는 대신 OrderedDict의 새 텐서를 forward에 넘기는 것이 핵심이다. num_inner_steps가 늘면 그래프와 메모리 부담도 커진다.

정확도 계산은 별도의 적응 루프를 다시 수행하고 각 단계에서 detach한다. 이는 기록용이며 메타 로스의 미분 경로와 구별된다. predict는 평가 모드에서도 서포트 적응에 경사를 사용하므로 전체 호출을 no_grad로 감싸면 안 된다. model.eval() 자체는 경사를 끄지 않는다. 예제는 무작위 과제 한 배치를 갱신하며 실제 퓨샷 일반화 성능을 측정하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
예제의 input_dim=784, hidden_dim=128, output_dim=5일 때 SimpleClassifier의 파라메터 수를 구하라.

</div>

??? success "연습문제 1 풀이"
    첫 레이어는 784×128+128=100,480개, 둘째 레이어는 128×128+128=16,512개, 출력 레이어는 128×5+5=645개다. 합계는 117,637개이며 ReLU에는 학습 파라메터가 없다. 생성자에는 기본 차원값이 없으므로 사용 예의 설정을 기준으로 한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
각 과제의 입력과 라벨, 과제 배치에 어떤 검증이 필요한가?

</div>

??? success "연습문제 2 풀이"
    입력은 (표본 수, input_dim)의 실수 텐서이며 모델과 장치·자료형이 호환되어야 한다. 라벨은 표본 수와 같은 길이의 정수이며 0 이상 output_dim 미만이어야 한다. 서포트와 쿼리의 클래스 매핑을 일치시키고 두 집합의 표본을 분리한다. 빈 과제 목록은 평균에서 0으로 나누므로 허용하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
한 번의 내부 갱신에서 메타 경사와 1차 근사의 차이를 설명하라.

</div>

??? success "연습문제 3 풀이"
    정확한 메타 경사는 $(I-\alpha H_S)^\top\nabla_{\theta'}L_Q(\theta')$이며 $H_S$는 서포트 로스의 헤시안이다. 1차 MAML은 헤시안 항을 생략한다. 코드의 inner_loop는 create_graph=True로 정확한 경로를 유지하며, 정확도 계산에서 detach하는 것과 혼동해서는 안 된다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 풀이의 `ModuleList` 코드가 원래 모델에 바로 적용될 수 있는지 검토하라. 입력·출력 차원, 활성화, 생성자와 `forward`에서 필요한 수정을 설명하라.

</div>

??? success "연습문제 4 풀이"
    아래 코드는 깊이를 설정하는 아이디어를 위한 조각이다. 원래 모델의 생성자와 텐서 차원을 반영한 완성 구현은 아니다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `ModuleList`는 하위 모듈을 등록하지만 실행 순서나 활성화를 자동으로 정하지 않는다. 입력 레이어와 출력 레이어는 과제의 차원을 유지하고, 은닉 레이어 사이에 필요한 비선형성을 넣어야 한다. 선형 레이어만 반복하면 아핀 함수 하나로 합쳐진다. 원래 클래스에 없는 `num_layers` 인수를 그대로 넘길 수 없으므로 생성자와 `forward`를 함께 수정해야 한다.

## 정리하며

MAML은 적응 후의 쿼리 로스로 초기 웨이트를 학습한다. 함수형 내부 갱신의 미분 경로를 유지하고, 평가 시에도 서포트 적응에는 경사를 허용해야 한다.
