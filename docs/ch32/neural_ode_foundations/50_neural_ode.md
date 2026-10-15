# 오일러 블록으로 읽는 신경 ODE

[Neural ODE](https://arxiv.org/abs/1806.07366)는 신경망으로 은닉상태의 시간 미분을 정의한다. 이 짧은 예제는 그 아이디어를 고정 스텝 오일러 반복으로 보여 준다. 적응형 솔버나 adjoint를 구현한 코드는 아니며, 반복의 끝점 처리에도 주의가 필요하다.

## 1. 코드와 보존된 출력

```python
#!/usr/bin/env python3
'''
신경 상미분 방정식 - Neural Ordinary Differential Equations
논문: "Neural Ordinary Differential Equations" (2018)
2018년 NeurIPS 최우수 논문상을 받았다
핵심: 이어진 깊이 모델, 기억을 아끼는 뒷걸음 퍼뜨리기
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class ODEFunc(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.Tanh(),
            nn.Linear(64, dim),
        )
    
    def forward(self, t, y):
        return self.net(y)

class ODEBlock(nn.Module):
    def __init__(self, odefunc, method='euler', step_size=0.1):
        super().__init__()
        self.odefunc = odefunc
        self.method = method
        self.step_size = step_size
    
    def forward(self, x):
        # 단순한 오일러 적분(보여 주기용)
        t = 0
        t_end = 1
        
        while t < t_end:
            dx = self.odefunc(t, x)
            x = x + self.step_size * dx
            t += self.step_size
        
        return x

class NeuralODE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=64, num_classes=10):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.odeblock = ODEBlock(ODEFunc(hidden_dim))
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        x = self.odeblock(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    model = NeuralODE()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    x = torch.randn(32, 784)
    print(f"Input: {x.shape}, Output: {model(x).shape}")
```

```
Parameters: 59,210
Input: torch.Size([32, 784]), Output: torch.Size([32, 10])
```

출력은 임의 입력의 모양과 파라미터 수를 확인하는 데모다. 데이터셋·학습·분류 정확도나 수치 해의 오차를 검증하는 실험은 없다.

## 2. 세 모듈의 역할

입력을 배치별로 펼쳐 encoder가 784차원에서 64차원으로 바꾼다. ODEFunc는 64→64→64의 Tanh MLP로 순간 변화량을 출력한다. 시간 t를 인수로 받지만 사용하지 않으므로 자율 벡터장이다.

ODEBlock은 매 스텝 x←x+h fθ(x)를 계산한다. 같은 ODEFunc를 반복 호출하므로 스텝마다 다른 파라미터를 가진 MLP를 쌓는 것과 다르다. 마지막 classifier는 64차원 상태를 10개 logits로 바꾼다. 기본 배치 입력 32×784의 출력은 32×10이다.

encoder 파라미터는 50,240개, ODEFunc는 8,320개, classifier는 650개로 합계 59,210개다. 오일러 스텝 수를 바꾸어도 같은 벡터장을 재사용하는 파라미터 수는 바뀌지 않는다.

## 3. method 인수와 실제 솔버

method를 저장하지만 forward에서 검사하거나 사용하지 않는다. 따라서 method='rk4'로 바꾸어도 여전히 오일러 업데이트를 한다. step_size만 고정해 적용하며 국소 오차 추정·수락·거절·적응 스텝은 없다.

파라미터 gradient는 반복한 PyTorch 연산 그래프로 직접 계산한다. adjoint 역적분을 하지 않아 스텝 수에 무관한 메모리 절약을 구현했다고 설명할 수 없다. 연속 ODE의 기울기와 유한 오일러 반복의 기울기도 수치적으로 다를 수 있다.

## 4. 반복 끝점과 유효 입력

t를 Python 실수로 누적하고 while t<1로 검사한다. **기본 0.1의 부동소수점 누적에서는 10번 뒤 t가 1보다 조금 작아 추가 스텝이 생길 수 있어**, 의도한 [0,1] 적분과 결과가 달라진다. 일반적으로 1이 h로 나누어떨어지지 않아도 마지막 스텝을 줄이지 않아 구간을 넘는다.

step_size가 0이나 음수이면 종료하지 않을 수 있다. 유한한 양수 여부와 총 스텝·마지막 부분 스텝을 명시적으로 설계해야 한다. 오일러의 안정성 조건도 스텝이 작다는 말만으로 대신할 수 없다. 자세한 내용은 [수치 솔버](numerical_solvers.md)를 참고한다.

입력은 첫 축이 배치이고 펼친 특징 수가 input_dim과 같아야 한다. 모델 가중치와 입력의 dtype·device가 호환되어야 하며 view는 메모리 배치에 따라 비연속 텐서에서 실패할 수도 있다. ODEFunc의 변화량은 입력 은닉상태와 같은 모양이어야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
기본 ODEFunc와 전체 NeuralODE의 파라미터 수를 계산하라.

</div>

??? success "연습문제 1 풀이"

    dim=64에서 각 Linear(64,64)는 64×64+64=4,160개이므로 ODEFunc는 8,320개다. encoder는 784×64+64=50,240개, classifier는 64×10+10=650개로 전체는 59,210개다. 파라미터를 공유하므로 오일러 반복 횟수를 곱하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
forward에 필요한 입력 조건과 step_size 조건을 설계하라.

</div>

??? success "연습문제 2 풀이"

    배치 축을 갖는 텐서인지, 펼친 특징 수가 input_dim인지, dtype·device가 가중치와 맞는지 확인한다. float32만 강제하기보다 모델의 실제 정밀도에 맞는 조건을 사용한다. step_size는 유한한 양수여야 하고 종료 시간·반복 수·마지막 부분 스텝을 명시해야 한다. method가 실제 지원되지 않으면 미지원 값을 거절하는 인터페이스가 일관된다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 코드에 특유한 두 가지 동작 불일치를 설명하라.

</div>

??? success "연습문제 3 풀이"

    method 인수가 사용되지 않아 rk4를 지정해도 오일러다. 또한 실수 t 누적의 while 반복은 기본 0.1에서도 추가 스텝을 만들 수 있고, 나누어떨어지지 않는 구간의 끝점을 넘는다. 스텝 목록과 최종 적분 시간을 확인하는 것이 일반적인 학습 성능 문제를 논하기 전의 점검이다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 ModuleList 골격을 ODEFunc의 깊이 옵션으로 완성하려면 무엇을 정의해야 하는가? 이것은 적분 스텝 수와 같은가?

</div>

??? success "연습문제 4 풀이"

    YourBlock, num_layers와 dim이 정의되지 않아 아래는 실행 가능한 확장 코드가 아니다. 입력 dim→hidden, 내부 hidden→hidden과 출력 hidden→dim의 차원·활성화 배치를 정하고 ModuleList의 순방향 순회를 정의해야 한다. num_layers 인수도 기존 생성자에는 없다. 벡터장 MLP의 깊이는 한 번의 f 계산 구조이고, 적분 스텝 수는 그 f를 몇 번 호출하는지이므로 서로 다르다. 여러 스텝에 별도 MLP를 만들면 원래의 파라미터 공유도 바뀐다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


## 정리하며

이 예제는 파라미터를 공유한 벡터장의 고정 오일러 반복이다. 미사용 method, 부동소수점 끝점과 양의 스텝 조건을 확인하고, 벡터장 깊이·적분 스텝 수·adjoint 메모리 절약을 구분한다.
