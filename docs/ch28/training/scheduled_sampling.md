# Scheduled sampling

Scheduled sampling은 학습 입력의 실제 토큰을 일부 모델 예측으로 대체하는 휴리스틱이다. [Bengio 등의 제안](https://proceedings.neurips.cc/paper/2015/hash/e995f98d56967d946471af29d7bf99f1-Abstract.html)은 teacher forcing에서 자체 생성 입력으로 점차 옮기는 학습 일정을 사용한다. 특정 시퀀스 작업에서 개선이 관찰되었지만 원래 최대우도와 같은 목적이나 보편적인 생성 품질 보장은 아니다.

## 1. 혼합 입력과 일정

단계 i의 실제 토큰 사용 확률을 ε_i라 하면 각 입력 위치에서 확률 ε_i로 실제 토큰, 1−ε_i로 모델 토큰을 택한다. 예측은 원래 실제 접두사가 아니라 그때까지 만든 **혼합 접두사**에 조건부다. 모델 토큰을 argmax로 고를지 범주형 표본으로 뽑을지도 별도 선택이다.

가능한 일정은 다음과 같다.

$$\epsilon_i=\max(\epsilon_{min},k^i),\qquad 0<k<1,$$

$$\epsilon_i=\max\left(\epsilon_{min},\frac{k}{k+\exp(i/k)}\right),\qquad k>0.$$

두 식의 k는 같은 의미의 상수가 아니다. ε_min은 0~1 범위이고 i가 스텝·에포크 중 무엇인지 명시한다. 역 sigmoid 식은 i=0에서도 k/(k+1)이라 정확히 1에서 시작하지 않는다. 큰 i의 지수 계산에는 넘침도 고려한다. 혼합 비율은 시퀀스 전체·시점·표본별로 정할 수 있으며 그 선택이 계산과 학습 목적에 영향을 준다.

## 2. 예시 구현의 동작

```python
def scheduled_sampling_step(model, x, epsilon):
    batch_size, seq_len = x.shape
    input_tokens = x[:, 0:1]  # 참값 시작 토큰에서 시작한다
    
    for t in range(1, seq_len):
        logits = model(input_tokens)
        predicted = logits[:, -1].argmax(dim=-1, keepdim=True)
        
        # 동전 던지기: 참값을 쓸지 헤아린 값을 쓸지
        use_gt = (torch.rand(batch_size, 1) < epsilon).to(x.device)
        next_token = torch.where(use_gt, x[:, t:t+1], predicted)
        input_tokens = torch.cat([input_tokens, next_token], dim=1)
    
    # 온 차례에 대해 로스를 셈한다
    full_logits = model(input_tokens[:, :-1])
    return F.cross_entropy(full_logits.reshape(-1, vocab_size),
                          x[:, 1:].reshape(-1))
```

x는 `(B,T)` 정수 토큰이고 model은 모든 입력 위치의 logits `(B,L,V)`를 반환해야 한다. torch·F import와 외부 vocab_size 정의가 필요하다. ε는 0~1, T≥2이며 모델과 입력의 장치·어휘가 맞아야 한다. 앞 절의 마지막 위치만 출력하는 문자 모델에는 그대로 적용할 수 없다.

코드는 각 시점·표본별로 실제 토큰 사용 여부를 뽑고 모델 토큰은 **argmax**로 고른다. 범주형 표본 추출이 아니다. 매번 전체 혼합 접두사를 다시 계산하고 마지막에 한 번 더 forward하여 원래 정답 x[:,1:]과 교차엔트로피를 계산한다. 인과적 마스크나 재귀 방향을 모델이 보장해야 하며 이 함수 자체는 미래 정보 누출을 막지 않는다.

argmax와 정수 입력 연결을 통과해 모델 선택 과정으로 미분하지 않는다. 최종 forward에는 그래디언트가 흐르지만 혼합 접두사가 θ에 의존한다는 효과까지 미분한 전체 목적의 그래디언트는 아니다. 모델을 train·eval 중 어느 모드로 두는지도 호출부가 결정하며 dropout 등이 토큰 선택과 최종 계산에 영향을 줄 수 있다. ε의 일정·optimizer 단계·패딩 처리는 이 함수에 없다.

## 3. 최대우도와의 차이

원래 관측 로그우도는 실제 접두사에서 실제 다음 토큰을 평가한다. Scheduled sampling은 바뀐 접두사에도 원래 타깃을 붙이므로 다른 통계적 목적을 만든다. 그 타깃이 변경된 문맥의 올바른 조건부 표본이라는 보장은 없다. 이는 로그우도의 하한 여부만으로 설명할 문제가 아니며, 모델 의존 입력 선택의 경사 생략과 목적의 통계적 일관성도 구분해야 한다.

[Huszár의 분석](https://arxiv.org/abs/1511.05101)은 이 방식이 일반적으로 일관된 분포 추정으로 이어지지 않을 수 있음을 지적한다. 학습 데이터가 늘어도 원하는 참 결합 분포를 회복한다는 보장이 없다는 뜻이며 유한 작업에서의 경험적 유용성 여부와는 다른 주장이다.

## 4. 대안과 공통 평가

시퀀스 수준 리워드 최적화는 어떤 결과를 선호할지 목적을 바꾼다. REINFORCE는 조건을 갖춘 표본·그래디언트 추정에서 비바이어스일 수 있지만 높은 분산과 리워드 설계 문제가 남는다. 모든 강화학습 절차가 자동으로 비바이어스인 것은 아니다.

[Professor Forcing](https://arxiv.org/abs/1610.09038)은 teacher-forced 경로와 자유 생성 경로의 은닉 동작을 적대적으로 맞추는 접근이다. 이 또한 최적화·평가가 필요하고 생성 성공을 보장하지 않는다. Gumbel-softmax 같은 연속 완화나 straight-through 추정은 이산 선택의 경사를 다루는 다른 문제에 관여하며 원래 목적의 일관성을 자동 회복하지 않는다.

빔 서치는 학습 목표를 바꾸지 않는 추론 탐색이다. 혼합 학습과 같은 종류의 해결책으로 순위를 매기지 않는다. 공통 검증 우도와 실제 생성 목적·비용을 함께 측정하고 각 방법의 학습·디코딩 폴리시를 명시한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
Exposure bias를 단순히 “틀린 토큰이 반드시 계속 늘어나는 현상”으로 설명할 수 없는 이유는 무엇인가?

</div>

??? success "풀이"

    모델이 생성한 접두사는 관측 접두사와 다른 분포를 만들 수 있지만 오차 전파의 크기·회복 가능성은 모델·자료·디코딩에 달려 있다. 참 조건부를 정확히 학습한 모델의 조상 샘플링은 참 분포를 따르므로 teacher forcing 자체가 필연적으로 누적 오류를 만든다고 할 수 없다. 실제 검증 로스와 길이별 자유 생성 성능을 함께 봐야 한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
제공 코드의 혼합 확률·토큰 선택·일정을 구분하라.

</div>

??? success "풀이"

    ε는 각 시점·표본에서 실제 토큰을 사용할 확률이다. 모델 토큰은 argmax이며 매번 혼합 접두사에 조건부로 예측한다. 함수에는 ε를 바꾸는 일정이 없어 호출부가 스텝·에포크별 값을 정해야 한다. ε=1은 입력 관점에서 teacher forcing, ε=0은 첫 토큰 뒤 모두 greedy 모델 입력이다. 이때도 학습 로스 타깃은 원래 실제 시퀀스라 자유 생성 분포의 우도를 그대로 최대화하는 것은 아니다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
원래 최대우도와 다른 목적이라는 주장과 그래디언트 바이어스를 구분하라.

</div>

??? success "풀이"

    변경된 접두사·원래 타깃의 결합은 관측 결합 분포와 달라 원래 최대우도 목적이 아니다. 또한 입력 선택이 모델에 의존하지만 argmax·샘플링 경로를 고정 입력으로 처리하면 그 의존성을 포함한 그래디언트와도 다르다. 두 문제는 관련되지만 동일하지 않다. 단순히 학습 중 분포가 바뀌거나 ELBO가 아니라는 이유만으로 모든 방법이 잘못된 것은 아니며, 목표·추정량·통계적 일관성을 각각 분석해야 한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
Scheduled sampling·빔 서치·시퀀스 리워드·Professor Forcing의 조절 대상을 비교하라.

</div>

??? success "풀이"

    Scheduled sampling은 학습 입력 분포를, 빔 서치는 추론 탐색을, 시퀀스 리워드 학습은 최적화 목적을, Professor Forcing은 두 실행 모드의 은닉 동작을 조절한다. 조절 대상이 다르므로 어느 하나가 항상 가장 단순하고 효과적이거나 가장 널리 쓰인다는 결론은 낼 수 없다. 같은 자료·평가 폴리시·튜닝 예산에서 검증 우도와 목적별 생성 지표·비용을 비교한다.



## 정리하며

Scheduled sampling은 혼합 접두사에서 원래 타깃을 학습하는 방식이다. 혼합 일정·선택 단위·argmax 또는 샘플링·경사 처리를 명시하고 최대우도와의 차이를 이해해야 한다. 생성 개선과 분포 추정의 일관성은 서로 다른 기준으로 평가한다.
