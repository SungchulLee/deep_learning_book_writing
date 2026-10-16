# 자기 증류의 지수 이동 평균 교사

EMA 교사는 학생의 과거 파라메터를 지수 가중 평균해 목표 모델을 구성한다. 변화가 완만한 목표를 만들 수 있지만 항상 더 정확하거나 붕괴·국소 최솟값을 막는 것은 아니다. EMA의 수학과 목표 경사 분리·버퍼·실행 모드를 함께 살펴본다.

## 1. 핵심 개념

- **EMA 계수**: 과거 교사를 얼마나 유지할지 정한다.
- **초기화**: 보통 학생 파라메터를 복사해 시작한다.
- **목표 경사 분리**: 학생 로스의 역전파로 교사를 직접 갱신하지 않는다.
- **파라메터와 버퍼**: BatchNorm 통계 등은 별도 폴리시가 필요하다.
- **학습 목적**: MoCo·BYOL·DINO 등은 EMA를 쓰지만 음성·로스·붕괴 대응 설계는 서로 다르다.

## 2. 수학적 정식화

### 갱신과 초기화

고정 계수 $0\le\tau<1$에서 다음과 같이 갱신한다고 하자.

$$
\theta_t^{\mathrm{EMA}}=\tau\theta_{t-1}^{\mathrm{EMA}}+(1-\tau)\theta_t^{\mathrm{student}},\qquad
\theta_0^{\mathrm{EMA}}=\theta_0^{\mathrm{student}}
$$

### 초기값을 포함한 전개

$$
\theta_t^{\mathrm{EMA}}=\tau^t\theta_0^{\mathrm{EMA}}+(1-\tau)\sum_{i=1}^{t}\tau^{t-i}\theta_i^{\mathrm{student}}
$$

초기값 항까지 포함하면 웨이트 합은 1이다. 초기 사본을 사용하는 이 규칙과 0에서 시작한 바이어스 보정 이동 평균은 다르다. 시간별 tau가 달라지면 웨이트는 단일 거듭제곱 대신 해당 계수들의 곱으로 정리된다.

## 3. 안정성 분석

고정 tau<1이고 학생 파라메터가 한 점으로 수렴하면 EMA도 같은 점으로 수렴한다. 학생이 계속 변하거나 분포가 이동하면 그 주장은 그대로 적용되지 않는다.

독립·동일 분산의 정상 잡음이라는 단순 가정에서 초기 효과가 사라진 뒤, 한 성분의 EMA 분산은 다음과 같다.

$$
\operatorname{Var}(\theta^{\mathrm{EMA}})=\frac{1-\tau}{1+\tau}\sigma^2
$$

이는 등비급수 $(1-\tau)^2\sum_{j\ge0}\tau^{2j}$에서 얻는다. tau=0.999이면 비율은 약 0.00050025, 즉 0.0500%다. 실제 연속 학습 계수는 강하게 상관되어 있어 이 분산식을 그대로 적용할 수 없고, 파라메터 분산 감소가 비선형 모델의 출력 분산·정확도 개선을 자동 보장하지도 않는다.

## 4. EMA 계수 고르기

tau가 클수록 과거 상태를 오래 유지하고 학생의 최근 변화를 늦게 반영한다. 단순한 유효 평균 길이는 약 $1/(1-\tau)$ 갱신으로 볼 수 있지만 데이터 분포·러닝레이트·배치·갱신 간격도 함께 영향을 준다.

tau를 러닝레이트/데이터셋 크기만으로 결정하는 보편적인 식은 없다. 초기·최종 계수와 스케줄을 검증하고, 학생 갱신 전후 어느 시점의 계수를 평균하는지도 기록한다. tau=1은 교사를 고정하고 tau=0은 현재 학생을 복사한다.

## 5. EMA 교사를 쓰는 학습 틀

### 두 가지 구조

```
Input
├─ Student Branch ──> Prediction ──> Loss₁
│                                     │
└─ EMA Teacher Branch ──> Features ──┘
                              │
                         EMA Update
```

### 순방향 알고리즘

```
1. Forward through student: z_s = Student(x)
2. Forward through EMA teacher: z_t = EMA_Teacher(x)
3. Compute loss: L = D(z_s, z_t)
4. Backward through student: ∇L_s = ∂L/∂θ_s
5. Update student: θ_s ← θ_s - α∇L_s
6. Update EMA teacher: θ_t ← τθ_t + (1-τ)θ_s
```

---

## 6. 목표 경사 분리

목표 계산을 no_grad 또는 detach로 분리하면 구현된 로스의 교사 경로를 역전파하지 않는다. 원래 로스 함수가 수학적으로 교사 계수에 무관하다는 뜻은 아니다.

$$
\mathcal L=\ell(f_s(x_1),\operatorname{stopgrad}(f_t(x_2)))
$$

교사는 EMA로만 갱신하더라도 학생의 학습 궤적에는 의존한다. stop-gradient나 EMA 하나만으로 표현 붕괴·불안정·국소 최솟값을 보편적으로 방지할 수는 없다. 음성·예측기·정규화·중심화 등 방법별 목표 설계를 함께 보아야 한다.

## 7. 구현의 세부

교사는 보통 경사·옵티마이저 상태 없이도 별도 계수 사본과 순전파 메모리를 사용한다. 전체 메모리는 학생·교사의 크기·dtype, 버퍼·활성·옵티마이저와 샤딩 방식에 따라 달라진다. 고정적인 2.1배 계산이나 2% EMA 오버헤드 규칙은 없다.

학생의 optimizer 갱신 뒤 교사를 EMA로 갱신하는 등 순서를 명시하고, BatchNorm 이동 통계·정수 버퍼의 복사·평균 폴리시를 따로 정한다. teacher.eval은 드롭아웃·BatchNorm 동작이고 no_grad는 경사 기록이므로 서로 대체하지 않는다.

작은 (1−tau) 갱신을 저정밀로 누적하면 반올림으로 사라질 수 있어 FP32 사본 등을 고려한다. 장치 이동·저장·복원 시 학생과 교사·버퍼의 대응도 유지해야 한다.

## 8. 더 나아간 EMA 변형

EMA 계수를 시간에 따라 바꾸거나 주기적으로 학생을 완전히 복사하는 변형을 구성할 수 있다. 예를 들어 다음 식은 시작 계수에서 최종 계수로 점근적으로 이동하는 하나의 설계다.

$$
\tau(t)=\tau_{\mathrm{end}}-(\tau_{\mathrm{end}}-\tau_{\mathrm{start}})e^{-t/\lambda},\quad\lambda>0
$$

이는 모든 자기 증류의 표준식이 아니다. 계수 범위·최종 시점·업데이트 간격과 실제 붕괴·품질 지표를 검증한다. 주기적 완전 복사는 누적 평균을 초기화하는 효과가 있어 연속 EMA와 다르다.

## 9. 이론적 이음

EMA는 최근 상태에 더 큰 웨이트를 준다. 반면 Polyak·Ruppert 평균은 대표적으로 과거 반복값에 균등 웨이트를 준다.

$$
\bar\theta_t=\frac1t\sum_{i=1}^{t}\theta_i
$$

고정 tau의 EMA와 이 평균은 같은 연산이 아니며 일반적으로 같은 분포로 수렴한다는 보장도 없다. 비선형 뉴럴 넷트웍에서 파라메터 평균은 예측 앙상블과도 다르다. 교사의 기대 로스가 학생보다 항상 작다는 부등식은 성립하지 않는다.

## 10. 계량 금융에서의 쓰임

EMA 목표를 시계열 표현 학습에 적용하는 실험을 설계할 수 있다. 그러나 시장 변화에서 느린 갱신은 잡음 완화와 적응 지연을 함께 만든다. 과거 자료로만 갱신하고 시간 분할에서 학생·교사의 오차와 변화 반응을 비교한다. 이 기법 자체가 위험 추정·포트폴리오 제약이나 수익성을 보장하지 않는다.

## 11. 관련 주제

- [지식 증류의 기초](distillation_basics.md)
- [MoCo](../contrastive_learning/contrastive_moco.md): EMA와 음성 큐
- [SimCLR](../contrastive_learning/contrastive_simclr.md): EMA가 없는 배치 내 대조 방식

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
EMA 교사의 초기값·갱신과 목표 경사 분리를 설명하라.

</div>

??? success "연습문제 1 풀이"
    학생 계수를 복사해 교사를 초기화하고 과거 교사와 현재 학생의 가중 평균으로 갱신한다. 목표 경로는 역전파에서 분리하지만 교사는 학생 궤적에 의존한다. EMA만으로 붕괴 방지가 보장되지는 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
EMA가 변화를 완만하게 만드는 이유와 성능 해석의 한계를 설명하라.

</div>

??? success "연습문제 2 풀이"
    과거 상태의 지수 가중 평균이라 최근 학생 갱신의 일부만 반영한다. 독립 정상 잡음의 분산 감소식은 연속 학습 계수의 상관을 생략한 모델이다. 목표 안정화와 적응 지연이 모두 가능하며 출력·정확도 개선은 별도 평가한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 EMA 갱신 조각이 처리하는 대상과 누락된 버퍼·정렬 조건을 설명하라.

</div>

??? success "연습문제 3 풀이"
    torch import와 같은 구조·순서·장치의 학생·교사 파라메터를 가정한다. zip은 길이가 다르면 조용히 짧은 쪽에서 끝나므로 이름·모양까지 대응을 확인해야 한다. 함수는 초기 복사·교사 동결·eval·버퍼 폴리시를 처리하지 않는다. no_grad 아래에서는 data 대신 텐서 자체의 제자리 갱신도 가능하며, decay 범위를 검증해야 한다.

    ```python
    @torch.no_grad()
    def update_ema(student, teacher, decay=0.999):
        for s_param, t_param in zip(student.parameters(), teacher.parameters()):
            t_param.data.mul_(decay).add_(s_param.data, alpha=1 - decay)
    ```

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
너무 빠르거나 느린 EMA의 절충과 스케줄 선택을 설명하라.

</div>

??? success "연습문제 4 풀이"
    tau가 낮으면 최근 학생을 빠르게 따라가고 높으면 과거를 오래 유지해 적응이 늦을 수 있다. 특정 수치가 항상 너무 높거나 낮다는 기준은 없으며 배치·러닝레이트·업데이트 횟수·분포 변화에 따라 달라진다. 스케줄과 갱신 시점을 검증하고 교사·학생을 함께 평가한다.

## 정리하며

EMA의 초기값 항·시간 웨이트·가정이 있는 분산 분석을 구분한다. 교사 경사 분리·모드·버퍼와 업데이트 순서를 맞추고, 평균이 항상 좋은 목표나 더 낮은 로스를 보장한다는 주장을 피한다.
