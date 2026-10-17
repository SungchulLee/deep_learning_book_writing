# 실전 학습 요령

러닝레이트 일정, 데이터 어그멘테이션, 정규화와 초기화는 서로 영향을 주는 학습 설정이다. 이 절은 자주 사용하는 선택지를 소개하고, 검증셋에서 각각의 효과와 결합 효과를 확인하는 방법을 설명한다.

## 1. 요령 1: 러닝레이트 스케줄링

러닝레이트 일정은 갱신 크기를 시간에 따라 조정한다. 비교할 만한 한 후보는 선형 워밍업과 코사인 감쇠의 결합이다.

$$
\eta(t)=\begin{cases}
\eta_{\max}t/T_w & 0\le t<T_w,\\
\eta_{\min}+\frac{\eta_{\max}-\eta_{\min}}{2}\left[1+\cos\left(\pi\frac{t-T_w}{T-T_w}\right)\right] & T_w\le t\le T.
\end{cases}
$$

여기서 $0<T_w<T$이고 t의 단위는 실제 옵티마이저 갱신 수 등으로 일관되게 정한다. 구현에서는 첫 호출의 인덱스와 끝값 도달 시점을 확인한다. 워밍업은 초기 큰 갱신을 완화할 수 있지만 발산을 반드시 막지는 않는다.

배치를 k배 키울 때 러닝레이트도 k배 키우는 선형 조정은 일부 SGD 학습에서 쓰는 경험칙이다. 모든 모델과 옵티마이저에 대한 수렴 보장은 아니며, 배치 증가로 에포크당 갱신 수가 줄어드는 효과도 고려한다. LAMB 같은 방법도 큰 배치의 정확도 유지나 속도 개선을 항상 보장하지 않는다.

최고 러닝레이트, 워밍업 길이, 최저 러닝레이트를 같은 검증 기준과 예산에서 비교한다. 고정 러닝레이트나 계단식 감쇠도 후보로 두고, 실제 처리량과 목표 성능까지의 시간을 함께 평가한다.

---

## 2. 요령 2: 정규화 쌓기

웨이트 감쇠, 드롭아웃, 레이블 스무딩과 데이터 어그멘테이션은 함께 사용할 수 있지만 효과가 단순히 더해지지는 않는다. 기준 모델에 하나씩 추가하는 실험과 결합 실험을 구분하고 정규화 강도를 검증셋에서 선택한다.

| 기법 | 주요 역할 | 점검할 조건 |
|---|---|---|
| 웨이트 감쇠 | 웨이트 크기에 영향을 주는 갱신 | 옵티마이저와 적용 파라메터 |
| 드롭아웃 | 학습 중 활성화에 무작위 마스크 적용 | 확률과 학습·평가 모드 |
| 레이블 스무딩 | 분류 목표의 분포 변경 | 정확도와 확률 보정의 개별 평가 |
| Mixup | 입력과 목표를 함께 보간 | 보간된 표본과 목표의 타당성 |

레이블 스무딩의 한 정의는 $y_i^{\mathrm{smooth}}=(1-\epsilon)y_i+\epsilon/K$이다. 과도한 확신을 줄이거나 확률 보정에 도움이 될 수 있지만 개선을 보장하지 않는다.

Mixup은 $\tilde x=\lambda x_i+(1-\lambda)x_j$, $\tilde y=\lambda y_i+(1-\lambda)y_j$를 사용하며 보통 $\lambda\sim\mathrm{Beta}(\alpha,\alpha)$로 정한다. 시계열이나 구조적 입력에서는 보간된 표본이 유효한지 먼저 판단해야 한다. 표본 간 혼합이 원래 의미나 시간 제약을 자동으로 보존하지는 않는다.

---

## 3. 요령 3: 웨이트 감쇠의 좋은 관행

모든 파라메터를 똑같이 정규화해야 하는 것은 아니다.

### 바이어스에는 감쇠를 걸지 않기

간단하지만 효과적인 요령이 있다. 웨이트 감쇠를 웨이트 행렬에만 적용하고 바이어스나 정규화 파라메터에는 걸지 않는 것이다.

```python
def get_parameter_groups(model, weight_decay=1e-2):
    """파라메터를 감쇠를 걸 무리와 걸지 않을 무리로 나눈다."""
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'norm' in name or 'bn' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    return [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
```

**적용 기준**: 바이어스와 정규화 레이어의 스케일·이동 파라메터를 감쇠에서 제외하는 설정은 유용한 비교 대상이다. 다만 모든 모델의 성능을 높이는 규칙은 아니다. 아래 코드는 이름과 차원을 이용해 그룹을 나누므로, 사용자 정의 파라메터가 의도한 그룹에 들어가는지도 확인해야 한다.

---

## 4. 요령 4: 잔차 뉴럴 넷트웍의 초기화

잔차 연결이 있는 뉴럴 넷트웍(요즘 구조에서 흔하다)에서는 다음과 같이 한다.

### 마지막 배치 정규화를 0으로 초기화하기

잔차 분기의 마지막 BatchNorm에서 $\gamma=0$, $\beta=0$으로 시작하면 그 분기의 초기 출력을 0으로 만들 수 있다. 항등 지름길과 덧셈만 있는 경우 블록은 초기 상태에서 입력을 그대로 전달한다. 투영 지름길이나 덧셈 뒤 활성화가 있으면 블록 전체가 모든 입력에서 항등 함수라는 설명은 수정해야 한다.

$$
\text{output} = x + \gamma \cdot F(x) \quad \xrightarrow{\gamma=0} \quad \text{output} = x
$$

이 초기화가 특정 잔차 모델의 학습에 도움이 될 수 있지만 구조와 학습 설정에 따라 효과가 달라진다. 정확도 향상의 고정된 범위를 가정하지 말고 해당 모델에서 비교한다.

---

## 5. 요령 5: 기법 결합하기 — 완전한 학습 설정

아래 사전은 후보 학습 설정을 모아 둔 예시이다. 그 자체로 각 기법을 구현하거나 금융·언어·시계열 과제에서의 효과를 입증하지 않는다. 실제 학습 루프에 적용하고 각 항목의 기능이 구현되어 있는지 확인해야 한다.

```python
training_config = {
    # 최적화기
    'optimizer': 'AdamW',
    'base_lr': 1e-3,
    'weight_decay': 0.01,
    'no_bias_decay': True,
    
    # 일정
    'warmup_epochs': 5,
    'total_epochs': 100,
    'scheduler': 'cosine',
    'min_lr': 1e-6,
    
    # 정칙화
    'label_smoothing': 0.1,
    'mixup_alpha': 0.2,
    'dropout': 0.1,
    
    # 초기화
    'zero_init_residual': True,
    
    # 학습
    'gradient_clip_norm': 1.0,
    'ema_decay': 0.999,  # 웨이트의 지수 이동 평균
}
```

### 결합 효과 검증

여러 기법의 효과는 서로 의존한다. 기준 설정, 한 기법만 바꾼 설정, 결합 설정을 같은 데이터 분할과 예산으로 비교한다. 특정 실험의 정확도 증가분을 다른 데이터나 모델에 그대로 더해 예상 성능으로 사용하지 않는다. 여러 시드의 변동과 계산 비용도 함께 기록한다.

---

## 6. 퀀트 금융에서의 고려 사항

시계열에서는 시간 순서를 보존한 학습·검증 분할과 예측 시점에 이용 가능한 피처를 먼저 확인한다. 입력 보간이나 레이블 스무딩을 적용했다는 사실만으로 국면 변화에 강하거나 확률이 잘 보정되었다고 할 수 없다.

EMA 웨이트는 학습 파라메터를 평활화하지만 예측이나 거래 회전율의 개선을 자동으로 보장하지 않는다. 예측 오차, 확률 보정과 후속 의사결정의 효과는 별도 기준으로 평가한다. 러닝레이트를 배치 크기에 비례해 키워도 수렴이 보장되지는 않는다.

---

## 7. 핵심 정리

1. 데이터 분할과 평가 기준을 먼저 정하고, 같은 예산에서 후보를 비교한다.
2. 각 기법의 개별 효과와 결합 효과를 구분한다.
3. 워밍업과 정규화의 필요성·강도를 해당 모델에서 검증한다.
4. 설정, 시드, 환경과 선택 절차를 기록한다. 시드 고정만으로 모든 재현성이 보장되지는 않는다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
표준 학습 설정을 다른 데이터셋으로 옮길 때 무엇을 다시 검증해야 하는가?

</div>

??? success "연습문제 1 풀이"
    러닝레이트와 일정, 배치 크기, 정규화, 어그멘테이션과 학습 기간을 새 데이터의 크기·피처·목표에 맞춰 비교해야 한다. 한 ImageNet 설정을 보편적인 표준값으로 취급하지 않는다. 검증 분할과 탐색 예산을 기록한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
워밍업의 목적과 적용 여부를 판단하는 방법을 설명하라.

</div>

??? success "연습문제 2 풀이"
    초반 갱신 크기를 낮춰 초기 불안정을 완화하려는 방법이다. 워밍업 유무와 길이를 같은 예산에서 비교하고 로스의 유한성, 검증 성능과 처리량을 확인한다. 모든 트랜스포머나 전이 학습에 반드시 필요하다고 단정하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 혼합 정밀도 학습 루프를 검토하고, 실제 프로젝트에 연결하려면 필요한 주변 정의와 호출 순서를 설명하라. 그래디언트 초기화, 스케일 해제, 클리핑과 스케줄러의 갱신 단위를 점검하라.

</div>

??? success "연습문제 3 풀이"
    아래 코드는 주변의 모델·옵티마이저·데이터 로더 등이 준비되어 있다는 가정의 조각이다. 첫 역전파 전에 기존 그래디언트가 없는지 확인하고, 클리핑은 unscale_ 뒤에 적용한다. 오버플로로 옵티마이저 갱신이 생략되는 경우 스케줄러의 의도한 진행 단위도 검토한다.
    ```python
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    for epoch in range(100):
        for x, y in loader:
            with torch.cuda.amp.autocast():
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        scheduler.step()
    ```

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
그래디언트 누적과 혼합 정밀도를 함께 쓸 때 연산 순서를 설명하라.

</div>

??? success "연습문제 4 풀이"
    누적 묶음 시작 전에 그래디언트를 초기화하고 각 작은 배치에서 정규화한 로스를 스케일해 역전파한다. 묶음이 끝나면 스케일을 해제하고 클리핑한 뒤 옵티마이저를 갱신한다. 스케일러와 스케줄러는 의도한 갱신 단위에 맞춰 진행하며 EMA는 실제 파라메터 갱신 뒤 적용한다.

## 정리하며

학습 기법은 고정된 성능 증가분을 쌓는 도구가 아니다. 데이터와 모델에 맞는 후보를 정하고 독립적인 평가, 동일한 예산과 반복 실험으로 효과를 확인한다. 코드 조각의 설정 사전과 실제 구현도 구분한다.

**참고 문헌**

1. He, T., et al. (2019). "Bag of Tricks for Image Classification with Convolutional Neural Networks." CVPR.
2. Goyal, P., et al. (2017). "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour." arXiv.
3. You, Y., et al. (2020). "Large Batch Optimization for Deep Learning: Training BERT in 76 Minutes." ICLR.
4. Zhang, H., et al. (2018). "mixup: Beyond Empirical Risk Minimization." ICLR.
5. Muller, R., et al. (2019). "When Does Label Smoothing Help?" NeurIPS.
