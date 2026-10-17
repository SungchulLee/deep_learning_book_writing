# GRU와 LSTM 비교

LSTM과 GRU는 게이트로 상태를 조절하는 순환 뉴럴 넷트웍이다. LSTM은 은닉·셀 상태를 구분하고 GRU는 상태 하나를 사용한다. 이 절에서는 구조적 차이와 특정 실험에서만 판단할 수 있는 속도·성능 차이를 나누어 비교한다.

---

## 1. 구조 비교

### 구조 개관

| 항목 | LSTM | GRU |
|--------|------|-----|
| 상태 | 2개 (은닉 상태 $h$, 셀 상태 $c$) | 1개 (은닉 상태 $h$) |
| 문 | 3개 (망각, 입력, 출력) | 2개 (갱신, 재설정) |
| 문의 결합 | 서로 독립 | 묶임 (갱신 = 1 − 망각) |
| 출력 거르기 | 있음 (출력 게이트) | 없음 |
| 나온 해 | 1997 | 2014 |

### 수식 나란히 놓기

**LSTM:**

$$f_t = \sigma(W_f[h_{t-1}, x_t] + b_f) \quad \text{(forget gate)}$$

$$i_t = \sigma(W_i[h_{t-1}, x_t] + b_i) \quad \text{(input gate)}$$

$$o_t = \sigma(W_o[h_{t-1}, x_t] + b_o) \quad \text{(output gate)}$$

$$\tilde{c}_t = \tanh(W_c[h_{t-1}, x_t] + b_c) \quad \text{(candidate)}$$

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \quad \text{(cell update)}$$

$$h_t = o_t \odot \tanh(c_t) \quad \text{(hidden state)}$$

**GRU:**

$$z_t = \sigma(W_z[h_{t-1}, x_t] + b_z) \quad \text{(update gate)}$$

$$r_t = \sigma(W_r[h_{t-1}, x_t] + b_r) \quad \text{(reset gate)}$$

$$\tilde{h}_t = \tanh(W_h[r_t \odot h_{t-1}, x_t] + b_h) \quad \text{(candidate)}$$

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t \quad \text{(state update)}$$

### 근본적인 설계의 차이

핵심적인 철학의 차이는 기억을 다루는 방식에 있다.

**LSTM: 역할을 나누기**

- 셀 상태 $c_t$: 장기 기억 저장소
- 은닉 상태 $h_t$: 작업 기억이자 출력
- 출력 게이트: 무엇을 드러낼지 다스린다
- 이렇게 나누면 정보를 곧바로 쓰지 않고도 담아 둘 수 있다

**GRU: 하나로 합친 상태**

- 은닉 상태 하나가 두 구실을 모두 한다
- 더 간단하지만 덜 유연하다
- 담아 둔 것이 곧 드러나는 것이다

---

## 2. 두 구조의 문 대응시키기

### 구조적 대응

| GRU | LSTM | 참고 |
|-----|------|-------|
| 업데이트 게이트 $z$ | 망각 $f$ + 입력 $i$ | GRU는 둘을 묶는다: $i = z$, $f = 1-z$ |
| 리셋 게이트 $r$ | (부분적으로) 출력 게이트 | 둘 다 쓰기 전에 상태를 거른다 |
| — | 출력 게이트 $o$ | GRU에는 명시적인 출력 게이트가 없다 |
| — | 셀 상태 $C$ | GRU는 은닉 상태만 쓴다 |

### 표현력의 트레이드오프

**LSTM의 묶이지 않은 문은 다음을 허용한다.**

- 많이 지키고 조금 더하기: $f = 0.9, i = 0.1$ → 90%를 지키고 10%를 더한다
- 많이 지키고 많이 더하기: $f = 0.9, i = 0.9$ → 90%를 지키고 90%를 더한다
- 조금 지키고 많이 더하기: $f = 0.1, i = 0.9$ → 10%를 지키고 90%를 더한다
- 상태에 걸리는 전체 웨이트는 $f + i$이며 1보다 작거나 같거나 클 수 있다

**GRU의 묶인 문은 다음을 강제한다.**

- $z = 0.1$ → 90%를 지키고 10%를 더한다
- $z = 0.5$ → 50%를 지키고 50%를 더한다
- $z = 0.9$ → 10%를 지키고 90%를 더한다
- 상태에 걸리는 전체 웨이트는 언제나 $(1-z) + z = 1$이다

GRU의 이 볼록 결합 제약은 암묵적인 규제가 되지만 유연함을 제한한다.

---

## 3. 파라메터 수 분석

### 이론적인 계산

입력 차원이 $d$이고 은닉 차원이 $n$인 LSTM이나 GRU에 대해 다음과 같다.

| 구조 | 파라메터 | 상대적 크기 |
|--------------|------------|----------|
| 기본 RNN | $n^2 + nd + n$ | 1배 |
| GRU | $3(n^2 + nd + n)$ | 3배 |
| LSTM | $4(n^2 + nd + n)$ | 4배 |

같은 입력·은닉 차원과 레이어·방향 수, 같은 바이어스 규약에서 GRU 순환 부분은 LSTM보다 25% 적은 파라메터를 사용한다. 위 표는 묶음마다 바이어스 하나인 식이고 아래 PyTorch 계산은 두 바이어스이므로 절대 개수는 다르다. 임베딩이나 출력 레이어를 포함한 전체 모델에서는 이 비율이 달라질 수 있다.

```python
import torch
import torch.nn as nn

def parameter_scaling_analysis():
    """모델의 크기에 따라 파라메터 수가 어떻게 늘어나는지 분석한다."""
    configs = [
        (64, 128), (128, 256), (256, 512), (512, 1024), (1024, 2048),
    ]
    
    print("Parameter Scaling Analysis")
    print("=" * 70)
    print(f"{'Input':<8} {'Hidden':<8} {'LSTM':<15} {'GRU':<15} {'Reduction':<10}")
    print("-" * 70)
    
    for input_size, hidden_size in configs:
        lstm = nn.LSTM(input_size, hidden_size)
        gru = nn.GRU(input_size, hidden_size)
        lstm_p = sum(p.numel() for p in lstm.parameters())
        gru_p = sum(p.numel() for p in gru.parameters())
        print(f"{input_size:<8} {hidden_size:<8} {lstm_p:<15,} {gru_p:<15,} {(1 - gru_p/lstm_p)*100:.1f}%")

# parameter_scaling_analysis()
```

### 메모리 사용량

아래 함수는 몇 가지 상태·게이트 배열의 크기를 float32로 가정해 합산한 추정치다. 실제 런타임 메모리를 측정하지 않으며 파라메터, 그래디언트와 옵티마이저 상태, 임시 작업 공간도 포함하지 않는다. “33.3% 절감”을 전체 메모리 사용량으로 읽으면 안 된다.

파라메터 말고도 학습 중의 엑티베이션 메모리를 살펴야 한다.

```python
def memory_footprint_analysis(batch_size, seq_length, hidden_size):
    """순전파와 역전파 중의 메모리 사용량을 견준다."""
    # LSTM이 담는 것: 시각마다 h_t, c_t, 그리고 모든 문의 엑티베이션
    lstm_per_step = 6 * hidden_size  # h, c, f, i, o, c_tilde
    lstm_total = batch_size * seq_length * lstm_per_step * 4  # float32
    
    # GRU가 담는 것: h_t와 문의 엑티베이션
    gru_per_step = 4 * hidden_size  # h, z, r, h_tilde
    gru_total = batch_size * seq_length * gru_per_step * 4
    
    print(f"Activation Memory (batch={batch_size}, seq={seq_length}, hidden={hidden_size}):")
    print(f"  LSTM: {lstm_total / 1024**2:.2f} MB")
    print(f"  GRU:  {gru_total / 1024**2:.2f} MB")
    print(f"  GRU saves: {(lstm_total - gru_total) / lstm_total * 100:.1f}%")

# memory_footprint_analysis(batch_size=32, seq_length=512, hidden_size=512)
```

---

## 4. 그래디언트 흐름 비교

### 직접 경로와 전체 미분

LSTM에는 망각 게이트로 조절되는 직접 셀 경로가 있다. 여기서 쓰는 GRU 규약에는 $\operatorname{diag}(1-z_t)$의 직접 보존 경로가 있다. GRU의 전체 미분에는 후보와 z 자체의 상태 의존성도 포함된다. PyTorch 내장 GRU는 z를 이전 상태의 계수로 쓰므로, 본문 수식과 이름만으로 대응하면 안 된다.

아래 실험은 학습하지 않은 모델의 마지막 출력 합에 대한 첫·끝 입력 그래디언트 비율을 측정한다. 여러 무작위 시행의 평균·표준편차이며 기억 과제의 성능이나 신뢰 구간이 아니다. 모델별 입력과 초기값을 맞추지 않았고 시드도 고정하지 않았으므로 특정 수치를 일반화할 수 없다.

```python
def compare_gradient_flow(seq_lengths=[50, 100, 200, 500, 1000]):
    """LSTM과 GRU의 그래디언트 흐름을 실험으로 견준다."""
    import numpy as np
    
    input_size = 64
    hidden_size = 128
    num_trials = 20
    
    results = {'lstm': {}, 'gru': {}}
    
    for seq_len in seq_lengths:
        for name, model_class in [('lstm', nn.LSTM), ('gru', nn.GRU)]:
            gradients = []
            
            for _ in range(num_trials):
                model = model_class(input_size, hidden_size, batch_first=True)
                x = torch.randn(1, seq_len, input_size, requires_grad=True)
                
                outputs, _ = model(x)
                loss = outputs[0, -1, :].sum()
                loss.backward()
                
                first_grad = x.grad[0, 0, :].norm().item()
                last_grad = x.grad[0, -1, :].norm().item()
                gradients.append(first_grad / (last_grad + 1e-10))
            
            results[name][seq_len] = {
                'mean': np.mean(gradients),
                'std': np.std(gradients)
            }
    
    print("Gradient Retention (first/last timestep ratio)")
    print("=" * 70)
    for seq_len in seq_lengths:
        lstm_r = results['lstm'][seq_len]
        gru_r = results['gru'][seq_len]
        print(f"  Seq {seq_len:>5}: LSTM={lstm_r['mean']:.4f}±{lstm_r['std']:.4f}  "
              f"GRU={gru_r['mean']:.4f}±{gru_r['std']:.4f}")

# compare_gradient_flow()
```

---

## 5. 기억 용량: 출력 게이트의 이점

LSTM의 출력 게이트는 특별한 능력을 준다. **드러내지 않고 정보를 담아 두는 것**이다.

**늦은 회상을 위한 LSTM의 전략:**

- 0스텝: $i_t = 1, f_t = 0$ → 세포에 담는다
- 1~99스텝: $f_t = 1, o_t = 0$ → 지키되 감춘다
- 100스텝: $o_t = 1$ → 담아 둔 정보를 드러낸다

**GRU의 어려움:**

- 명시적인 출력 게이트는 없다.
- 다만 상태의 일부 성분을 이후 출력 레이어에서 사용하지 않는 등 별도 표현을 학습할 수 있다.
- 상태 크기만으로 필요한 기억 과제의 우열을 확정할 수는 없다.

그래서 골라서 읽고 쓰는 접근이 필요한 복잡한 기억 방식을 LSTM이 다룰 수 있을 때가 있다.

### 정보 병목

| 항목 | LSTM | GRU |
|--------|------|-----|
| 상태의 용량 | 값 $2n$개 ($c_t$과 $h_t$) | 값 $n$개 ($h_t$만) |
| 저장과 출력의 분리 | 있음 (세포 ≠ 은닉 상태) | 없음 (상태 하나) |
| 상태 값 개수 | 두 벡터의 관계도 고려해야 한다 | 한 벡터를 사용한다 |

---

## 6. 과제별 성능

언어 모델, 번역, 분류와 시계열 등 과제 이름만으로 LSTM·GRU의 우열을 정할 수 없다. 같은 이름의 과제라도 데이터 크기, 길이, 전처리와 출력 목표가 다르다.

비교 연구의 결과는 사용한 데이터와 하이퍼파라미터 범위에서 읽는다. 현재 과제에서는 같은 검증 분할과 학습 예산을 두고 각 구조를 조정한다. 같은 은닉 크기로 구조를 비교하는 것과 같은 파라메터·시간 예산에서 비교하는 것은 서로 다른 질문이다.

## 7. 실무에서의 판단 기준

### 판단 흐름도

아래 흐름도는 휴리스틱 예시이며 검증된 선택 법칙이 아니다. 길이 500이나 표본 1만 개는 보편적 경계가 아니고, GRU의 속도·과적합 우위도 측정 없이 보장할 수 없다.

```
                                START
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │  Is this a prototype/MVP?   │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                   YES                          NO
                    │                           │
                    ▼                           ▼
              Use GRU              ┌─────────────────────────┐
          (faster iteration)      │  Sequence length > 500? │
                                  └───────────┬─────────────┘
                                              │
                                ┌─────────────┴─────────────┐
                                │                           │
                               YES                          NO
                                │                           │
                                ▼                           ▼
                    ┌───────────────────┐     ┌─────────────────────────┐
                    │   Lean LSTM       │     │  Dataset < 10k samples? │
                    │ (better long-term)│     └───────────┬─────────────┘
                    └───────────────────┘                 │
                                              ┌───────────┴───────────┐
                                              │                       │
                                             YES                      NO
                                              │                       │
                                              ▼                       ▼
                                        Use GRU           Try both, pick based
                                    (less overfitting)    on validation metrics
```

### 설정 권장값

함수의 숫자는 실험의 출발점일 뿐이다. `task_type`은 함수 안에서 사용되지 않아 과제별 판단을 수행하지 않는다. 주어진 길이·크기만으로 적절한 은닉 크기와 드롭아웃을 결정할 수 없다.

```python
def get_recommended_config(task_type, seq_length, dataset_size, 
                           inference_speed_critical=False):
    """과제의 성격에 따라 권장 구조를 얻는다."""
    recommendations = {
        'arch': None, 'hidden_size': None,
        'num_layers': None, 'dropout': None, 'reasoning': []
    }
    
    if seq_length > 500:
        recommendations['arch'] = 'LSTM'
        recommendations['reasoning'].append(
            f"Long sequences ({seq_length}) favor LSTM's cell state")
    elif dataset_size < 10000:
        recommendations['arch'] = 'GRU'
        recommendations['reasoning'].append(
            f"Small dataset ({dataset_size}) - GRU less prone to overfitting")
    elif inference_speed_critical:
        recommendations['arch'] = 'GRU'
        recommendations['reasoning'].append("Speed critical - GRU is 15-25% faster")
    else:
        recommendations['arch'] = 'GRU'
        recommendations['reasoning'].append(
            "Default to GRU, switch to LSTM if performance plateaus")
    
    recommendations['hidden_size'] = (128 if dataset_size < 5000 
                                       else 256 if dataset_size < 50000 else 512)
    recommendations['num_layers'] = 2 if seq_length > 200 else 1
    recommendations['dropout'] = (0.5 if dataset_size < 10000 
                                   else 0.3 if dataset_size < 100000 else 0.1)
    
    return recommendations
```

---

## 8. 혼합형 접근

### 쌓아 만든 혼합 구조

GRU 위에 LSTM을 쌓는다고 아래 레이어가 지역 피처, 위 레이어가 장기 기억만 담당한다고 보장되지는 않는다. 두 레이어 모두 순환 상태로 전체 입력에 의존할 수 있다.

```python
class HybridStackedRNN(nn.Module):
    """
    혼합 구조: 피처 추출에는 GRU, 기억에는 LSTM.
    
    근거:
    - 아래 레이어는 지역 피처를 뽑는다 (GRU로 충분하고 더 빠르다)
    - 위 레이어는 먼 거리 의존을 다룬다 (LSTM이 유리하다)
    """
    
    def __init__(self, input_size, hidden_size, num_gru_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_gru_layers,
                          batch_first=True,
                          dropout=dropout if num_gru_layers > 1 else 0)
        self.lstm = nn.LSTM(hidden_size, hidden_size, 1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        gru_out = self.dropout(gru_out)
        lstm_out, (h_n, c_n) = self.lstm(gru_out)
        return lstm_out, h_n
```

### 양방향 혼합

아래 클래스에는 encode만 구현되어 있다. 디코더 모듈은 등록하지만 호출하지 않으며 디코더 초기 셀 상태, 입력 토큰과 생성 루프도 정의하지 않는다. 완성된 번역·생성 모델로 해석하면 안 된다.

```python
class BidirectionalHybrid(nn.Module):
    """
    문맥에는 양방향 GRU, 생성에는 단방향 LSTM.
    
    쓰임새: 인코더는 양방향 문맥이 필요하고 디코더는 인과적이어야 하는
    인코더-디코더 구조.
    """
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.encoder = nn.GRU(input_size, hidden_size, 
                              batch_first=True, bidirectional=True)
        self.bridge = nn.Linear(hidden_size * 2, hidden_size)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
    
    def encode(self, x):
        enc_out, h_n = self.encoder(x)
        h_combined = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        h_bridge = torch.tanh(self.bridge(h_combined))
        return enc_out, h_bridge
```

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
LSTM과 GRU를 상태, 게이트, 계산 묶음, 파라메터와 성능 검증 관점에서 비교하라.

</div>

??? success "풀이"
    LSTM은 은닉·셀 상태와 세 게이트·셀 후보를 쓰고, GRU는 은닉 상태와 두 게이트·상태 후보를 쓴다. 같은 차원·바이어스 규약의 순환 파라메터는 4:3이다. 실제 속도와 장기 과제 성능은 이 비율에서 결정되지 않으며 같은 조건에서 측정해야 한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
특정 과제에서 GRU가 LSTM보다 낫다는 결론을 내리려면 어떤 비교가 필요한가?

</div>

??? success "풀이"
    공통 분할·전처리·평가 기준에서 각각 하이퍼파라미터를 조정하고 여러 초기값의 변동을 확인한다. 정확도뿐 아니라 필요한 지연·메모리·학습 비용을 같은 하드웨어에서 측정한다. 과제 이름이나 작은 데이터라는 설명만으로 결과를 미리 정하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
리셋 게이트를 제거한 단순 셀을 모두 같은 MGU라고 부르면 왜 부정확한가?

</div>

??? success "풀이"
    한 게이트를 쓰는 순환 구조도 후보 식과 게이트 적용 위치가 다를 수 있다. 특정 논문의 MGU를 설명하려면 그 식을 확인해야 한다. 단순히 리셋을 없앤 보간 셀과 이름만으로 동일시하지 않는다. 두 계산 묶음을 사용하면 같은 바이어스 규약의 세 묶음 GRU보다 파라메터가 적지만 품질 보장은 아니다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
감성 분석에서 두 구조를 비교한 결과를 어떤 형태로 보고해야 하는가?

</div>

??? success "풀이"
    사용 데이터와 분할, 토큰화·최대 길이, 모델 크기와 학습 설정을 명시한다. 각 시행의 검증·테스트 지표와 실제 측정 시간을 함께 보고한다. 이 페이지에는 학습한 감성 분석 실험이 없으므로 87%·88% 정확도나 20~30% 속도 차이를 관측 결과로 제시할 수 없다.

## 정리하며

두 구조의 차이는 상태와 게이트 설계에 있다. 같은 규약의 순환 파라메터 수는 비교할 수 있지만 실제 속도·메모리·정확도는 별도 평가가 필요하다. 코드의 추정치, 초기화 그래디언트 실험과 완성된 학습 결과를 구분한다.

**참고 문헌**

1. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735-1780.

2. Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. *EMNLP*.

3. Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling. *NIPS Workshop*.

4. Jozefowicz, R., Zaremba, W., & Sutskever, I. (2015). An Empirical Exploration of Recurrent Network Architectures. *ICML*.

5. Greff, K., Srivastava, R. K., Koutník, J., Steunebrink, B. R., & Schmidhuber, J. (2017). LSTM: A Search Space Odyssey. *IEEE Transactions on Neural Networks and Learning Systems*, 28(10), 2222-2232.
