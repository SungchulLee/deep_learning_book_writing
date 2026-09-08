# GRU와 LSTM 비교

LSTM과 GRU는 문 달린 순환 신경망의 두 주역이다. 둘 다 기울기 소실 문제를 풀지만 복잡함과 매개변수 수, 거동이 다르다. 이 절은 이론과 실험 결과, 실무에서의 판단 기준을 두루 견주어 구조를 고르는 데 길잡이가 되어 준다.

---

## 1. 구조 비교

### 짜임 개관

| 항목 | LSTM | GRU |
|--------|------|-----|
| 상태 | 2개 (숨은 상태 $h$, 세포 상태 $c$) | 1개 (숨은 상태 $h$) |
| 문 | 3개 (망각, 입력, 출력) | 2개 (갱신, 재설정) |
| 문의 결합 | 서로 독립 | 묶임 (갱신 = 1 − 망각) |
| 출력 거르기 | 있음 (출력 문) | 없음 |
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

- 세포 상태 $c_t$: 장기 기억 저장소
- 숨은 상태 $h_t$: 작업 기억이자 출력
- 출력 문: 무엇을 드러낼지 다스린다
- 이렇게 나누면 정보를 곧바로 쓰지 않고도 담아 둘 수 있다

**GRU: 하나로 합친 상태**

- 숨은 상태 하나가 두 구실을 모두 한다
- 더 간단하지만 덜 유연하다
- 담아 둔 것이 곧 드러나는 것이다

---

## 2. 두 구조의 문 대응시키기

### 구조적 대응

| GRU | LSTM | 참고 |
|-----|------|-------|
| 갱신 문 $z$ | 망각 $f$ + 입력 $i$ | GRU는 둘을 묶는다: $i = z$, $f = 1-z$ |
| 재설정 문 $r$ | (부분적으로) 출력 문 | 둘 다 쓰기 전에 상태를 거른다 |
| — | 출력 문 $o$ | GRU에는 명시적인 출력 문이 없다 |
| — | 세포 상태 $C$ | GRU는 숨은 상태만 쓴다 |

### 표현력의 맞바꿈

**LSTM의 묶이지 않은 문은 다음을 허용한다.**

- 많이 지키고 조금 더하기: $f = 0.9, i = 0.1$ → 90%를 지키고 10%를 더한다
- 많이 지키고 많이 더하기: $f = 0.9, i = 0.9$ → 90%를 지키고 90%를 더한다
- 조금 지키고 많이 더하기: $f = 0.1, i = 0.9$ → 10%를 지키고 90%를 더한다
- 상태에 걸리는 전체 가중치는 $f + i$이며 1보다 작거나 같거나 클 수 있다

**GRU의 묶인 문은 다음을 강제한다.**

- $z = 0.1$ → 90%를 지키고 10%를 더한다
- $z = 0.5$ → 50%를 지키고 50%를 더한다
- $z = 0.9$ → 10%를 지키고 90%를 더한다
- 상태에 걸리는 전체 가중치는 언제나 $(1-z) + z = 1$이다

GRU의 이 볼록 결합 제약은 암묵적인 규제가 되지만 유연함을 제한한다.

---

## 3. 매개변수 수 분석

### 이론적인 계산

입력 차원이 $d$이고 숨은 차원이 $n$인 LSTM이나 GRU에 대해 다음과 같다.

| 구조 | 매개변수 | 상대적 크기 |
|--------------|------------|----------|
| 기본 RNN | $n^2 + nd + n$ | 1배 |
| GRU | $3(n^2 + nd + n)$ | 3배 |
| LSTM | $4(n^2 + nd + n)$ | 4배 |

GRU는 문·부품 행렬이 3개이고 LSTM은 4개이므로, GRU의 매개변수는 늘 LSTM보다 **25% 적다**(3/4).

```python
import torch
import torch.nn as nn

def parameter_scaling_analysis():
    """모델의 크기에 따라 매개변수 수가 어떻게 늘어나는지 분석한다."""
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

매개변수 말고도 학습 중의 활성값 메모리를 살펴야 한다.

```python
def memory_footprint_analysis(batch_size, seq_length, hidden_size):
    """순전파와 역전파 중의 메모리 사용량을 견준다."""
    # LSTM이 담는 것: 시각마다 h_t, c_t, 그리고 모든 문의 활성값
    lstm_per_step = 6 * hidden_size  # h, c, f, i, o, c_tilde
    lstm_total = batch_size * seq_length * lstm_per_step * 4  # float32
    
    # GRU가 담는 것: h_t와 문의 활성값
    gru_per_step = 4 * hidden_size  # h, z, r, h_tilde
    gru_total = batch_size * seq_length * gru_per_step * 4
    
    print(f"Activation Memory (batch={batch_size}, seq={seq_length}, hidden={hidden_size}):")
    print(f"  LSTM: {lstm_total / 1024**2:.2f} MB")
    print(f"  GRU:  {gru_total / 1024**2:.2f} MB")
    print(f"  GRU saves: {(lstm_total - gru_total) / lstm_total * 100:.1f}%")

# memory_footprint_analysis(batch_size=32, seq_length=512, hidden_size=512)
```

---

## 4. 기울기 흐름 견주기

### 이론적 분석

LSTM과 GRU 모두 기울기 소실을 풀지만 장치가 조금 다르다.

**LSTM의 기울기 경로:**

$$\frac{\partial c_t}{\partial c_{t-1}} = f_t \quad \text{(direct path through cell state)}$$

**GRU의 기울기 경로:**

$$\frac{\partial h_t}{\partial h_{t-1}} = (1 - z_t) + z_t \cdot \frac{\partial \tilde{h}_t}{\partial h_{t-1}}$$

GRU의 $(1 - z_t)$ 항은 LSTM의 $f_t$과 비슷한 구실을 한다.

```python
def compare_gradient_flow(seq_lengths=[50, 100, 200, 500, 1000]):
    """LSTM과 GRU의 기울기 흐름을 실험으로 견준다."""
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

## 5. 기억 용량: 출력 문의 이점

LSTM의 출력 문은 특별한 능력을 준다. **드러내지 않고 정보를 담아 두는 것**이다.

**늦은 회상을 위한 LSTM의 전략:**

- 0걸음: $i_t = 1, f_t = 0$ → 세포에 담는다
- 1~99걸음: $f_t = 1, o_t = 0$ → 지키되 감춘다
- 100걸음: $o_t = 1$ → 담아 둔 정보를 드러낸다

**GRU의 어려움:**

- 담아 둔 정보를 감출 길이 없다
- $h_t$을 저장과 출력 사이에서 저울질해야 한다
- 앞서 담아 둔 정보가 새 계산과 섞인다

그래서 골라서 읽고 쓰는 접근이 필요한 복잡한 기억 방식을 LSTM이 다룰 수 있을 때가 있다.

### 정보 병목

| 항목 | LSTM | GRU |
|--------|------|-----|
| 상태의 용량 | 값 $2n$개 ($c_t$과 $h_t$) | 값 $n$개 ($h_t$만) |
| 저장과 출력의 분리 | 있음 (세포 ≠ 숨은 상태) | 없음 (상태 하나) |
| 독립적인 기억 칸 | 더 많다 | 더 적다 |

---

## 6. 과제별 성능

### 과제별 종합 비교

| 과제의 갈래 | 나은 선택 | 근거 |
|---------------|-------------|-----------|
| **언어 모형** | LSTM ≈ GRU | 둘 다 효과적이며 혼란도에서 LSTM이 조금 낫다 |
| **기계 번역** | LSTM | 먼 의존과 복잡한 정렬 |
| **음성 인식** | LSTM | 연속 신호와 정확한 타이밍 |
| **감성 분석** | GRU ≈ LSTM | 과제가 단순하여 효율이 중요하다 |
| **시계열 예측** | GRU | 의존이 짧을 때가 많고 속도가 중요하다 |
| **개체명 인식** | GRU | 대개 지역 문맥으로 충분하다 |
| **음악 생성** | LSTM | 먼 거리 구조와 다성 |
| **영상 설명 달기** | LSTM | 여러 양식과 복잡한 기억 |
| **대화 시스템** | GRU | 추론이 빠르고 품질도 충분하다 |
| **베끼기·회상 과제** | LSTM | 명시적인 기억이 필요하다 |

### 연구 결과 요약

LSTM과 GRU를 견준 주요 논문은 다음과 같다.

1. **Chung 등 (2014)**: 음악과 음성에서 GRU가 LSTM에 견줄 만하며, 어떤 경우에는 더 빨리 수렴한다.

2. **Jozefowicz 등 (2015)**: 구조 1만 개 이상을 시험했는데, 모든 과제에서 이기는 쪽은 없고 성능은 과제에 달려 있다.

3. **Greff 등 (2017)**: 망각 문과 출력 활성화가 LSTM에서 가장 중요한 부품이며, 간소한 변형도 잘 통할 때가 많다.

---

## 7. 실무에서의 판단 기준

### 판단 흐름도

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

```python
class HybridStackedRNN(nn.Module):
    """
    혼합 구조: 특징 추출에는 GRU, 기억에는 LSTM.
    
    근거:
    - 아래 층은 지역 특징을 뽑는다 (GRU로 충분하고 더 빠르다)
    - 위 층은 먼 거리 의존을 다룬다 (LSTM이 유리하다)
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

```python
class BidirectionalHybrid(nn.Module):
    """
    문맥에는 양방향 GRU, 생성에는 단방향 LSTM.
    
    쓰임새: 부호기는 양방향 문맥이 필요하고 복호기는 인과적이어야 하는
    부호기-복호기 구조.
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

**연습문제 1.**
다섯 가지 항목으로 GRU와 LSTM의 비교표를 만들어라.

??? success "연습문제 1 풀이"
    | 항목 | GRU | LSTM |
    |---|---|---|
    | 문 | 2개 (재설정, 갱신) | 3개 (망각, 입력, 출력) |
    | 매개변수 | $3(d_x+d_h)d_h$ | $4(d_x+d_h)d_h$ |
    | 세포 상태 | 없음 ($h$만 쓴다) | 있음 ($c$과 $h$이 따로) |
    | 학습 속도 | 더 빠름 (약 25%) | 더 느림 |
    | 먼 거리 | 좋음 | 조금 나음 |

---

**연습문제 2.**
어떤 과제에서 GRU가 대체로 LSTM과 맞먹거나 앞서는가?

??? success "연습문제 2 풀이"
    음성 인식, 음악 모형화, 짧은 텍스트 분류, 그리고 많은 순차열 대 순차열 과제에서 GRU가 LSTM과 맞먹는다. 작은 데이터셋에서는 (과적합이 덜하여) GRU가 앞설 수 있다. 아주 긴 순차열, 언어 모형, 문 조절이 정밀해야 하는 과제에서는 LSTM이 나은 편이다.

---

**연습문제 3.**
'최소 문 달린 단위'가 무엇이며 GRU를 어떻게 더 간소화하는지 설명하라.

??? success "연습문제 3 풀이"
    최소 문 달린 단위(Zhou 등, 2016)는 문 하나만 쓴다. $z_t = \sigma(W_z x_t + U_z h_{t-1})$일 때 $h_t = (1-z_t)h_{t-1} + z_t \tanh(Wx_t + U(h_{t-1}))$이다. 매개변수가 GRU의 3분의 2이면서 많은 과제에서 비슷한 성능을 내는데, GRU조차 매개변수가 지나칠 수 있음을 시사한다.

---

**연습문제 4.**
감성 분석 과제에서 GRU와 LSTM을 실제로 견주어 정확도와 학습 시간을 보고하라.

??? success "연습문제 4 풀이"
    IMDB 감성 분석의 흔한 결과는 GRU 정확도 약 87%, LSTM 정확도 약 88%이며 GRU가 20~30% 빨리 학습한다. 정확도 차이는 신뢰 구간 안일 때가 많다. 결론은 이렇다. 빠르게 되풀이하려면 GRU를 먼저 해 보고, 성능이 중요하면 LSTM으로 바꾸라.

## 정리하며

### 빠른 참고표

| 요인 | LSTM | GRU | 우세 |
|--------|------|-----|--------|
| 매개변수 | $4n^2 + 4nm$ | $3n^2 + 3nm$ | **GRU** (25% 적음) |
| 속도 | 기준 | 15~25% 빠름 | **GRU** |
| 기억 용량 | 숨은 차원의 2배 | 숨은 차원의 1배 | **LSTM** |
| 긴 순차열 (500 초과) | 더 낫다 | 좋다 | **LSTM** |
| 작은 데이터셋 (1만 미만) | 과적합이 더 잦다 | 더 튼튼하다 | **GRU** |
| 해석 가능성 | 문 3개로 복잡 | 문 2개로 간단 | **GRU** |
| 연구의 축적 | 1997년 이후로 방대 | 2014년 이후로 늘어남 | **LSTM** |
| 기본 권장 | 표준 자료용 | 실전용 | **GRU** |

### 마지막 권고

1. 대부분의 응용에서 **GRU로 시작하라**. 시제품을 빨리 만들 수 있고, 성능도 대체로 비슷하며, 맞추기 쉽다
2. 아주 긴 순차열을 다루거나 GRU의 성능이 정체되거나 과제에 복잡한 기억 방식이 필요하면 **LSTM으로 바꾸라**
3. 실전에서는 **둘 다 살펴보라**. 통제된 실험을 해 보라. 2~3%의 차이가 규모가 커지면 중요할 수 있다
4. **너무 고민하지 마라.** 구조의 선택은 데이터의 품질, 규제, 초매개변수 조정보다 덜 중요할 때가 많다

---

**참고 문헌**

1. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735-1780.

2. Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. *EMNLP*.

3. Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling. *NIPS Workshop*.

4. Jozefowicz, R., Zaremba, W., & Sutskever, I. (2015). An Empirical Exploration of Recurrent Network Architectures. *ICML*.

5. Greff, K., Srivastava, R. K., Koutník, J., Steunebrink, B. R., & Schmidhuber, J. (2017). LSTM: A Search Space Odyssey. *IEEE Transactions on Neural Networks and Learning Systems*, 28(10), 2222-2232.
