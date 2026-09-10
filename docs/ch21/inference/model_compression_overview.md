# 단원 65: 모델 눌러 담기

이 단원은 받아들일 만한 정확도를 지키면서 기억 공간, 셈 값, 미룸 늦음을 줄이려 깊은 신경망을 눌러 담는 핵심 재주를 다룬다. 이 재주는 자원이 빠듯한 기기(손전화, 가장자리, 사물 인터넷)와 실전 환경에 모델을 펼치는 데 결정적이다.

---

## 1. 학습 목표

- 깊은 신경망의 셈과 기억 공간 병목을 이해한다
- 양자화 재주(익힌 뒤 양자화와 양자화를 헤아린 익히기)를 익힌다
- 여러 가지치기 전략(크기 바탕, 짜임 있는, 짜임 없는)을 짠다
- 앎 내리기로 모델을 눌러 담는다
- 눌러 담은 모델의 정확도와 효율 맞바꿈을 값매김한다

---

## 2. 미리 알아야 할 것

- 단원 20: 앞먹임 그물
- 단원 23: 누비기 신경망
- 단원 14: 손실 함수
- 단원 15: 가장 좋게 하개
- PyTorch nn.Module에 대한 기본 이해

---

## 3. 1부: 이론의 바탕

### 1.1 왜 모델을 눌러 담는가?

요즘 깊은 신경망은 결정적인 어려움 셋을 마주한다:

1. **기억 공간 자국**:
   - ResNet-50: 약 100MB(매개변수 2500만 × float32 4바이트)
   - GPT-3: 약 700GB(매개변수 1750억)
   
2. **셈 값**:
   - 손전화 기기: 제한된 FLOPS, 배터리 제약
   - 클라우드 미룸: 값이 셈 시간에 비례한다
   
3. **늦음 요건**:
   - 실시간 쓰임새는 미룸이 100ms 아래여야 한다
   - 스스로 모는 차, 의료 진단은 곧바른 응답이 필요하다

### 1.2 양자화

**정의**: 무게와 깨어남의 수치 정밀도를 뜬소수점(32비트)에서 더 적은 자릿수(8비트, 4비트, 심지어 두 값)로 줄이기.

#### 수학적 정식화

뜬소수점 값을 갖는 텐서 **x**에 대해 양자화는 다음처럼 대응시킨다:

```
x_float ∈ ℝ → x_quant ∈ {0, 1, ..., 2^b - 1}
```

여기서 `b`은 자릿수이다.

**양자화 함수**:
```
x_quant = round((x_float - zero_point) / scale)

여기서 각 기호는 다음과 같다.
scale = (x_max - x_min) / (2^b - 1)
zero_point = round(-x_min / scale)
```

**양자화 되돌리기**(셈을 위해):
```
x_dequant = scale × x_quant + zero_point
```

#### 양자화의 갈래

1. **익힌 뒤 양자화(PTQ)**:
   - 다시 익히지 않고 미리 익힌 모델을 양자화한다
   - 더 빠르지만 정확도를 더 잃을 수 있다
   - 남는 여유가 있는 모델에 알맞다

2. **양자화를 헤아린 익히기(QAT)**:
   - 익히는 동안 양자화를 흉내낸다
   - 양자화 어긋남에 튼튼한 무게를 배운다
   - 정확도는 더 좋지만 다시 익혀야 한다

#### 이론상의 이점

- **기억 공간 줄이기**: FP32→INT8은 4배, FP32→INT4는 8배
- **빠르기 나아짐**: 특화된 하드웨어에서 INT8 연산이 더 빠르다
- **에너지 효율**: 정밀도가 낮을수록 전력이 덜 든다

**정확도 맞바꿈**:
```
Δ_accuracy ∝ quantization_error²
where quantization_error = x_float - x_dequant
```

### 1.3 가지치기

**정의**: 그물에서 남아도는 무게와 신경 세포나 덜 중요한 것을 없애 성긴 모델을 만들기.

#### 가지치기의 갈래

1. **짜임 없는 가지치기**:
   - 크기에 따라 낱낱의 무게를 없앤다
   - 더 성기게 만들 수 있다(90% 이상)
   - 빨라지려면 성긴 텐서 연산이 필요하다
   
2. **짜임 있는 가지치기**:
   - 거르개, 채널, 층 전체를 없앤다
   - 덜 성기지만 보통의 연산으로도 된다
   - 여느 하드웨어에서 곧바로 빨라진다

#### 수학 얼거리

무게 텐서 **W**에 대해 중요도 점수를 정한다:
```
importance(w_i) = |w_i|  (magnitude-based)
                = |∂L/∂w_i|  (gradient-based)
                = w_i² × (∂L/∂w_i)²  (Hessian approximation)
```

**가지치기 잣대**:
```
Prune w_i if importance(w_i) < threshold_τ
```

**성김 비율**:
```
sparsity = (# pruned parameters) / (# total parameters)
```

#### 되풀이 크기 가지치기(IMP)

복권 가설은 다음을 말한다:

1. 그물을 모일 때까지 익힌다
2. 크기가 가장 작은 무게의 p%를 쳐 낸다
3. 남은 무게를 첫자리매김 값으로 되돌린다
4. 익히기를 되풀이한다

**모임 정리**(느슨하게):
```
∃ subnetwork S ⊂ Network that achieves:
accuracy(S) ≥ accuracy(Network) - ε
with |S| << |Network|
```

### 1.4 앎 내리기

**정의**: 내놓는 분포를 맞춰 작은 "제자" 모델이 큰 "스승" 모델을 흉내내도록 익히기.

#### 수학적 정식화

스승에게서 온 **부드러운 목표**:
```
p_i^teacher = softmax(z_i / T)
where T is temperature (T > 1 makes distribution softer)
```

**제자의 익히기 목표**:
```
L_total = α × L_hard + (1-α) × L_soft

L_hard = CrossEntropy(y_student, y_true)
L_soft = KL_divergence(p_student, p_teacher) × T²
```

T² 항이 기울기의 크기 잣수를 메워 준다.

**직관**:

- 딱딱한 이름표: {cat: 1, dog: 0, car: 0}
- 부드러운 이름표: {cat: 0.9, dog: 0.08, car: 0.02}
- 부드러운 이름표가 갈래 사이 관계에 대해 더 풍부한 앎을 준다

#### 통하는 까닭

1. **어두운 앎**: 스승의 틀린 확률이 닮음의 짜임을 담고 있다
2. **벌주기**: 제자가 딱딱한 이름표에 지나치게 맞춰지는 것을 막는다
3. **두루 통함**: 부드러운 목표가 이름표 부드럽게 하기 노릇을 한다

### 1.5 깊은 눌러 담기 물길

Han 외(2016)는 가지치기, 양자화, 엔트로피 부호를 세 단계 물길 하나로 아우르면 정확도를 잃지 않고 35~49배 눌러 담을 수 있음을 보였다. 재주를 따로따로 쓰는 대신 이 물길은 서로 채워 주는 센 점을 써먹는다:

$$
\text{Compression} = \underbrace{\text{Pruning}}_{\text{reduce connections}} \times \underbrace{\text{Quantization}}_{\text{reduce bits}} \times \underbrace{\text{Huffman Coding}}_{\text{exploit redundancy}}
$$

#### 1단계: 크기 바탕 가지치기

이 물길은 되풀이 가지치기로 시작한다. 곧 그물을 익히고, 배운 문턱값 아래 무게를 없애고, 정확도를 되찾으려 다시 익힌다. AlexNet에서는 매개변수가 9분의 1로, VGG-16에서는 13분의 1로 준다. 핵심 눈썰미는 그물에 매개변수가 아주 많이 남아돌고, 곱게 다듬으면 남은 무게가 없앤 것을 메울 수 있다는 것이다.

#### 2단계: 무게 나눠 쓰기를 곁들인 익힌 양자화

가지친 뒤 남은 무게를 k-평균으로 층마다 $k$개의 함께 쓰는 값으로 무리 짓는다. 무게마다 32비트 실수로 담는 대신 작은 부호책 번호만 담는다:

$$
w_i \approx c_{j}, \quad j = \arg\min_j |w_i - c_j|
$$

기울기를 무리마다 쌓아 함께 쓰는 가운데점에 걸므로 익히는 동안 부호책을 곱게 다듬을 수 있다. 그래서 담는 자리가 무게마다 32비트에서 $\log_2(k)$비트로 준다(층 갈래에 따라 흔히 4~8비트다). 누비기 층은 무리를 더 적게 쓰고(흔히 256개, 곧 8비트) 온통 이은 층은 더 적게도 쓸 수 있다(16~32개, 곧 4~5비트).

#### 3단계: 허프먼 부호

양자화한 무게와 성긴 번호의 분포는 고르지 않다. 곧 어떤 값이 다른 것보다 훨씬 자주 나온다. 허프먼 부호는 자주 나오는 값에 짧은 부호를 매겨, 가지치기와 양자화를 아우른 위에 20~30%를 더 눌러 담는다.

#### 아우른 결과

| 모델 | 본디 크기 | 가지친 뒤 | 양자화한 뒤 | 허프먼 뒤 | 전체 눌러 담기 |
|-------|--------------|---------------|-------------------|---------------|-------------------|
| AlexNet | 240메가바이트 | 27메가바이트(9배) | 6.9메가바이트(35배) | 6.2메가바이트(39배) | **39배** |
| VGG-16 | 552메가바이트 | 42메가바이트(13배) | 11.3메가바이트(49배) | 11.0메가바이트(50배) | **49배** |

#### 계량 금융에 뜻하는 바

깊은 눌러 담기 물길은 계량 모델을 펼치는 데 곧바로 맞닿는다:

- **늦음이 적은 거래**: 눌러 담은 모델이 L1/L2 곳간에 들어가 미룸 늦음이 밀리초에서 마이크로초로 준다. 고빈도 전략에 결정적이다.
- **가장자리 펼치기**: 눌러 담은 위험 모델이 클라우드를 오가지 않고 기기에서 돌아 실시간 자산 꾸러미를 지켜볼 수 있다.
- **모아 쓰기의 효율**: 눌러 담으면 같은 기억 공간 예산으로 더 큰 모음을 돌려 어림의 튼튼함이 나아진다.
- **값 줄이기**: 모델이 작아지면 많은 종목을 묶음으로 미룰 때의 클라우드 셈 값이 준다.
---

## 4. 2부: 짜기의 짜임

### 단원의 짜임

```
65_model_compression/
├── README.md                          # 이 파일
├── requirements.txt                   # 딸린 꾸러미
├── utils.py                           # 함께 쓰는 도구
│
├── 01_quantization_basics.py         # 첫걸음: 익힌 뒤 양자화
├── 02_pruning_basics.py               # 첫걸음: 크기 바탕 가지치기
├── 03_knowledge_distillation.py       # 첫걸음: 기본 앎 내리기
│
├── 04_quantization_aware_training.py  # 가운데: 양자화를 헤아린 익히기
├── 05_structured_pruning.py           # 가운데: 거르개/채널 가지치기
├── 06_iterative_pruning.py            # 가운데: 되풀이 크기 가지치기 알고리즘
│
├── 07_mixed_precision_quantization.py # 앞선: 층마다의 자릿수
├── 08_neural_architecture_search.py   # 앞선: 눌러 담기를 위한 자동 기계 배움
└── 09_combined_compression.py         # 앞선: 가지치기 + 양자화 + 앎 내리기
```

### 어려움 수준

**첫걸음**(01~03):

- 고갱이 개념을 이해하는 데 초점
- 자세한 주석을 곁들인 단순한 짜기
- 작은 모델(LeNet, 단순 누비기 신경망)
- 또렷한 정확도와 눌러 담기의 맞바꿈

**가운데**(04~06):

- 더 정교한 재주
- 익히기와 아우르기
- 가운데 크기 모델(ResNet-18, MobileNet)
- 곱게 다듬기와 되찾기 전략

**앞선**(07~09):

- 가장 앞선 방법
- 아우른 눌러 담기 물길
- 큰 모델(ResNet-50, 보기 변환기)
- 실전에 쓸 수 있는 짜기

---

## 5. 3부: 실전에서 헤아릴 점

### 하드웨어 받침

| 하드웨어 | INT8 | INT4 | FP16 | 성김 |
|----------|------|------|------|--------|
| CPU      | ✓    | ✗    | ✗    | ✗      |
| GPU      | ✓    | ✓    | ✓    | Limited|
| TPU      | ✓    | ✗    | ✓    | ✗      |
| ARM/손전화| ✓   | ✓    | ✓    | ✗      |

### 눌러 담기 지침

1. **늘 재어라**:
   - 정확도 떨어짐
   - 기억 공간 줄어듦
   - 미룸 빨라짐(이론뿐 아니라)
   
2. **조심스럽게 시작하라**:
   - 8비트 양자화로 시작한다
   - 세게 눌러 담기 앞에 50%를 쳐 낸다
   - 정확도를 되찾으려 앎 내리기를 쓴다

3. **층의 민감함**:
   - 첫 층과 마지막 층이 가장 민감하다
   - 가운데 층은 세게 눌러 담아도 된다
   - 묶음 고르게 맞추기 층은 흔히 FP32로 두어야 한다

---

## 6. 4부: 값매김 잣대

### 모델 크기
```python
size_mb = (num_parameters × bytes_per_parameter) / (1024²)
compression_ratio = size_original / size_compressed
```

### 미룸 시간
```python
latency = forward_pass_time (averaged over multiple runs)
speedup = latency_original / latency_compressed
```

### 정확도
```python
accuracy_drop = accuracy_original - accuracy_compressed
acceptable_drop < 1-2% for most applications
```

### 효율 점수
```python
efficiency = accuracy / (latency × model_size)
```

---

## 7. 쓰는 법

### 빠른 시작
```bash
# 딸린 꾸러미를 깐다
pip install -r requirements.txt

# 첫걸음 보기를 돌린다
python 01_quantization_basics.py
python 02_pruning_basics.py
python 03_knowledge_distillation.py

# 가운데 보기를 돌린다
python 04_quantization_aware_training.py
python 05_structured_pruning.py

# 앞선 보기를 돌린다
python 09_combined_compression.py
```

### 주피터 공책
모든 각본은 주고받으며 살펴보도록 주피터 공책에서 돌릴 수 있다:
```bash
jupyter notebook
```

---

## 8. 배움 길

**1주**: 양자화 이론 + 익힌 뒤 양자화(01)
**2주**: 가지치기 이론 + 크기 가지치기(02)
**3주**: 앎 내리기(03)
**4주**: 양자화를 헤아린 익히기 + 짜임 있는 가지치기(04~05)
**5주**: 앞선 재주(06~07)
**6주**: 아우른 방법 + 기획(08~09)

---

## 9. 더 볼 것

- PyTorch Quantization Tutorial: https://pytorch.org/docs/stable/quantization.html
- TensorFlow Model Optimization: https://www.tensorflow.org/model_optimization
- ONNX Runtime Quantization: https://onnxruntime.ai/docs/performance/quantization.html
- Papers With Code (Model Compression): https://paperswithcode.com/task/model-compression

---

## 연습문제

1. **양자화 살피기**: 양자화 어긋남이 층을 지나며 어떻게 퍼지는지 재어라
2. **가지치기 민감도**: 어느 층을 가장 세게 쳐 낼 수 있는지 가려내어라
3. **앎 내리기 실험**: 온도 값과 α 무게를 달리해 견주어라
4. **눌러 담기 물길**: 좋아하는 모델에 끝에서 끝까지의 물길을 세워라
5. **손전화 펼치기**: 눌러 담은 모델을 손전화 기기(iOS/안드로이드)에 펼쳐라

---

## 정리하며

이 마당은 학습 목표、미리 알아야 할 것、1부: 이론의 바탕、2부: 짜기의 짜임을 차례로 짚었다.

**참고 문헌**

1. **양자화**:
   - Jacob et al. "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (2018)
   - Banner et al. "Post training 4-bit quantization of convolutional networks for rapid-deployment" (2019)

2. **가지치기**:
   - Han et al. "Learning both Weights and Connections for Efficient Neural Networks" (2015)
   - Frankle & Carbin "The Lottery Ticket Hypothesis" (2019)

3. **앎 내리기**:
   - Hinton et al. "Distilling the Knowledge in a Neural Network" (2015)
   - Romero et al. "FitNets: Hints for Thin Deep Nets" (2015)

4. **총설**:
   - Cheng et al. "Model Compression and Acceleration for Deep Neural Networks" (2020)
   - Gholami et al. "A Survey of Quantization Methods for Efficient Neural Network Inference" (2021)

---
