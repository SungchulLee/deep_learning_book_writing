# 모델 압축 개요

모델 압축은 가중치의 저장 공간과 추론에 필요한 자원을 줄이는 방법이다. 양자화, 가지치기, 지식 증류는 서로 다른 부분을 바꾸므로 함께 사용할 수도 있다. 파일 크기 감소, 실행 중 메모리 감소, 지연 시간 단축은 각각 측정해야 한다. 작은 파일이 항상 빠른 모델을 뜻하지는 않는다.

## 1. 학습 목표와 사전 지식

이 절에서는 양자화의 표현 오차, 가지치기의 희소성과 실제 연산량, 지식 증류의 학습 목표를 이해하고 압축 전후의 품질과 비용을 비교한다. 선형 계층, 합성곱, 손실 함수, 역전파와 PyTorch `nn.Module`에 대한 기본 지식을 전제로 한다.

가중치만 FP32로 저장하면 파라미터 2,500만 개는 약 100 MB, 1,750억 개는 약 700 GB다. 여기서 GB는 10⁹바이트다. 활성화, KV 캐시, 작업 공간, 학습 중 그래디언트와 옵티마이저 상태는 별도다. 배포 환경에 따라 병목은 가중치 읽기, 연산, 통신 또는 메모리 용량일 수 있다. 실시간 응답 기준도 애플리케이션마다 정해야 한다.

## 2. 양자화: 적은 비트로 수치 표현하기

양자화는 실수 값을 제한된 대표값에 대응시킨다. 가중치만 양자화하는 방법과 활성화도 양자화하는 방법은 필요한 커널과 오차 특성이 다르다.

```
x_float ∈ ℝ → x_quant ∈ {0, 1, ..., 2^b - 1}
```

위는 부호 없는 b비트 정수의 표현 범위다. 부호 있는 정수를 쓰면 범위가 달라진다. 양의 스케일 s와 정수 영점 z를 사용하는 아핀 양자화는 다음처럼 정의할 수 있다.

$$q=\operatorname{clip}(\operatorname{round}(x/s)+z,q_{\min},q_{\max}),\qquad
\hat{x}=s(q-z).$$

다음에 보존된 수식 블록은 실수 단위 오프셋과 정수 영점의 의미를 혼용한다. 특히 `round(-x_min/scale)`로 정한 정수 영점을 실수 x에서 직접 빼거나 복원값에 그대로 더하면 위 정의와 일치하지 않는다. 개념과 계산에는 위의 일관된 정의를 사용한다.

```
x_quant = round((x_float - zero_point) / scale)

여기서 각 기호는 다음과 같다.
scale = (x_max - x_min) / (2^b - 1)
zero_point = round(-x_min / scale)
```

```
x_dequant = scale × x_quant + zero_point
```

**학습 후 양자화(PTQ)**는 사전 학습된 가중치를 추가적인 그래디언트 학습 없이 변환한다. 방법에 따라 대표 입력으로 범위를 보정하거나 계층 출력을 복원하는 최적화를 수행한다. **양자화 인지 학습(QAT)**은 학습이나 파인튜닝 중 양자화·복원 오차를 모사해 가중치를 적응시킨다. QAT가 항상 더 정확한 것은 아니며 데이터, 비트 수, 학습 설정과 배포 연산의 일치 여부가 중요하다.

FP32 가중치를 실제 INT8이나 INT4로 패킹하면 가중치 데이터 자체는 각각 1/4, 1/8로 줄어든다. 스케일·영점·그룹 정보와 양자화하지 않은 계층은 추가 공간을 쓴다. 추론 속도와 에너지 절감은 지원 하드웨어와 커널, 변환 비용, 배치 크기에 달려 있다.

```
Δ_accuracy ∝ quantization_error²
where quantization_error = x_float - x_dequant
```

위 비례식은 보편적인 정확도 법칙이 아니다. 같은 평균 제곱 양자화 오차라도 어느 계층과 방향에 발생했는지에 따라 최종 예측이 달라진다. 분류 정확도는 작은 출력 변화에도 불연속적으로 바뀔 수 있다.

## 3. 가지치기: 연결과 구조 줄이기

비구조적 가지치기는 개별 가중치를 제거한다. 구조적 가지치기는 채널, 헤드 또는 계층처럼 연산 단위를 제거한다. 비구조적 마스크로 0을 많이 만들어도 밀집 텐서를 그대로 저장하고 계산하면 파일 크기와 실행 시간이 줄지 않을 수 있다. 구조적 가지치기도 실제 텐서 크기와 연결된 계층을 함께 바꾸어야 연산량이 감소한다.

```
importance(w_i) = |w_i|  (magnitude-based)
                = |∂L/∂w_i|  (gradient-based)
                = w_i² × (∂L/∂w_i)²  (Hessian approximation)
```

크기 |w|는 간단한 중요도 지표이며, 그래디언트나 손실 변화의 근사를 활용하는 방법도 있다. 위의 w²(∂L/∂w)²는 그래디언트 기반의 경험적 점수로 볼 수 있으나 일반적인 헤시안 그 자체의 근사는 아니다. 2차 테일러 전개를 이용한 중요도에는 손실의 2차 미분과 추가 가정이 필요하다.

```
Prune w_i if importance(w_i) < threshold_τ
```

```
sparsity = (# pruned parameters) / (# total parameters)
```

희소도는 전체 중 제거된 가중치의 비율이다. 분모가 전체 계층인지 가지치기 대상 계층인지 명시한다. 임계값과 같은 점수, 동점 처리와 최소 채널 수에 따라 실제 제거 비율이 달라질 수 있다.

반복적 크기 기반 가지치기(IMP)는 학습, 작은 가중치 제거, 남은 가중치의 초기값 또는 지정 시점 값으로 되돌리기, 재학습을 반복하는 방식이다. 매 반복마다 남은 가중치의 비율 p를 제거하면 k회 뒤 남은 비율은 (1−p)ᵏ다. 원래 전체의 p를 매번 제거하는 것과 구분한다.

```
∃ subnetwork S ⊂ Network that achieves:
accuracy(S) ≥ accuracy(Network) - ε
with |S| << |Network|
```

위 문장은 복권 가설의 직관을 요약한 것이다. 모든 모델에 대해 작은 부분망이 반드시 같은 정확도를 달성한다는 수렴 정리가 아니다. 초기화, 학습 예산과 부분망을 찾는 절차가 결과에 영향을 준다.

## 4. 지식 증류: 작은 모델 학습하기

지식 증류는 교사 모델의 출력이나 중간 표현을 학습 신호로 사용해 학생 모델을 훈련한다. 교사와 학생이 같은 클래스 집합을 예측하는 경우 온도 T>0로 로짓을 나눈 소프트맥스 분포를 비교할 수 있다.

```
p_i^teacher = softmax(z_i / T)
where T is temperature (T > 1 makes distribution softer)
```

T를 높이면 동일한 로짓의 분포가 완만해져 정답 외 클래스 사이의 상대적인 정보가 더 드러날 수 있다. 학생 분포에도 같은 온도를 적용한다.

```
L_total = α × L_hard + (1-α) × L_soft

L_hard = CrossEntropy(y_student, y_true)
L_soft = KL_divergence(p_teacher ‖ p_student) × T²   (앞자리가 교사다)
```

여기서 하드 레이블 손실은 보통 T=1의 학생 로짓과 정답 사이의 교차 엔트로피다. 소프트 손실은 교사에서 학생 방향의 KL 발산이며 0≤α≤1이다. T²는 고온에서 줄어드는 로짓 그래디언트 크기를 보정하기 위한 관례다. 교사의 출력은 고정된 목표로 취급한다.

교사가 오답일 때도 그 확률이 유용한 유사성을 포함할 수 있지만 오류와 편향도 전달할 수 있다. 증류가 학생의 일반화 성능을 개선하는지는 별도의 검증 문제다. 정답 레이블이 없는 증류나 특징 기반 증류도 가능하며 위 손실만이 유일한 방법은 아니다.

## 5. 압축 방법 결합하기

[Deep Compression 논문](https://arxiv.org/abs/1510.00149)은 가지치기, 학습된 가중치 공유, 허프먼 부호화를 결합했다. 해당 실험에서 AlexNet과 VGG-16의 저장 공간을 각각 약 35배와 49배 줄였으며 기준 정확도를 유지했다. 이는 논문의 모델과 실험 조건에 대한 결과다.

가중치 공유에서는 남은 가중치를 k개의 중심값으로 묶고 각 가중치에 중심값 인덱스를 저장한다. 고정 길이 인덱스는 값마다 ⌈log₂k⌉비트가 필요하고, 코드북과 희소 인덱스도 저장해야 한다. 허프먼 부호화는 출현 빈도를 활용해 추가 압축하지만 실행 시 해독과 접근 비용이 생길 수 있다. 각 단계의 압축 배율은 같은 기준과 저장 항목을 사용한 경우에만 일관되게 곱할 수 있다.

금융 등 실제 서비스에서는 예측 성능과 함께 지연 시간 분포, 메모리 사용량, 데이터 전처리와 통신 비용을 평가한다. 모델이 작아졌다는 이유만으로 밀리초 응답이 마이크로초로 바뀌거나 투자 성과가 좋아진다고 결론 내리지 않는다.

## 6. 예제 구성과 읽는 순서

다음 디렉터리 구조는 보존된 예제 안내다. 이 문서의 실제 경로와 파일 목록을 그대로 나타내지는 않는다.

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

이 절 이후에는 [언어 모델 양자화](quantization.md), [양자화 인지 학습](04_quantization_aware_training.md), [구조적 가지치기](05_structured_pruning.md), [압축 방법 결합](06_combined_compression.md), [증류 벤치마크](07_hf_distillation_benchmark.md)를 차례로 살펴본다. 예제의 구현 범위와 실제로 측정한 결과를 구분하며, 본문에 있는 개념 설명이 모든 코드에 완성된 형태로 구현되어 있다고 가정하지 않는다.

## 7. 배포와 평가 기준

INT8·INT4·FP16·희소 연산의 지원 여부는 장치 이름만으로 판단할 수 없다. 같은 GPU나 CPU에서도 세대, 연산 종류, 메모리 배치, 백엔드와 커널에 따라 지원과 성능이 다르다. 먼저 목표 실행 환경에서 지원하는 표현을 정하고 계층별 민감도와 품질 변화를 확인한다. 첫 계층과 마지막 계층을 높은 정밀도로 남기는 방식도 실험할 선택지다.

```
size_mb = (num_parameters × bytes_per_parameter) / (1024²)
compression_ratio = size_original / size_compressed
```

위 `size_mb`는 1024²로 나누므로 단위가 엄밀히는 MiB다. 파라미터 수에 바이트 수를 곱한 값에는 버퍼, 희소 인덱스와 스케일이 빠진다. 직렬화 파일 크기, 가중치 메모리, 실행 중 최대 메모리를 각각 기록한다.

```
latency = forward_pass_time (averaged over multiple runs)
speedup = latency_original / latency_compressed
```

지연 시간은 워밍업 후 같은 배치·입력 길이·장치에서 비교한다. GPU는 비동기 실행을 고려해 동기화하거나 장치 이벤트를 사용한다. 평균뿐 아니라 분산과 상위 백분위 지연 시간, 처리량도 목적에 맞게 측정한다.

```
accuracy_drop = accuracy_original - accuracy_compressed
acceptable_drop < 1-2% for most applications
```

정확도 감소는 절대 퍼센트포인트와 상대 감소율을 구분한다. 1~2%라는 허용 범위는 보편적인 기준이 아니다. 실제 용도에서 필요한 최소 성능, 분야별 데이터와 실패 비용에 따라 기준을 정한다. 언어 모델은 퍼플렉서티와 다운스트림 태스크도 함께 평가할 수 있다.

```
efficiency = accuracy / (latency × model_size)
```

위 효율 점수는 단위와 정확도 표기 방식에 따라 값이 달라지는 임의 지표다. 품질·지연 시간·크기를 따로 보고, 한 지표를 개선하려면 다른 지표를 포기해야 하는 파레토 관계를 확인하는 편이 해석하기 쉽다.

## 8. 실습 환경과 학습 계획

아래 명령은 예제 파일과 의존성이 해당 작업 디렉터리에 존재한다는 전제의 안내다. 이 페이지의 링크를 따라간 경로와 명령 속 파일명이 일치하는지 먼저 확인한다.

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

```bash
jupyter notebook
```

양자화의 표현 오차를 이해한 뒤 가지치기와 증류를 각각 살펴보고, 마지막에 결합 방법을 비교한다. 실습 시에는 기준 모델과 압축 모델의 데이터 분할, 평가 전처리와 측정 조건을 맞춘다. 압축 설정을 고르는 검증 데이터와 최종 성능을 보고할 테스트 데이터도 구분한다.

## 연습문제

1. FP32 파라미터 1억 개를 INT8로 저장할 때 가중치 데이터만의 크기와 이상적인 압축 배율을 계산하라. 실제 파일에는 어떤 정보가 더 필요한가?
2. 가중치의 80%를 마스크로 0으로 만든 모델이 밀집 연산에서 빨라지지 않는 이유를 설명하라.
3. 남은 가중치의 20%를 세 번 제거할 때 최종 희소도를 구하라. 각 단계 뒤의 재학습은 어떤 역할을 하는가?
4. 증류에서 온도를 높이는 목적과 교사의 오류가 전달될 수 있는 이유를 설명하라.
5. 압축 전후의 품질, 파일 크기, 최대 메모리와 지연 시간을 공정하게 비교할 평가 계획을 작성하라.

## 정리하며

양자화는 수치 표현을, 가지치기는 연결이나 구조를, 지식 증류는 작은 모델의 학습 목표를 바꾼다. 표현 비트 수나 0의 개수만으로 실제 속도를 예측할 수는 없다. 압축 후 품질과 저장 공간, 실행 메모리, 처리량과 지연 시간을 동일한 조건에서 따로 비교해야 한다.

**참고 문헌**

1. **양자화**:
   - Jacob et al. "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (2018)
   - Banner et al. "Post training 4-bit quantization of convolutional networks for rapid-deployment" (2019)

2. **가지치기**:
   - Han et al. "Learning both Weights and Connections for Efficient Neural Networks" (2015)
   - Frankle & Carbin "The Lottery Ticket Hypothesis" (2019)

3. **지식 증류**:
   - Hinton et al. "Distilling the Knowledge in a Neural Network" (2015)
   - Romero et al. "FitNets: Hints for Thin Deep Nets" (2015)

4. **총설**:
   - Cheng et al. "Model Compression and Acceleration for Deep Neural Networks" (2020)
   - Gholami et al. "A Survey of Quantization Methods for Efficient Neural Network Inference" (2021)

---
