# 1장: 입문과 PyTorch 기초

!!! note "이 장은 두 묶음이다"
    **1.1~1.3**은 딥러닝을 돌릴 **기계와 도구**를 훑는다 — 컴퓨터 구조, NumPy·Pandas·Matplotlib,
    그리고 PyTorch 첫걸음. 처음부터 차례로 읽는 자리다.

    **1.4~1.7**은 그 PyTorch를 **체계적으로 파고드는** 자리다 — 텐서, 텐서 속성, 자동 미분,
    경사 하강법. 분량이 크고, [3장](../ch03/index.md)부터의 사다리를 걷는 데 꼭 먼저 읽어야 하는
    것은 아니다. **걷다가 막히는 자리에서 찾아보는 참고 묶음**으로 두어도 된다.

딥러닝 모델은 컴퓨터에서 실행되는 계산이다. 이 장에서는 계산을 수행하는 하드웨어와 데이터를 처리하는 소프트웨어를 먼저 살펴본다.

뉴럴 네트워크를 배우면서 마주치는 물음 가운데 상당수는 수학이 아니라 기계에 관한 것이다.

- 왜 `.to(device)` 같은 줄을 써야 하는가
- 왜 배치 크기를 늘리면 메모리가 모자란다고 하는가
- 왜 GPU가 있는데도 학습이 느린가
- 왜 텐서 연산을 Python 반복문보다 우선하는가

이 물음들은 뒤에서 코드를 만나는 순간 곧바로 닥친다. 그때마다 멈추어 설명하는 대신, 여기서 한 번에 정리하고 넘어간다.

| 절 | 내용 | 답하는 물음 |
|---|---|---|
| 1.1 | 컴퓨터 구조 | 계산은 어디서 일어나고 데이터는 어디에 있는가 |
| 1.2 | Python 데이터 분석 라이브러리 | 수치 데이터를 어떤 도구로 처리하는가 |
| 1.3 | PyTorch | GPU 연산과 자동 미분을 어떻게 사용하는가 |
| 1.4 | 텐서 | 만들고, 다루고, 장치에 올리고, 저장한다 |
| 1.5 | 텐서 속성 | 모양·자료형·장치, 그리고 메모리 위의 배치 |
| 1.6 | 자동 미분 | `loss.backward()` 가 무엇을 하는가 |
| 1.7 | 경사 하강법 | 그 경사로 웨이트를 어떻게 움직이는가 |

## 1.1 컴퓨터 구조

CPU는 적은 수의 똑똑한 일꾼이고 GPU는 아주 많은 수의 단순한 일꾼이다. 뉴럴 네트워크의 계산이 "같은 계산을 수백만 번" 하는 일이기 때문에 GPU가 딥러닝의 기계가 되었다.

메모리와 저장장치는 레지스터, 캐시, RAM, SSD 등으로 구성되며, 용량과 접근 속도가 서로 다르다. 여기서 딥러닝 코드의 첫 번째 규율이 나온다. **계산을 줄이려 애쓰기 전에 데이터가 어디에 있는지 보라.**

다만 GPU는 만능이 아니다. 실제로 재어 보면 작은 과제에서는 GPU가 CPU보다 **느리다.** 그 까닭과 임계를 1.3에서 직접 재어 확인한다.

작은 모델 하나가 GPU 자원을 충분히 활용하지 못한다면, **독립적인 실험을 동시에 실행하는 방법**을 고려할 수 있다. 다만 메모리와 연산 자원을 공유하므로 동시 실행이 항상 유리한 것은 아니다. 다음은 여러 시드의 실험을 병렬로 실행한 사례다.

| 동시에 돌린 수 | 처리량 |
|---|---|
| 1 | 1.00× |
| 3 | **2.00×** |
| 4 | 2.11× |
| 5 | 1.51× |
| 6 | 1.25× |

[4.4절](../ch04/02_depth.md)의 실험을 M3에서 재어 본 값이다. 셋을 동시에 돌리면 각각은 1.6배 느려지지만 **셋을 한꺼번에 끝내므로 순 처리량이 두 배**가 된다. 다섯을 넘기면 다투기 시작해 도로 떨어진다.

**어디가 꼭대기인지는 기계마다 다르므로 재어 보아야 한다.** 동시 실행 수를 늘리면 처리량이 개선될 수 있지만, 자원 경쟁이 커지면 오히려 느려질 수 있다. 이 표의 최적 실행 수를 다른 하드웨어에 그대로 적용해서는 안 된다.

## 1.2 Python 데이터 분석 라이브러리

- **NumPy** — 수치 배열을 다루는 라이브러리. 반복문으로 작성한 계산을 배열 연산으로 표현하는 방법을 익힌다
- **Pandas** — 표로 된 데이터를 읽고 다듬어 숫자 배열로 넘겨준다
- **Matplotlib** — 결과를 눈으로 확인한다. 학습이 잘못되어도 프로그램은 오류를 내지 않으므로, 그려 보는 일이 곧 검사이다

## 1.3 PyTorch

PyTorch 텐서는 NumPy 배열과 비슷한 방식으로 데이터를 표현하며, 장치 선택과 자동 미분 기능을 제공한다. 다음 식은 두 도구의 관계를 이해하기 위한 비유다.

$$\text{Tensor} = \text{ndarray} + \text{장치} + \text{자동 미분}$$

배열 연산의 문법이 비슷하므로 NumPy를 익히면 PyTorch를 배우기 쉽다. 이 절에서는 두 도구의 공통점을 바탕으로 **장치 관리**와 **자동 미분**을 소개한다.

## 1.4 텐서

PyTorch의 기본 자료구조. 만드는 법부터 GPU에 올리고 저장하는 법까지.

**만들기**

- [스칼라를 텐서로](../ch06/tensors/01_scalar_to_tensor.md) · [리스트를 텐서로](../ch06/tensors/02_list_to_tensor.md) · [범위를 텐서로](../ch06/tensors/03_range_to_tensor.md)
- [NumPy 배열을 텐서로](../ch06/tensors/04_numpy_array_to_tensor.md) · [Pandas를 텐서로](../ch06/tensors/05_pandas_to_tensor.md)
- [팩토리 함수](../ch06/tensors/06_factory_functions.md) — `zeros`, `ones`, `empty`, `rand` 가 각각 무엇을 보장하는가

**다루기**

- [별칭, 뷰, clone의 차이](../ch06/tensors/07_clone_vs_inplace.md) — 어느 것이 메모리를 함께 쓰는가
- [인덱싱과 슬라이싱](../ch06/tensors/08_indexing_slicing.md) · [재구성과 뷰](../ch06/tensors/09_reshaping_views.md)
- [산술 연산](../ch06/tensors/10_arithmetic_operations.md) · [브로드캐스팅](../ch06/tensors/11_broadcasting.md) · [논리 연산](../ch06/tensors/13_logical_operations.md)
- [텐서 비교](../ch06/tensors/17_tensor_comparison.md) — `==`, `equal`, `allclose` 가 다른 질문에 답한다
- [고윳값과 SVD](../ch06/tensors/15_eigenvalues_svd.md)

**학습에 쓰기**

- [야코비안과 고차 미분](../ch06/tensors/19_gradient_computation.md) · [사용자 정의 함수](../ch06/tensors/20_custom_functions.md)
- [간단한 뉴럴 네트워크](../ch06/tensors/21_simple_neural_network.md) · [최적화 기초](../ch06/tensors/22_optimization_basics.md)

**실무**

- [GPU 연산](../ch06/tensors/23_gpu_operations.md) · [혼합 정밀도](../ch06/tensors/24_mixed_precision.md) · [메모리 관리](../ch06/tensors/25_memory_management.md)
- [데이터 적재](../ch06/tensors/26_data_loading.md) · [모델 체크포인트](../ch06/tensors/27_model_checkpoint.md)
- [실전 예제](../ch06/tensors/28_real_world_example.md) — 제대로 돌아가는 학습 하나를 네 가지로 망가뜨려 본다

---

## 1.5 텐서 속성

텐서가 들고 있는 정보 — 모양, 자료형, 장치, 그리고 메모리 위의 배치.

- [메모리 배치와 스트라이드](../ch06/tensor_attrs/memory_layout_strides.md) — 저장소·스트라이드 모델. 뷰가 왜 공짜인지가 여기서 나온다
- [메모리와 저장소](../ch06/tensor_attrs/04_memory_and_storage.md) — `expand`, `repeat`, `channels_last`
- [형상, 자료형, 장치](../ch06/tensor_attrs/03_shape_dtype_device.md) · [생성 방법](../ch06/tensor_attrs/10_creation_methods.md)
- [데이터 관련 속성](../ch06/tensor_attrs/02_data_related.md) · [Autograd 관련 속성](../ch06/tensor_attrs/01_autograd_related.md)
- [Linear 웨이트 관례](../ch06/tensor_attrs/05_linear_weight_convention.md) — `nn.Linear` 의 웨이트가 왜 (출력, 입력) 인가
- [제자리 연산과 autograd 제약](../ch06/tensor_attrs/07_inplace_operations.md) — 언제 막히고 언제 조용히 틀리는가

**모양 바꾸기와 고르기**

- [이어 붙이기와 나누기](../ch06/tensor_attrs/13_concatenation_splitting.md) · [cat 시각화](../ch06/tensor_attrs/06_cat_visualization.md)
- [불리언 마스킹과 where](../ch06/tensor_attrs/09_indexing_masking.md)
- [축약과 통계](../ch06/tensor_attrs/11_reductions_statistics.md) — `correction` 인수가 무엇을 바꾸는가
- [행렬 연산](../ch06/tensor_attrs/14_matrix_operations.md) · [비교, 범위 제한, 정렬](../ch06/tensor_attrs/15_comparison_clamping_sorting.md)

---

## 1.6 자동 미분

경사 기반 학습을 떠받치는 엔진. 이 장에서 가장 긴 절이고, 그만큼 걸려 넘어질 자리가 많다.

**바탕**

- [경사 계산](../ch06/autograd/gradient_computation.md) — 야코비안, 벡터-야코비안 곱, 전진 모드와 후진 모드
- [스칼라에 대한 기본 역전파](../ch06/autograd/01_basic_scalar_backward.md) — 누가 `.grad` 를 받는가
- [Autograd 바닥부터 만들기](../ch06/autograd/autograd_from_scratch.md) — 직접 짜 보면 위상 정렬이 왜 필요한지가 드러난다

**경사를 쌓고 지우기**

- [경사 누적](../ch06/autograd/02_gradient_accumulation.md) — 기본값이 더하기인 까닭
- [경사를 0으로 만드는 세 가지 방법](../ch06/autograd/18_three_ways_to_zero_gradients.md)
- [큰 배치를 위한 경사 누적](../ch06/autograd/19_gradient_accumulation_for_large_batches.md) · [경사 누적을 쓰는 학습 루프](../ch06/autograd/21_training_loop_with_grad_accumulation.md)
- [경사 크기 조정이 중요한 이유](../ch06/autograd/20_why_gradient_scaling_matters.md)
- [retain_graph](../ch06/autograd/03_retain_graph.md) — 그래프를 남기는 것과 경사를 남기는 것은 다른 일이다

**추적을 멈추기**

- [경사 추적을 언제 멈출 것인가](../ch06/autograd/07_when_to_stop_tracking_grads.md) · [경사 추적 끄기](../ch06/autograd/08_turn_off_gradient_tracking.md)
- [detach와 detach_](../ch06/autograd/09_detach_vs_detach_.md) · [with torch.no_grad()](../ch06/autograd/10_with_torch_no_grad.md)
- [NumPy로 변환하기](../ch06/autograd/12_converting_to_numpy.md) · [파라메터 동결과 해제](../ch06/autograd/16_freeze_and_unfreeze_parameters.md)

**학습 루프 세우기**

- [작은 학습 단계](../ch06/autograd/04_mini_training_step.md) · [옵티마이저 방식의 갱신](../ch06/autograd/11_optimizer_like_updates.md)
- [선형 회귀](../ch06/autograd/05_complete_linear_regression.md)
- [레이어 출력하기](../ch06/autograd/13_print_layers.md) · [OrderedDict를 이용한 레이어 이름 짓기](../ch06/autograd/14_layer_naming_with_ordereddict.md) · [사용자 정의 Module을 이용한 레이어 이름 짓기](../ch06/autograd/15_layer_naming_with_custom_module.md)
- [함정과 점검 사항](../ch06/autograd/17_pitfalls_and_sanity_checks.md)

**벡터 역전파**

- [v를 이용한 벡터 역전파](../ch06/autograd/22_vector_backward_with_v.md) — `backward()` 에 주는 그 인수가 무엇인가
- [선형 사상의 VJP](../ch06/autograd/23_vector_backward_linear_case_1.md) · [비선형 전체 야코비안](../ch06/autograd/25_vector_backward_nonlinear_full_jacobian.md)
- [벡터 역전파의 함정](../ch06/autograd/26_pitfalls_of_vector_backward.md) · [배치 VJP](../ch06/autograd/27_batch_vjp.md)

---

## 1.7 경사 하강법

뉴럴 네트워크 학습의 한가운데에 있는 최적화 알고리즘.

**원리**

- [최급상승 방향으로서의 경사](../ch06/gradient_descent/steepest_direction.md) — 증명은 세 줄이고, 그 세 줄이 말하지 않는 것이 더 많다
- [러닝레이트와 이동 폭](../ch06/gradient_descent/learning_rate.md) — 수렴 경계, 조건수, 탐색기, 스케줄
- [배치, 미니배치, 확률적 경사 하강법](../ch06/gradient_descent/batch_minibatch_sgd.md) — 분산이 $1/B$ 로 주는가
- [볼록 최적화와 비볼록 최적화](../ch06/gradient_descent/convex_nonconvex.md) — 출발점을 바꿔 보면 차이가 드러난다

**직접 해 보기** (아래 일곱 쪽은 차례로 읽도록 되어 있다)

1. [NumPy로 직접 구현하는 경사 하강법](../ch06/gradient_descent/01_manual_gradient_numpy.md) — 자동 미분 없이 네 단계를 손으로
2. [PyTorch Autograd 기초](../ch06/gradient_descent/02_pytorch_autograd_basics.md) — `backward()` 가 대신해 주는 것과 그러지 않는 것
3. [간단한 선형 회귀](../ch06/gradient_descent/03_simple_linear_regression.md) — `nn.Linear` 와 옵티마이저, 그리고 지표 넷
4. [경사 하강법 시각화](../ch06/gradient_descent/04_visualizing_gradient_descent.md) — 무엇을 어떤 축에 그릴 것인가
5. [배치와 미니배치 SGD](../ch06/gradient_descent/01_batch_vs_minibatch_sgd.md) — 공정하게 견주는 법
6. [Adam 옵티마이저](../ch06/gradient_descent/01_adam_optimizer.md) — 손으로 짜서 `torch.optim.Adam` 과 맞춰 보기
7. [MNIST 뉴럴 네트워크](../ch06/gradient_descent/01_neural_network_mnist.md) — 전체 파이프라인과 거기서만 생기는 실수들

---

## 1.4~1.7을 읽는 법

**[3장](../ch03/index.md)을 읽다가 막혀서 왔다면** 막힌 대목에 해당하는 쪽만 보면 된다. 각 쪽은 그것만 읽어도 되도록 썼다.

**처음부터 읽는다면** 1.4 → 1.6 → 1.7 순서를 권한다. 1.5는 참고용에 가깝고, 메모리 배치가 궁금해질 때 돌아와 읽으면 된다.

**코드만 빨리 보고 싶다면** 1.7의 「직접 해 보기」 일곱 쪽이 그 자리다. NumPy 한 줄짜리 모델에서 시작해 MNIST 파이프라인까지 간다.

모든 쪽은 **프로그램 전체와 그 프로그램이 낸 출력 전체** 를 함께 싣는다. 쪽을 읽는 것만으로 충분하도록, 그리고 실린 수가 정말 그 코드에서 나온 것인지 확인할 수 있도록 하기 위해서다. 코드 블록을 그대로 갈무리하면 돌아간다 — 쪽에 실린 것이 프로그램의 전부이기 때문이다.

---

## 이 장을 마치면

기본적인 하드웨어와 데이터 처리 도구를 이해하게 된다. [2장](../ch02/index.md)은 뉴럴 네트워크 이전의 기계학습이 문제를 어떤 틀로 나누어 보았는지를 살피고, [3장](../ch03/index.md)에서 MNIST 하나를 붙들고 템플릿 학습에서 컨볼루션 뉴럴 네트워크까지 네 스텝을 걷는다.

PyTorch의 도구를 체계적으로 파고드는 것은 **이 장의 1.4~1.7절**이다. 1.1~1.3은 그 준비 운동이므로, 여기서 모든 것을 외우려 하지 않아도 된다. 뒤에서 막힐 때 돌아와 찾아보면 된다.
