# 7장: PyTorch 기초

[3장](../ch03/index.md)은 MNIST 하나로 네 모델을 짓고 견주었다. 그 코드가 쓰던 도구 — 텐서, 자동 미분, 경사 하강법 — 를 이 장이 체계적으로 다룬다.

3장을 먼저 두고 이 장을 뒤에 둔 데에는 뜻이 있다. `loss.backward()`가 무엇을 하는지 모르는 채로도 뉴럴 넷트웍 하나를 끝까지 학습시켜 보는 편이, 미분 규칙을 먼저 외우는 것보다 배우기 쉽다고 보기 때문이다. 이 장은 그렇게 직접 써 본 도구를 다시 열어 안을 들여다보는 자리다.

처음부터 끝까지 읽어도 되고, 앞 장의 예제를 읽다가 막히는 대목이 나올 때 찾아보아도 된다.

---

## 7.1 텐서

PyTorch의 기본 자료구조. 만드는 법부터 GPU에 올리고 저장하는 법까지.

**만들기**

- [스칼라를 텐서로](tensors/01_scalar_to_tensor.md) · [리스트를 텐서로](tensors/02_list_to_tensor.md) · [범위를 텐서로](tensors/03_range_to_tensor.md)
- [NumPy 배열을 텐서로](tensors/04_numpy_array_to_tensor.md) · [Pandas를 텐서로](tensors/05_pandas_to_tensor.md)
- [팩토리 함수](tensors/06_factory_functions.md) — `zeros`, `ones`, `empty`, `rand` 가 각각 무엇을 보장하는가

**다루기**

- [별칭, 뷰, clone의 차이](tensors/07_clone_vs_inplace.md) — 어느 것이 메모리를 함께 쓰는가
- [인덱싱과 슬라이싱](tensors/08_indexing_slicing.md) · [재구성과 뷰](tensors/09_reshaping_views.md)
- [산술 연산](tensors/10_arithmetic_operations.md) · [브로드캐스팅](tensors/11_broadcasting.md) · [논리 연산](tensors/13_logical_operations.md)
- [텐서 비교](tensors/17_tensor_comparison.md) — `==`, `equal`, `allclose` 가 다른 질문에 답한다
- [고윳값과 SVD](tensors/15_eigenvalues_svd.md)

**학습에 쓰기**

- [야코비안과 고차 미분](tensors/19_gradient_computation.md) · [사용자 정의 함수](tensors/20_custom_functions.md)
- [간단한 뉴럴 넷트웍](tensors/21_simple_neural_network.md) · [최적화 기초](tensors/22_optimization_basics.md)

**실무**

- [GPU 연산](tensors/23_gpu_operations.md) · [혼합 정밀도](tensors/24_mixed_precision.md) · [메모리 관리](tensors/25_memory_management.md)
- [데이터 적재](tensors/26_data_loading.md) · [모델 체크포인트](tensors/27_model_checkpoint.md)
- [실전 예제](tensors/28_real_world_example.md) — 제대로 돌아가는 학습 하나를 네 가지로 망가뜨려 본다

---

## 7.2 텐서 속성

텐서가 들고 있는 정보 — 모양, 자료형, 장치, 그리고 메모리 위의 배치.

- [메모리 배치와 스트라이드](tensor_attrs/memory_layout_strides.md) — 저장소·스트라이드 모델. 뷰가 왜 공짜인지가 여기서 나온다
- [메모리와 저장소](tensor_attrs/04_memory_and_storage.md) — `expand`, `repeat`, `channels_last`
- [형상, 자료형, 장치](tensor_attrs/03_shape_dtype_device.md) · [생성 방법](tensor_attrs/10_creation_methods.md)
- [데이터 관련 속성](tensor_attrs/02_data_related.md) · [Autograd 관련 속성](tensor_attrs/01_autograd_related.md)
- [Linear 웨이트 관례](tensor_attrs/05_linear_weight_convention.md) — `nn.Linear` 의 웨이트가 왜 (출력, 입력) 인가
- [제자리 연산과 autograd 제약](tensor_attrs/07_inplace_operations.md) — 언제 막히고 언제 조용히 틀리는가

**모양 바꾸기와 고르기**

- [이어 붙이기와 나누기](tensor_attrs/13_concatenation_splitting.md) · [cat 시각화](tensor_attrs/06_cat_visualization.md)
- [불리언 마스킹과 where](tensor_attrs/09_indexing_masking.md)
- [축약과 통계](tensor_attrs/11_reductions_statistics.md) — `correction` 인수가 무엇을 바꾸는가
- [행렬 연산](tensor_attrs/14_matrix_operations.md) · [비교, 범위 제한, 정렬](tensor_attrs/15_comparison_clamping_sorting.md)

---

## 7.3 자동 미분

경사 기반 학습을 떠받치는 엔진. 이 장에서 가장 긴 절이고, 그만큼 걸려 넘어질 자리가 많다.

**바탕**

- [경사 계산](autograd/gradient_computation.md) — 야코비안, 벡터-야코비안 곱, 전진 모드와 후진 모드
- [스칼라에 대한 기본 역전파](autograd/01_basic_scalar_backward.md) — 누가 `.grad` 를 받는가
- [Autograd 바닥부터 만들기](autograd/autograd_from_scratch.md) — 직접 짜 보면 위상 정렬이 왜 필요한지가 드러난다

**경사를 쌓고 지우기**

- [경사 누적](autograd/02_gradient_accumulation.md) — 기본값이 더하기인 까닭
- [경사를 0으로 만드는 세 가지 방법](autograd/18_three_ways_to_zero_gradients.md)
- [큰 배치를 위한 경사 누적](autograd/19_gradient_accumulation_for_large_batches.md) · [경사 누적을 쓰는 학습 루프](autograd/21_training_loop_with_grad_accumulation.md)
- [경사 크기 조정이 중요한 이유](autograd/20_why_gradient_scaling_matters.md)
- [retain_graph](autograd/03_retain_graph.md) — 그래프를 남기는 것과 경사를 남기는 것은 다른 일이다

**추적을 멈추기**

- [경사 추적을 언제 멈출 것인가](autograd/07_when_to_stop_tracking_grads.md) · [경사 추적 끄기](autograd/08_turn_off_gradient_tracking.md)
- [detach와 detach_](autograd/09_detach_vs_detach_.md) · [with torch.no_grad()](autograd/10_with_torch_no_grad.md)
- [NumPy로 변환하기](autograd/12_converting_to_numpy.md) · [파라메터 동결과 해제](autograd/16_freeze_and_unfreeze_parameters.md)

**학습 루프 세우기**

- [작은 학습 단계](autograd/04_mini_training_step.md) · [최적화기 방식의 갱신](autograd/11_optimizer_like_updates.md)
- [완전한 선형 회귀](autograd/05_complete_linear_regression.md) · [Module을 이용한 선형 회귀](autograd/06_linear_regression_with_module.md)
- [레이어 출력하기](autograd/13_print_layers.md) · [OrderedDict를 이용한 레이어 이름 짓기](autograd/14_layer_naming_with_ordereddict.md) · [사용자 정의 Module을 이용한 레이어 이름 짓기](autograd/15_layer_naming_with_custom_module.md)
- [함정과 점검 사항](autograd/17_pitfalls_and_sanity_checks.md)

**벡터 역전파**

- [v를 이용한 벡터 역전파](autograd/22_vector_backward_with_v.md) — `backward()` 에 주는 그 인수가 무엇인가
- [선형 사례 1](autograd/23_vector_backward_linear_case_1.md) · [선형 사례 2](autograd/24_vector_backward_linear_case_2.md) · [비선형 전체 야코비안](autograd/25_vector_backward_nonlinear_full_jacobian.md)
- [벡터 역전파의 함정](autograd/26_pitfalls_of_vector_backward.md) · [배치 VJP](autograd/27_batch_vjp.md)

---

## 7.4 경사 하강법

뉴럴 넷트웍 학습의 한가운데에 있는 최적화 알고리즘.

**원리**

- [최급상승 방향으로서의 경사](gradient_descent/steepest_direction.md) — 증명은 세 줄이고, 그 세 줄이 말하지 않는 것이 더 많다
- [러닝레이트와 이동 폭](gradient_descent/learning_rate.md) — 수렴 경계, 조건수, 탐색기, 스케줄
- [배치, 미니배치, 확률적 경사 하강법](gradient_descent/batch_minibatch_sgd.md) — 분산이 $1/B$ 로 주는가
- [볼록 최적화와 비볼록 최적화](gradient_descent/convex_nonconvex.md) — 출발점을 바꿔 보면 차이가 드러난다

**직접 해 보기** (아래 일곱 쪽은 차례로 읽도록 되어 있다)

1. [NumPy로 직접 구현하는 경사 하강법](gradient_descent/01_manual_gradient_numpy.md) — 자동 미분 없이 네 단계를 손으로
2. [PyTorch Autograd 기초](gradient_descent/02_pytorch_autograd_basics.md) — `backward()` 가 대신해 주는 것과 그러지 않는 것
3. [간단한 선형 회귀](gradient_descent/03_simple_linear_regression.md) — `nn.Linear` 와 옵티마이저, 그리고 지표 넷
4. [경사 하강법 시각화](gradient_descent/04_visualizing_gradient_descent.md) — 무엇을 어떤 축에 그릴 것인가
5. [배치와 미니배치 SGD](gradient_descent/01_batch_vs_minibatch_sgd.md) — 공정하게 견주는 법
6. [Adam 최적화기](gradient_descent/01_adam_optimizer.md) — 손으로 짜서 `torch.optim.Adam` 과 맞춰 보기
7. [MNIST 뉴럴 넷트웍](gradient_descent/01_neural_network_mnist.md) — 전체 파이프라인과 거기서만 생기는 실수들

---

## 이 장을 읽는 법

**3장을 읽다가 막혀서 왔다면** 막힌 대목에 해당하는 쪽만 보면 된다. 각 쪽은 그것만 읽어도 되도록 썼다.

**처음부터 읽는다면** 7.1 → 7.3 → 7.4 순서를 권한다. 7.2는 참고용에 가깝고, 메모리 배치가 궁금해질 때 돌아와 읽으면 된다.

**코드만 빨리 보고 싶다면** 7.4의 「직접 해 보기」 일곱 쪽이 그 자리다. NumPy 한 줄짜리 모델에서 시작해 MNIST 파이프라인까지 간다.

모든 쪽은 **프로그램 전체와 그 프로그램이 낸 출력 전체** 를 함께 싣는다. 쪽을 읽는 것만으로 충분하도록, 그리고 실린 수가 정말 그 코드에서 나온 것인지 확인할 수 있도록 하기 위해서다. 7.2 이후의 쪽들에는 같은 이름의 `.py` 파일이 옆에 있어 내려받아 그대로 돌릴 수 있다.
