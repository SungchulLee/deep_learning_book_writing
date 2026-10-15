# 속도 성능 시험

고정된 실수 입력에 대한 세 분류 모델의 순방향 시간을 비교하는 예제다. 기존 출력은 특정 환경의 예시로 보존하며 이번 편집에서 재측정하지 않았다. 입력 모양을 같게 하는 것과 모델 품질·계산량까지 맞추는 것은 다르다.

## 1. 코드

```python
import torch
import time
from transformer_model import TransformerForComparison
from rnn_baseline import RNNBaseline
from cnn_baseline import CNNBaseline


def benchmark():
    batch_size = 32
    seq_len = 100
    input_dim = 64

    x = torch.randn(batch_size, seq_len, input_dim)

    models = {
        'Transformer': TransformerForComparison(input_dim),
        'RNN': RNNBaseline(input_dim),
        'CNN': CNNBaseline(input_channels=64)
    }

    for name, model in models.items():
        model.eval()
        start = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = model(x)
        elapsed = time.time() - start
        print(f"{name}: {elapsed:.3f}s for 100 forward passes")


if __name__ == '__main__':
    benchmark()
```

**출력:**

```
Transformer: 8.967s for 100 forward passes
RNN: 3.472s for 100 forward passes
CNN: 2.786s for 100 forward passes
```

## 2. 논의

기본 함수는 CPU 텐서 $(32,100,64)$와 세 모델을 만들고 eval·no_grad에서 각 100회 호출 시간을 잰다. 데이터 로딩·토큰화·학습·역전파는 포함하지 않는다. GPU로 옮기는 코드도 없고 워밍업·반복 구간 분산·스레드 수를 기록하지 않아 결과를 다른 환경으로 일반화할 수 없다.

계수는 트랜스포머 4,783,370개, LSTM 858,634개, CNN 138,186개로 크게 다르다. 출력 (B,10)은 같지만 풀링·수용 영역·깊이가 다르며 학습된 정확도도 비교하지 않는다. 구조만의 순수한 속도 차이나 어느 모델의 보편적 우위를 측정하는 실험은 아니다.

정밀한 비교에는 초기 커널 워밍업, 동일 장치·정밀도·배치, 여러 반복의 통계와 동기화가 필요하다. GPU는 구간 양쪽에서 동기화하거나 적절한 이벤트를 사용한다. eval과 no_grad는 서로 다른 역할이며 둘 다 사용해야 평가 조건을 맞출 수 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
아래 길이별 확장 코드가 200 이상에서 완료되지 않는 이유와 교차점 해석의 조건을 설명하라.

</div>

??? success "연습문제 1 풀이"
    ```python
    import matplotlib.pyplot as plt

    seq_lengths = [50, 100, 200, 500, 1000]
    results = {name: [] for name in ['Transformer', 'RNN', 'CNN']}

    for seq_len in seq_lengths:
        x = torch.randn(32, seq_len, 64)
        for name, ModelClass in [
            ('Transformer', lambda: TransformerForComparison(64)),
            ('RNN', lambda: RNNBaseline(64)),
            ('CNN', lambda: CNNBaseline(input_channels=64)),
        ]:
            model = ModelClass()
            model.eval()
            start = time.time()
            with torch.no_grad():
                for _ in range(50):
                    model(x)
            results[name].append(time.time() - start)

    for name, times in results.items():
        plt.plot(seq_lengths, times, label=name)
    plt.xlabel("Sequence Length")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.show()
    ```
    TransformerForComparison의 학습 위치 표는 100개이므로 길이 200부터 덧셈 모양 오류가 난다. 먼저 위치 처리와 비교 조건을 정해야 한다. CPU 길이 500~1000에서 속도 순위가 바뀐다는 보편적인 기준은 없고 이 예제의 출력도 그 실험을 수행하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
아래 CUDA 측정 조각의 외부 변수·장치 조건과 동기화를 검토하라.

</div>

??? success "연습문제 2 풀이"
    ```python
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for name, model in models.items():
        model = model.to(device).eval()
        x_gpu = x.to(device)

        # 워밍업
        with torch.no_grad():
            for _ in range(10):
                model(x_gpu)
        torch.cuda.synchronize()

        start = time.time()
        with torch.no_grad():
            for _ in range(100):
                model(x_gpu)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"{name} (GPU): {elapsed:.3f}s")
    ```
    models와 x는 기본 benchmark 함수의 지역 변수라 이 조각에서 바로 참조할 수 없다. 별도로 구성해야 하며, CUDA가 없을 때 CPU를 선택해도 무조건 cuda.synchronize를 호출하는 부분은 처리해야 한다. GPU 측정은 워밍업 후 시작·끝에서 동기화하되 상대 속도는 계수·입력·커널에 따라 달라진다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
아래 CUDA 최고 메모리 측정에서 모델 간 비교를 위해 보완할 점을 설명하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    for name, model in models.items():
        model = model.to(device).eval()
        x_gpu = x.to(device)
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            model(x_gpu)

        memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"{name}: {memory_mb:.1f} MB peak memory")
    ```
    max_memory_allocated는 현재 프로세스의 CUDA 할당 최고치로 해당 모델 하나의 추가 활성만 뜻하지 않는다. 앞 모델들이 GPU에 남아 있으면 기준량이 섞이고 기존 출력 참조도 영향을 준다. 현재 할당량·최고치·예약 메모리를 구분하고 모델별 독립 상태에서 측정한다. 장치와 외부 변수 조건도 먼저 맞춰야 하며 구조별 메모리 순위를 단정하지 않는다.

## 정리하며

이 벤치마크는 특정 크기의 서로 다른 분류 모델을 CPU에서 반복 호출하는 예제다. 기존 출력의 범위를 넘는 속도·메모리 순위를 주장하지 않고, 길이 제한·외부 변수·장치·워밍업·동기화와 기준 할당량을 함께 확인한다.
