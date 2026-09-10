# 속도 성능 시험

트랜스포머와 순환 신경망과 합성곱 신경망의 추론 속도를 견주어 보면 저마다의 계산 맞바꿈을 알 수 있다. 트랜스포머는 병렬로 하는 자기 주의로 먼 거리 의존을 잡아내는 데 뛰어나지만 그 대가로 수열 길이에 대해 이차 비용을 치른다. 이 성능 시험은 앞먹임 처리량을 재어 그 차이를 실제 수치로 보인다.

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

## 2. 논의

이 성능 시험은 고정된 입력 텐서를 만들고 기울기 계산을 끈 채 모형마다 앞먹임을 100번 돌린다. `torch.no_grad()`로 기울기를 끄면 계산 그래프를 세우고 중간 활성을 담아 두는 짐이 사라져 순수한 추론 속도를 더 정확히 잴 수 있다.

순환 신경망은 수열을 한 걸음씩 처리하므로 비용이 수열 길이에 비례하지만 본디 차례를 따라야 해서 시간 단계에 걸쳐 병렬로 할 수 없다. 합성곱 신경망은 모든 자리에 합성곱을 병렬로 적용하여 짧은 수열에서 처리량이 뛰어나지만 받는 영역이 핵의 크기와 깊이에 매인다. 트랜스포머는 모든 자리에 한꺼번에 주의하여 병렬성을 얻지만 수열 길이에 대해 $O(n^2)$ 비용을 치른다.

GPU에서는 트랜스포머의 연산이 병렬로 하기 좋은 행렬 곱이므로 어지간한 길이의 수열에서 대체로 순환 신경망을 앞선다. 다만 아주 긴 수열에서는 이차인 주의 비용 때문에 트랜스포머가 다른 것보다 느려질 수 있다. 합성곱 신경망은 간단하고 잘 다듬어진 합성곱 연산 덕분에 길이가 고정된 입력에서 추론 시간이 가장 빠른 경우가 많다.

## 연습문제

**연습문제 1.**
`seq_len`을 `[50, 100, 200, 500, 1000]`으로 바꾸어 가며 성능 시험을 넓히고 모형마다 실행 시간을 수열 길이의 함수로 그려라. 어느 지점에서 트랜스포머가 순환 신경망보다 느려지는가?

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
    트랜스포머가 순환 신경망보다 느려지는 갈림 지점은 하드웨어에 따라 다르지만 CPU에서는 대체로 수열 길이 500~1000쯤에서 나타난다.

---

**연습문제 2.**
(쓸 수 있다면) 모형과 데이터를 CUDA로 옮겨 GPU 성능 시험을 더하라. 시간을 재기 전에 `torch.cuda.synchronize()`로 제대로 맞추어라. GPU에서 상대적인 속도는 어떻게 달라지는가?

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
    GPU에서는 트랜스포머가 병렬화의 덕을 가장 많이 보아 어지간한 길이의 수열에서 대체로 가장 빠르다. 순환 신경망은 차례를 따르는 성질 때문에 GPU를 잘 쓰지 못해 덜 이롭다. GPU 연산은 비동기이므로 `torch.cuda.synchronize()`가 꼭 필요하다.

---

**연습문제 3.**
모형마다 앞먹임 앞뒤로 `torch.cuda.max_memory_allocated()`를 살펴 속도와 함께 기억 사용량도 재어라. 어느 구조가 기억을 가장 아끼는가?

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
    순환 신경망은 한 번에 한 걸음씩 처리하고 $n \times n$ 주의 행렬을 통째로 담아 두지 않으므로 대체로 기억을 가장 아낀다. 트랜스포머는 주의 점수에 $O(n^2)$의 기억이 들어 긴 수열에서 기억을 가장 많이 쓴다. 합성곱 신경망은 그 사이로, 기억이 수열 길이에 비례해 는다.

## 정리하며

**다룬 것** — 속도 성능 시험

이 성능 시험은 고정된 입력 텐서를 만들고 기울기 계산을 끈 채 모형마다 앞먹임을 100번 돌린다.

앞의 연습문제 3개로 직접 확인할 수 있다.
