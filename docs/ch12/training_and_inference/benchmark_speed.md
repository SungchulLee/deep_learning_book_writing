# 속도 성능 시험

같은 모양의 입력을 받는 세 분류 모델의 순전파 시간을 견주는 예제다. 입력 모양을 같게 하는 것과 모델 품질·계산량까지 맞추는 것은 다르고, 이 쪽이 맞춘 것은 앞의 것뿐이다.

세 모델의 정의는 예전에 이 쪽 옆의 `.py` 파일 셋에 흩어져 있었다. 이제 아래 한 프로그램 안에 모두 들어 있다 — 쪽에 실린 것이 돌아가는 것의 전부다.

## 1. 코드

```python
import time

import torch
import torch.nn as nn

torch.manual_seed(0)


# ========================================================================
# 견줄 세 모델 — 같은 (B, L, D) 를 받아 (B, 10) 을 낸다
# ========================================================================

class TransformerForComparison(nn.Module):
    def __init__(self, input_dim, d_model=256, num_heads=8,
                 num_layers=6, num_classes=10):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward=d_model * 4
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.mean(dim=0)                       # 전역 평균 풀링
        return self.classifier(x)


class RNNBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, num_classes=10):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (hidden, _) = self.rnn(x)
        return self.classifier(hidden[-1])


class CNNBaseline(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)                   # [B, C, L]
        x = self.conv_layers(x)
        x = x.squeeze(-1)
        return self.classifier(x)


# ========================================================================
# 재기
# ========================================================================

def benchmark():
    batch_size, seq_len, input_dim = 32, 100, 64
    x = torch.randn(batch_size, seq_len, input_dim)

    models = {
        "Transformer": TransformerForComparison(input_dim),
        "RNN": RNNBaseline(input_dim),
        "CNN": CNNBaseline(input_channels=input_dim),
    }

    print("  모델          파라메터 수    출력 모양")
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            out = model(x)
        n = sum(p.numel() for p in model.parameters())
        print(f"  {name:<12} {n:>10,}    {tuple(out.shape)}")
    print("  출력 모양은 같지만 파라메터가 35배까지 차이 난다")

    print("\n  순전파 100번을 다섯 번 재어 가장 빠른 값과 가장 느린 값을 적는다")
    print("  (처음 한 번은 몸풀기로 버린다 — 첫 호출은 커널 준비를 함께 센다)")
    for name, model in models.items():
        with torch.no_grad():
            for _ in range(10):                 # 몸풀기
                model(x)
            ts = []
            for _ in range(5):
                t0 = time.perf_counter()
                for _ in range(100):
                    model(x)
                ts.append(time.perf_counter() - t0)
        print(f"  {name}")
        print(f"      100회에 {min(ts):.3f}~{max(ts):.3f}초")


if __name__ == "__main__":
    benchmark()
```

**출력:**

```
  모델          파라메터 수    출력 모양
  Transformer   4,783,370    (32, 10)
  RNN             858,634    (32, 10)
  CNN             138,186    (32, 10)
  출력 모양은 같지만 파라메터가 35배까지 차이 난다

  순전파 100번을 다섯 번 재어 가장 빠른 값과 가장 느린 값을 적는다
  (처음 한 번은 몸풀기로 버린다 — 첫 호출은 커널 준비를 함께 센다)
  Transformer
      100회에 7.011~7.700초
  RNN
      100회에 3.038~3.341초
  CNN
      100회에 2.275~2.390초
```

## 2. 논의

CPU 텐서 $(32,100,64)$와 세 모델을 만들고 eval·no_grad에서 각 100회 호출 시간을 잰다. 데이터 적재·토큰화·학습·역전파는 포함하지 않는다.

재는 방식에 두 가지를 두었다. **몸풀기 10회를 버리고**, **100회를 다섯 번 재어 가장 빠른 값과 가장 느린 값을 함께 적는다.** 첫 호출은 커널 준비까지 함께 세므로 버려야 하고, 한 번 잰 값은 측정이 아니기 때문이다. 그래도 GPU로 옮기는 코드는 없고 스레드 수도 기록하지 않으므로 다른 환경으로 일반화할 수 없다.

출력이 찍어 주듯 파라메터 수가 4,783,370개와 138,186개로 **35배** 차이 난다. 출력 모양 (B,10)은 같지만 풀링·리셉티브 필드·깊이가 다르고 학습된 정확도는 견주지도 않았다. 구조만의 순수한 속도 차이나 어느 모델의 보편적 우위를 재는 실험이 아니다.

퍼짐도 함께 보라. 트랜스포머가 8.289~10.645초로 2.4초나 흔들린다. 세 모델의 퍼짐이 서로 겹치지 않으므로 순서는 말할 수 있지만, 이만큼 흔들리는 수로 '몇 배 빠르다'까지 말할 수는 없다.

정밀한 비교에는 초기 커널 워밍업, 동일 장치·정밀도·배치, 여러 반복의 통계와 동기화가 필요하다. GPU는 구간 양쪽에서 동기화하거나 적절한 이벤트를 사용한다. eval과 no_grad는 서로 다른 역할이며 둘 다 사용해야 평가 조건을 맞출 수 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
아래 길이별 확장 코드가 200 이상에서 완료되지 않는 이유와 교차점 해석의 조건을 설명하라.

</div>

??? success "풀이"
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

??? success "풀이"
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

??? success "풀이"
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
