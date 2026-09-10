# CPU와 GPU를 재어 보기

---

## 1. 학습 목표

이 절을 마치면 다음을 할 수 있게 된다.

- GPU를 재는 올바른 방법(동기화와 예열)을 알기
- 일감의 크기에 따라 GPU의 이득이 어떻게 달라지는지 재기
- GPU가 CPU보다 느려지는 구간이 있음을 확인하기
- 실제 학습에서 이 결과가 무엇을 뜻하는지 말하기

---

## 2. GPU를 재는 법

GPU를 재는 코드는 CPU와 같은 방식으로 쓰면 **틀린 값**이 나온다. 두 가지를 지켜야 한다.

**첫째, 동기화.** GPU는 명령을 받으면 곧바로 하지 않고 줄을 세워 두고 파이썬에는 제어를 돌려준다. 그래서 계산이 끝나기 전에 시계를 멈추게 된다. 반드시 `torch.cuda.synchronize()`(또는 `torch.mps.synchronize()`)로 끝날 때까지 기다려야 한다.

**둘째, 예열.** 처음 몇 번의 호출에는 커널을 준비하고 메모리를 잡는 일회성 값이 섞인다. 몇 번 돌려 버린 뒤에 재야 한다.

```python
import time
import torch

# 이 기계에서 쓸 수 있는 GPU를 고른다
if torch.cuda.is_available():
    gpu, gpu_sync = "cuda", torch.cuda.synchronize
elif torch.backends.mps.is_available():
    gpu, gpu_sync = "mps", torch.mps.synchronize
else:
    gpu, gpu_sync = None, lambda: None

def timed(fn, sync, reps=5, warmup=2):
    """예열한 뒤 reps번 돌려 한 번당 걸린 값을 돌려준다."""
    for _ in range(warmup):
        fn()
    sync()                                  # 예열이 끝나기를 기다린다
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    sync()                                  # 본 계산이 끝나기를 기다린다
    return (time.perf_counter() - start) / reps

print(f"PyTorch {torch.__version__}")
print(f"쓸 수 있는 GPU: {gpu if gpu else '없음 (CPU만)'}")
print(f"CPU 스레드 수 : {torch.get_num_threads()}")
```

**출력:**

```
PyTorch 2.5.1
쓸 수 있는 GPU: mps
CPU 스레드 수 : 4
```

---

## 3. 크기에 따라 달라지는 이득

행렬 곱을 크기를 바꾸어 가며 재어 본다.

```python
print(f"{'크기':>6} | {'CPU (ms)':>10} | {'GPU (ms)':>10} | {'배수':>7} | 어느 쪽이 빠른가")
print("-" * 62)

for n in (256, 512, 1024, 2048, 4096):
    a = torch.randn(n, n)
    b = torch.randn(n, n)

    t_cpu = timed(lambda: a @ b, lambda: None)

    if gpu:
        ag, bg = a.to(gpu), b.to(gpu)
        t_gpu = timed(lambda: ag @ bg, gpu_sync)
        ratio = t_cpu / t_gpu
        winner = "GPU" if ratio > 1 else "CPU"
        print(f"{n:>6} | {t_cpu*1e3:>10.2f} | {t_gpu*1e3:>10.2f} | {ratio:>6.1f}x | {winner}")
    else:
        print(f"{n:>6} | {t_cpu*1e3:>10.2f} | {'—':>10} | {'—':>7} | GPU 없음")
```

**출력:**

```
    크기 |   CPU (ms) |   GPU (ms) |      배수 | 어느 쪽이 빠른가
--------------------------------------------------------------
   256 |       0.03 |       0.19 |    0.2x | CPU
   512 |       0.18 |       0.53 |    0.3x | CPU
  1024 |       2.05 |       2.50 |    0.8x | CPU
  2048 |      19.28 |       6.27 |    3.1x | GPU
  4096 |     155.49 |      48.50 |    3.2x | GPU
```

표를 위에서 아래로 읽으면 이 절의 요점이 드러난다. **작은 행렬에서는 CPU가 이긴다.** 배수가 1보다 작다는 것은 GPU가 더 느리다는 뜻이다. 크기가 커질수록 배수가 올라가고 어느 지점에서 GPU가 앞선다.

까닭은 앞에서 본 대로이다. GPU에 일을 시키려면 준비하는 데 고정된 값이 드는데, 일감이 작으면 그 준비 값이 계산 값보다 크다.

$$\text{GPU 전체 값} = \underbrace{\text{준비}}_{\text{크기와 무관}} + \underbrace{\text{계산}}_{\text{크기에 비례}}$$

$n$이 작으면 앞의 항이 지배하고, $n$이 커지면 뒤의 항이 지배한다.

---

## 4. 계산의 종류에 따라서도 달라진다

행렬 곱은 GPU에 가장 유리한 계산이다. 원소 하나를 얻는 데 $n$번의 곱셈과 덧셈이 들기 때문이다. 반면 원소별 덧셈은 값 하나를 읽어 한 번 더하고 쓰면 끝이므로, 계산보다 **자료를 나르는 일**이 지배한다.

```python
n = 2048
a = torch.randn(n, n)
b = torch.randn(n, n)

print(f"{'연산':<22} | {'CPU (ms)':>10} | {'GPU (ms)':>10} | {'배수':>7}")
print("-" * 58)

ops = [
    ("행렬 곱  a @ b",      lambda x, y: x @ y),
    ("원소별 곱 a * b",     lambda x, y: x * y),
    ("원소별 합 a + b",     lambda x, y: x + y),
    ("지수      exp(a)",    lambda x, y: torch.exp(x)),
]

for name, op in ops:
    t_cpu = timed(lambda: op(a, b), lambda: None)
    if gpu:
        ag, bg = a.to(gpu), b.to(gpu)
        t_gpu = timed(lambda: op(ag, bg), gpu_sync)
        print(f"{name:<22} | {t_cpu*1e3:>10.2f} | {t_gpu*1e3:>10.2f} | {t_cpu/t_gpu:>6.1f}x")
    else:
        print(f"{name:<22} | {t_cpu*1e3:>10.2f} | {'—':>10} | {'—':>7}")
```

**출력:**

```
연산                     |   CPU (ms) |   GPU (ms) |      배수
----------------------------------------------------------
행렬 곱  a @ b            |      21.77 |       6.05 |    3.6x
원소별 곱 a * b            |       0.95 |       0.71 |    1.3x
원소별 합 a + b            |       1.18 |       0.75 |    1.6x
지수      exp(a)         |       2.54 |       0.43 |    5.9x
```

행렬 곱의 배수가 가장 크게 나온다. 신경망이 GPU에서 큰 이득을 보는 까닭이 여기에 있다. 신경망 계산의 대부분이 행렬 곱이기 때문이다.

---

## 5. 자료를 옮기는 값

계산이 아무리 빨라도 자료를 옮기는 데 시간을 다 쓰면 소용이 없다.

```python
if gpu:
    x = torch.randn(4096, 4096)                 # 64 MB

    t_move = timed(lambda: x.to(gpu), gpu_sync)
    xg = x.to(gpu)
    t_calc = timed(lambda: xg @ xg, gpu_sync)

    print(f"자료 크기            : {x.nbytes / 1e6:.0f} MB")
    print(f"CPU -> GPU 옮기기    : {t_move * 1e3:7.2f} ms")
    print(f"GPU에서 행렬 곱 한 번 : {t_calc * 1e3:7.2f} ms")
    print(f"옮기는 값 / 계산 값   : {t_move / t_calc:.3f}")
    print()
    if t_move / t_calc < 0.2:
        print("옮기는 값이 계산에 견주어 작다. 이 기계에서는 병목이 아니다.")
    else:
        print("옮기는 값이 만만치 않다. 자료를 GPU에 올려 둔 채로 두어야 한다.")
```

**출력:**

```
자료 크기            : 67 MB
CPU -> GPU 옮기기    :    3.30 ms
GPU에서 행렬 곱 한 번 :   48.02 ms
옮기는 값 / 계산 값   : 0.069

옮기는 값이 계산에 견주어 작다. 이 기계에서는 병목이 아니다.
```

!!! warning "이 숫자는 기계마다 다르다"
    이 책의 값들은 **애플 M3**에서 잰 것이다. 애플 실리콘은 CPU와 GPU가 메모리를 함께 쓰므로 옮기는 값이 유난히 싸다.

    PCIe로 이어진 NVIDIA 독립 GPU라면 옮기는 값이 훨씬 비싸고, 대신 계산은 훨씬 빨라 배수가 수십 배까지 나온다. 위 코드를 **자기 기계에서 그대로 돌려 보는 것**이 남의 숫자를 외우는 것보다 낫다.

---

## 6. 실제 학습에서 뜻하는 것

재어 본 결과가 실무의 규칙으로 이어진다.

| 잰 결과 | 실무의 규칙 |
|---|---|
| 작은 일감에서 GPU가 느리다 | 묶음을 너무 작게 잡지 마라 |
| 행렬 곱에서 이득이 가장 크다 | 층을 넓게 잡으면 GPU가 잘 쓰인다 |
| 옮기는 값이 든다 | 학습 루프 안에서 `.cpu()`를 부르지 마라 |
| 준비 값이 고정으로 든다 | 작은 연산 여럿보다 큰 연산 하나가 낫다 |

특히 마지막 줄이 중요하다. GPU에서는 연산을 **합치는 것**(fusion)이 이득이 된다. 작은 커널을 여러 번 띄우는 대신 한 번에 처리하면 준비 값을 한 번만 치르기 때문이다.

---

## 연습문제

**연습문제 1.**
위 표에서 GPU가 CPU를 앞서기 시작하는 크기를 자기 기계에서 찾아라. 그 크기가 뜻하는 바는 무엇인가?

??? success "연습문제 1 풀이"
    그 크기가 이 기계에서 GPU를 쓸 값어치가 생기는 문턱이다. 문턱보다 작은 일감만 다룬다면 GPU는 도움이 되지 않는다.

    실제 학습에서는 묶음 크기와 층의 너비가 이 문턱을 넘도록 잡아야 GPU가 제 몫을 한다. 묶음 8로 작은 모델을 학습시키면서 "GPU가 느리다"고 하는 경우가 흔한데, 그것은 GPU 탓이 아니다.

---

**연습문제 2.**
`timed`에서 `sync()` 호출을 빼면 GPU 시간이 어떻게 나오겠는가?

??? success "연습문제 2 풀이"
    터무니없이 짧게 나온다. 계산이 끝나기 전에 시계를 멈추기 때문이다. 극단적으로는 0에 가까운 값이 나와 "GPU가 1000배 빠르다" 같은 잘못된 결론에 이른다.

    GPU 성능을 다룬 글에서 믿기 어려운 배수를 보면 동기화를 했는지부터 의심해 볼 만하다.

---

**연습문제 3.**
예열(`warmup`)을 하지 않으면 어떤 일이 생기는가?

??? success "연습문제 3 풀이"
    첫 호출에 커널을 컴파일하고 메모리를 잡는 일회성 값이 섞여 실제보다 느리게 나온다. 특히 GPU에서 이 차이가 크다.

    반복 횟수를 늘리면 이 값이 평균에 묻혀 옅어지지만, 아예 버리는 편이 정확하다.

## 정리하며

**다룬 것** — 장치를 바꾸는 일의 값어치를 직접 재기

GPU를 잴 때는 **동기화**와 **예열**을 지켜야 한다. 이 둘을 빠뜨린 측정은 믿을 수 없다.

재어 보면 GPU는 만능이 아니다. 일감이 작으면 준비하는 값이 계산 값을 넘어 CPU보다 느리고, 크기가 커져야 앞선다. 계산의 종류도 중요해서, 행렬 곱처럼 읽어 온 값 하나로 여러 번 계산하는 연산에서 이득이 가장 크다.

신경망이 GPU에서 큰 이득을 보는 까닭은 그 계산이 대부분 **크고 독립적인 행렬 곱**이기 때문이다. 이 성질이 딥러닝과 GPU를 묶어 놓았다.

앞의 연습문제 3개로 직접 확인할 수 있다.
