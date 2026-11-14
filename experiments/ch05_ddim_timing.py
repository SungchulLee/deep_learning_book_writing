"""DDIM 샘플링 시간이 같은 코드를 다시 돌릴 때 얼마나 흔들리는지 잰다.

docs/ch05/diffusion/02_ddim.md 연습문제 3 풀이가 싣는 수를 만든다.
쪽의 U-Net 과 sample_ddim 을 그 쪽의 코드 블록에서 그대로 읽어 온다. 시간은 웨이트 값과 무관하므로
학습하지 않은 U-Net 으로 잰다(체크포인트 mnist_ddpm.pt 가 저장소에 없다).
쪽의 표와 달리 평가 분류기에 읽히는 시간은 넣지 않고 샘플링만 잰다.

실행: 저장소 뿌리에서  python experiments/ch05_ddim_timing.py
"""
import re, sys, time
import torch, torch.nn as nn, torch.nn.functional as F

md = open("docs/ch05/diffusion/02_ddim.md").read()
code = md[md.index("```python") + 9:]
code = code[:code.index("```")]
a = code[code.index("# === 정방향 과정"):code.index("net = UNet()")]
b = code[code.index("@torch.no_grad()\ndef sample_ddim"):code.index('if __name__ == "__main__":')]
# 쪽의 코드 블록에서 두 조각만 잘라 온다: (a) 스케줄 상수와 Block·UNet 정의,
# (b) sample_ddim. 자료 읽기·평가 분류기·체크포인트 읽기는 가져오지 않는다.
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEED, T, N_SAMPLE = 42, 1000, 5000
exec(a); exec(b)
# [조건 1] 학습하지 않은 U-Net 이다. 쪽은 mnist_ddpm.pt 를 읽지만 그 파일이
# 저장소에 없다. 순전파 한 번에 드는 시간은 웨이트의 값이 아니라 구조와 입력
# 크기로 정해지므로, 같은 구조를 무작위 초기화한 채로 잰다. 따라서 여기서 나온
# 그림은 숫자가 아니고, 이 스크립트는 품질 지표를 하나도 내지 않는다.
torch.manual_seed(0)
net = UNet().to(device).eval()
# 첫 호출에는 MPS 커널 준비 시간이 섞이므로 재기 전에 한 번 버린다
sample_ddim(net, 2, n=500)                     # 워밍업
for steps, reps in ((10, 6), (20, 5)):
    ts = []
    for _ in range(reps):
        # [조건 2] 재는 것은 sample_ddim 한 번, 곧 5,000장을 뽑는 시간뿐이다.
        # 쪽의 표(26s, 69s, ...)는 같은 구간에 evaluate() — 평가 분류기에 5,000장을
        # 읽히는 시간 — 까지 넣어 잰 것이라 이 수와 곧장 견줄 수 없다.
        # 견줄 것은 되풀이 사이의 폭과 스텝당 시간이 한결같은지다.
        t0 = time.time(); sample_ddim(net, steps)
        # MPS 는 비동기라, 끝난 것을 확인한 뒤에 시계를 읽는다
        if device.type == "mps": torch.mps.synchronize()
        ts.append(time.time() - t0)
    # [조건 3] 한 기계에서 다른 학습을 돌리지 않는 동안 잰 것이다. 폭(최대/최소)은
    # 그 형편에서의 값이고, 다른 일이 함께 돌면 더 벌어진다.
    m = sum(ts) / len(ts)
    print(f"{steps}스텝 x{reps}: " + " ".join(f"{t:.1f}" for t in ts)
          + f"  평균 {m:.1f}s  최소~최대 {min(ts):.1f}~{max(ts):.1f}"
          + f"  (최대/최소 {max(ts)/min(ts):.3f}, 스텝당 {m/steps:.2f}s)", flush=True)
