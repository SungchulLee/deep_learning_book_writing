# IMDB 텍스트 분류

이 모듈은 분류라는 맥락에서 IMDB 텍스트 분류의 구현을 제시한다.

이 튜토리얼은 PyTorch에서 로지스틱 회귀에 대한 기초적인 이해를 쌓는다. 코드를 따라가 보면 모델을 세우고, 손실 함수를 정의하고, 경사 하강법으로 학습시키고, 분류 과제에서 성능을 평가하는 법을 알 수 있다.

## 코드

```python

#!/usr/bin/env python3
"""
가장 단출한 IMDb 글월 가름 과제(맨바닥부터, torchtext 없이).
- 기본으로 큰 영화 평 자료 묶음(aclImdb)을 ./data에 내려받는다
- 단순한 낱말 사전을 짓는다
- 평균 묻힘 가름개를 익힌다(PyTorch)
쓰는 법(보기):
    python main.py --epochs 2 --batch_size 64 --max_len 256
"""
from dataclasses import dataclass
from pathlib import Path
import argparse
import torch

# ========================================================================
# 메인
# ========================================================================

from imdb.download_data import download_imdb
from imdb.load_data import make_dataloaders
from imdb.model import AverageEmbeddingsClassifier
from imdb.train import train_loop, evaluate_loop, save_checkpoint

@dataclass
class Config:
    data_dir: Path = Path("./data")
    epochs: int = 2
    batch_size: int = 64
    lr: float = 1e-3
    seed: int = 0
    embed_dim: int = 128
    max_vocab_size: int = 30000
    min_freq: int = 2
    max_len: int = 256
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: Path = Path("./save")

def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=Path, default=Path("./data"))
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--max_vocab_size", type=int, default=30000)
    p.add_argument("--min_freq", type=int, default=2)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_dir", type=Path, default=Path("./save"))
    args = p.parse_args()
    return Config(**vars(args))

def main():
    cfg = parse_args()
    cfg.save_dir.mkdir(parents=True, exist_ok=True)

    # 1) 데이터셋이 요청대로 ./data(또는 cfg.data_dir) 아래에 있는지 확인한다
    root = download_imdb(cfg.data_dir)

    # 2) 데이터로더와 어휘 구성
    train_loader, val_loader, test_loader, vocab, pad_idx = make_dataloaders(
        root=root,
        batch_size=cfg.batch_size,
        max_vocab_size=cfg.max_vocab_size,
        min_freq=cfg.min_freq,
        max_len=cfg.max_len,
        num_workers=cfg.num_workers,
        seed=cfg.seed
    )

    print(f"Vocab size: {len(vocab)}  (pad_idx={pad_idx})")
    device = torch.device(cfg.device)

    # 3) 모델 구성
    model = AverageEmbeddingsClassifier(vocab_size=len(vocab), embed_dim=cfg.embed_dim, pad_idx=pad_idx).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    # 4) 학습
    best_val_acc = 0.0
    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_loop(model, train_loader, optim, criterion, device)
        val_loss, val_acc = evaluate_loop(model, val_loader, criterion, device)
        print(f"[Epoch {epoch:02d}] Train loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"Val loss={val_loss:.4f} acc={val_acc:.3f}")
        # 최고 성능을 저장한다
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(cfg.save_dir / "imdb_avgemb_best.pt", model, vocab, pad_idx, cfg)

    # 5) 최종 시험
    test_loss, test_acc = evaluate_loop(model, test_loader, criterion, device)
    print(f"[Test] loss={test_loss:.4f} acc={test_acc:.3f}")

    # 마지막 상태를 저장한다
    save_checkpoint(cfg.save_dir / "imdb_avgemb_last.pt", model, vocab, pad_idx, cfg)

if __name__ == "__main__":
    main()```

## 논의

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 확장된다. 초매개변수, 구조의 변형, 여러 데이터셋을 두루 실험해 보면 이해가 깊어지고 분류 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
학습 루프에서 `optimizer.zero_grad()` 호출을 없애면 어떤 일이 일어나는지 설명하라. 고친 코드를 실행하고 학습 손실의 수렴에 미치는 영향을 서술하라.

??? success "연습문제 1 풀이"
    `optimizer.zero_grad()`가 없으면 PyTorch가 새 경사를 기존 `.grad` 텐서에 덮어쓰지 않고 더하기 때문에 반복에 걸쳐 경사가 누적된다. 이는 사실상 학습률에 누적된 단계 수를 곱하는 셈이어서 최적화가 점점 크고 불규칙한 걸음을 내딛게 된다. 학습 손실은 매끄럽게 수렴하는 대신 심하게 진동하거나 발산한다. 해결책은 간단하다. `loss.backward()`를 호출하기 전에 언제나 경사를 0으로 만들어라.

---

**연습문제 2.**
최적화기를 Adam으로 바꾸고(`torch.optim.Adam`에 `lr=0.001`을 쓴다) 원래 최적화기와 학습 수렴을 비교하라. 두 손실 곡선을 같은 그래프에 그려라.

??? success "연습문제 2 풀이"
    최적화기를 만드는 줄을 `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`으로 바꾼다. Adam은 매개변수마다 적응적인 학습률과 운동량 추정값을 유지하므로 초반 에폭에서 대체로 더 빠르게 수렴한다. Adam의 손실 곡선은 보통 처음 몇 에폭에서 더 가파르게 떨어지지만, 최적점 근처에서는 운동량을 쓴 SGD보다 조금 더 흔들릴 수 있다. 공정한 비교를 위해 둘을 같은 난수 씨앗과 같은 에폭 수로 실행하라.

---

**연습문제 3.**
이 구현에서 생길 수 있는 실패 양상 두 가지를 서술하고, 각각을 어떻게 진단하고 고칠지 설명하라.

??? success "연습문제 3 풀이"
    흔한 실패 양상은 다음과 같다. (1) **경사 소실/폭발** — 경사의 노름을 지켜보아 진단한다(`torch.nn.utils.clip_grad_norm_`을 쓰거나 층마다 `param.grad.norm()`을 기록한다). 경사 자르기, 더 나은 초기화(Xavier/Kaiming), 또는 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** — 학습 손실은 줄어드는데 검증 손실이 늘어나면 진단된다. 정칙화(드롭아웃, 가중치 감쇠, 데이터 증강)나 모델 용량 축소로 고친다. 이런 문제를 일찍 잡아내려면 언제나 학습 지표와 검증 지표를 함께 살펴라.

---

**연습문제 4.**
조기 종료를 구현하라. 매 에폭 후 검증 손실을 추적하고, 10 에폭 연속으로 개선이 없으면 학습을 멈춘다. 가장 좋은 모델 가중치를 저장하고 복원하라.

??? success "연습문제 4 풀이"
    인내 횟수 카운터와 최저 손실 추적기를 추가한다.
    ```python
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    for epoch in range(num_epochs):
        # ... 학습 단계 ...
        val_loss = evaluate(model, val_loader)
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        if patience_counter >= 10:
            print(f'Early stopping at epoch {epoch}')
            model.load_state_dict(best_state)
            break
    ```
    이렇게 하면 따로 떼어 둔 데이터에서 모델이 더 나아지지 않을 때 멈추므로 과적합을 막을 수 있다.
