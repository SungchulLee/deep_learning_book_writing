"""변분 자기 부호기의 손실.

VAE의 손실은 두 항의 합이다.

    L = 다시 세우기 손실 + beta * KL 갈림

다시 세우기 항은 복호기가 입력을 얼마나 되살렸는지를 재고, KL 항은 숨은
분포를 표준 정규분포 쪽으로 당긴다. beta가 1이면 보통의 VAE이고, 1보다 크면
beta-VAE가 되어 숨은 축이 서로 풀리는 쪽을 더 세게 밀어붙인다.
"""

import torch
import torch.nn.functional as F

__all__ = ["vae_loss", "beta_vae_loss"]


def vae_loss(reconstruction, target, mu, logvar, beta=1.0, reduction="sum"):
    """VAE의 손실을 셈한다.

    인수:
        reconstruction: 복호기가 내놓은 값. [0,1]이어야 한다(시그모이드를 거친 것)
        target: 원래 입력. 마찬가지로 [0,1]이어야 한다
        mu, logvar: 부호기가 내놓은 숨은 분포의 평균과 로그 분산
        beta: KL 항의 무게. 1이면 보통의 VAE
        reduction: 'sum'이면 묶음 전체의 합, 'mean'이면 표본마다의 평균

    반환값:
        (전체 손실, 다시 세우기 손실, KL 손실)
    """
    # 목표가 [0,1] 밖이면 BCE가 거부한다. 미리 일러 주는 편이 낫다.
    if target.min() < 0 or target.max() > 1:
        raise ValueError(
            "target은 [0,1] 안에 있어야 한다. 화소를 고르게 할 때 "
            "Normalize로 음수를 만들지 않았는지 살펴보라."
        )

    recon_loss = F.binary_cross_entropy(
        reconstruction.reshape(target.shape[0], -1),
        target.reshape(target.shape[0], -1),
        reduction="sum",
    )

    # KL(N(mu, sigma^2) || N(0, 1)) 의 닫힌 꼴
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    if reduction == "mean":
        n = target.shape[0]
        recon_loss, kl_loss = recon_loss / n, kl_loss / n

    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def beta_vae_loss(reconstruction, target, mu, logvar, beta=4.0, reduction="sum"):
    """beta-VAE의 손실. vae_loss에 beta를 크게 준 것과 같다.

    beta를 키우면 숨은 축이 서로 풀리는 쪽으로 더 세게 밀리지만, 그만큼
    다시 세우기가 흐려진다. 이 맞바꿈이 beta-VAE의 요점이다.
    """
    return vae_loss(reconstruction, target, mu, logvar, beta=beta, reduction=reduction)
