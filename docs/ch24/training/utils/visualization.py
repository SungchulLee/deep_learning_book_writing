"""자기 부호기의 결과를 눈으로 확인하는 도움 함수.

VAE는 손실만 보아서는 잘 되고 있는지 알기 어렵다. 다시 세운 그림이 흐릿한지,
숨은 공간에서 뽑은 표본이 그럴듯한지는 그려 보아야 드러난다.
"""

import matplotlib.pyplot as plt
import torch

__all__ = ["visualize_reconstruction", "visualize_samples"]


def _to_image(t):
    """(C,H,W) 텐서를 matplotlib이 받는 모양으로 바꾼다."""
    t = t.detach().cpu()
    if t.dim() == 3 and t.shape[0] == 1:      # 회색조
        return t.squeeze(0), "gray"
    if t.dim() == 3:                          # 색깔: (C,H,W) -> (H,W,C)
        return t.permute(1, 2, 0), None
    return t, "gray"


def visualize_reconstruction(model, data_loader, num_images=10, device="cpu",
                             conditional=False, save_path=None):
    """원래 그림과 다시 세운 그림을 위아래로 나란히 놓는다.

    인수:
        model: forward가 (다시 세운 값, mu, logvar)를 돌려주는 자기 부호기
        data_loader: 볼 자료를 주는 로더
        num_images: 보여 줄 그림 수
        device: 모델이 올라가 있는 장치
        conditional: 조건부 모델이면 True. 이름표도 함께 넣어 준다
        save_path: 주면 그림을 그 경로로 저장한다
    """
    model.eval()
    images, labels = next(iter(data_loader))
    images = images[:num_images].to(device)
    labels = labels[:num_images].to(device)

    with torch.no_grad():
        # 온전히 이어진 자기 부호기는 펼친 입력을 받고, 누비기 쪽은 그림
        # 모양 그대로를 받는다. 어느 쪽인지 모르므로 그림 모양으로 먼저
        # 넣어 보고, 거부하면 펼쳐서 다시 넣는다.
        try:
            out = model(images, labels) if conditional else model(images)
        except RuntimeError:
            flat = images.view(images.size(0), -1)
            out = model(flat, labels) if conditional else model(flat)
    # forward가 (recon, mu, logvar)를 돌려주기도 하고 recon만 돌려주기도 한다
    recon = out[0] if isinstance(out, (tuple, list)) else out
    recon = recon.view_as(images)

    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 1.2, 3))
    for i in range(num_images):
        img, cmap = _to_image(images[i])
        axes[0, i].imshow(img, cmap=cmap)
        axes[0, i].axis("off")
        rec, cmap = _to_image(recon[i])
        axes[1, i].imshow(rec, cmap=cmap)
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Reconstruction")
    fig.suptitle("Top: original    Bottom: reconstruction")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def visualize_samples(model, latent_dim, num_samples=10, device="cpu",
                      class_label=None, save_path=None):
    """숨은 공간에서 마구잡이로 뽑아 복호기에 넣어 본다.

    다시 세우기와 달리 이쪽은 모델이 정말로 만들어 낼 줄 아는지를 본다.
    앞확률 N(0, I)에서 뽑으므로, 학습이 잘되었다면 그럴듯한 그림이 나와야 한다.
    """
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim, device=device)
        if class_label is not None:
            labels = torch.full((num_samples,), int(class_label),
                                dtype=torch.long, device=device)
            samples = model.decode(z, labels)
        else:
            samples = model.decode(z)

    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 1.2, 1.6))
    if num_samples == 1:
        axes = [axes]
    for i in range(num_samples):
        img, cmap = _to_image(samples[i])
        axes[i].imshow(img, cmap=cmap)
        axes[i].axis("off")
    title = "Samples from the latent space"
    if class_label is not None:
        title += f" (class {class_label})"
    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
