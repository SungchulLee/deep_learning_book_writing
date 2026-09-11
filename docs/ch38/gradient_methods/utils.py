"""38.2 기울기 방법 절이 함께 쓰는 도움 함수.

이 절의 예제들(맨 기울기, SmoothGrad, 쌓은 기울기, Grad-CAM, 이끈 되짚기)은
모두 같은 일을 앞뒤로 한다. 미리 익힌 모델을 불러오고, 그림 하나를 모델이
받는 꼴로 다듬고, 나온 두드러짐 지도를 원래 그림 위에 얹어 보인다.
그 공통부를 여기에 모아 둔다.

예제들은 `from utils import *`로 이 이름들을 한꺼번에 들여온다.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F          # 예제들이 utils를 거쳐 F를 쓴다
from PIL import Image

__all__ = [
    "F",
    "get_device",
    "load_pretrained_model",
    "preprocess_image",
    "denormalize",
    "visualize_saliency",
    "visualize_multiple_saliencies",
    "create_output_dir",
]

# ImageNet으로 익힌 모델이 쓰는 값. 다듬을 때와 되돌릴 때 같은 값을 써야 한다.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_device():
    """쓸 수 있는 가장 빠른 장치를 고른다."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_pretrained_model(model_name="resnet50", device=None, eval_mode=True):
    """torchvision에서 미리 익힌 모델을 불러온다.

    내려받기가 안 되면(망이 없을 때) 익히지 않은 모델로 물러난다. 기울기
    방법을 보이는 데에는 가중치가 참으로 학습된 것이 아니어도 그림이 나온다.
    """
    from torchvision import models

    device = device or get_device()
    try:
        model = getattr(models, model_name)(weights="DEFAULT")
    except Exception:
        print(f"  미리 익힌 가중치를 받지 못했다 — {model_name}을 맨 상태로 쓴다.")
        model = getattr(models, model_name)(weights=None)

    model = model.to(device)
    if eval_mode:
        model.eval()
    return model


def preprocess_image(image, size=224, device=None):
    """그림을 모델이 받는 (1, 3, size, size) 텐서로 다듬는다.

    인수:
        image: 파일 경로이거나 PIL 그림
        size: 자를 크기
        device: 텐서를 올릴 장치

    파일이 없으면 회색 그림으로 물러난다. 예제가 그림 하나 때문에
    통째로 멈추지 않게 하기 위해서다.
    """
    from torchvision import transforms

    device = device or get_device()

    if isinstance(image, (str, os.PathLike)):
        try:
            image = Image.open(image).convert("RGB")
        except FileNotFoundError:
            print(f"  '{image}'을 찾지 못했다 — 민 회색 그림으로 대신한다.")
            image = Image.new("RGB", (size, size), (128, 128, 128))
    elif not isinstance(image, Image.Image):
        raise TypeError("image는 파일 경로이거나 PIL 그림이어야 한다")

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform(image).unsqueeze(0).to(device)


def denormalize(image_tensor):
    """고르게 한 텐서를 [0,1]로 되돌려 그릴 수 있게 만든다."""
    t = image_tensor.detach().cpu()
    if t.dim() == 4:
        t = t[0]
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def _as_map(saliency):
    """두드러짐을 (H, W) 넘파이 배열로 맞추고 [0,1]로 고르게 한다."""
    s = saliency.detach().cpu() if torch.is_tensor(saliency) else torch.as_tensor(saliency)
    while s.dim() > 2:
        s = s.squeeze(0) if s.shape[0] == 1 else s.max(0).values
    s = s.numpy().astype(np.float32)
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else np.zeros_like(s)


def visualize_saliency(image_tensor, saliency, title="Saliency", colormap="jet",
                       alpha=0.4, save_path=None):
    """원래 그림, 두드러짐 지도, 그 둘을 겹친 것을 나란히 보인다."""
    img = denormalize(image_tensor)
    sal = _as_map(saliency)
    if sal.shape != img.shape[:2]:
        sal = np.array(Image.fromarray((sal * 255).astype(np.uint8))
                       .resize((img.shape[1], img.shape[0]))) / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img);              axes[0].set_title("Input");    axes[0].axis("off")
    axes[1].imshow(sal, cmap=colormap); axes[1].set_title("Saliency"); axes[1].axis("off")
    axes[2].imshow(img)
    axes[2].imshow(sal, cmap=colormap, alpha=alpha)
    axes[2].set_title("Overlay");     axes[2].axis("off")
    # 제목은 한글이 섞일 수 있다. matplotlib 기본 글꼴에 한글이 없으므로
    # 그림 위가 아니라 표준 출력으로 알린다.
    print(f"  [그림] {title}")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def visualize_multiple_saliencies(image_tensor, saliencies, colormap="jet",
                                  alpha=0.4, save_path=None):
    """여러 방법의 두드러짐을 한 줄에 늘어놓아 견준다.

    인수:
        saliencies: {이름: 두드러짐} 사전. 이름은 표준 출력으로 알린다
    """
    img = denormalize(image_tensor)
    names = list(saliencies)
    fig, axes = plt.subplots(1, len(names) + 1, figsize=(4 * (len(names) + 1), 4))
    axes[0].imshow(img); axes[0].set_title("Input"); axes[0].axis("off")

    for ax, name in zip(axes[1:], names):
        sal = _as_map(saliencies[name])
        if sal.shape != img.shape[:2]:
            sal = np.array(Image.fromarray((sal * 255).astype(np.uint8))
                           .resize((img.shape[1], img.shape[0]))) / 255.0
        ax.imshow(img)
        ax.imshow(sal, cmap=colormap, alpha=alpha)
        ax.axis("off")
    print("  [그림] " + " | ".join(n.replace("\n", " ") for n in names))
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def create_output_dir(path="outputs"):
    """그림을 담을 자리를 만들고 그 경로를 돌려준다."""
    os.makedirs(path, exist_ok=True)
    return path
