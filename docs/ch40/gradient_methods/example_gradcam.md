# Grad-CAM 예제

쓰는 보기: PyTorch 모델에 거는 Grad-CAM. 이 글은 Grad-CAM으로 CNN의 추론을 그리는 법을 보인다.

뉴럴 넷트웍이 무엇을 배우는지 아는 일은 믿음을 쌓고 모델의 버그를 잡는 데 중요하다. 이 모듈은 모델이 입력을 어떻게 다루고 어떻게 판단하는지 드러내는 그래디언트 기반 풀이 재주를 보이며, 네트워크의 액션을 눈으로 보고 수로 재게 해 준다.

## 1. 코드

```python
"""
쓰는 보기: PyTorch 모형에 거는 Grad-CAM

이 글은 Grad-CAM으로 CNN의 미루어 봄을 그리는 법을 보인다.
"""

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from gradcam import GradCAM, GradCAMPlusPlus, get_target_layer

# ========================================================================
# 메인
# ========================================================================


def load_and_preprocess_image(image_path: str, image_size: int = 224):
    """
    모형에 넣을 그림을 부르고 미리 다듬는다.

    Args:
        image_path: 그림 두루마리의 길
        image_size: 겨눈 그림 크기

    Returns:
        미리 다듬은 그림 텐서와 본디 그림 배열
    """
    # 그림을 부른다
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image)

    # 미리 다듬기를 세운다
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # 미리 다듬는다
    input_tensor = preprocess(image).unsqueeze(0)

    # 겹쳐 보이려고 본디 그림의 크기를 맞춘다
    original_image = np.array(Image.fromarray(original_image).resize((image_size, image_size)))

    return input_tensor, original_image


def example_resnet_gradcam():
    """
    보기: ResNet50에 Grad-CAM 걸기
    """
    print("=" * 60)
    print("보기 1: ResNet50에 거는 Grad-CAM")
    print("=" * 60)

    # 미리 익힌 ResNet50을 부른다
    model = models.resnet50(pretrained=True)
    model.eval()

    # 마지막 겹치는 켜를 집는다
    target_layer = model.layer4[-1]

    # Grad-CAM의 첫자리를 잡는다
    gradcam = GradCAM(model, target_layer)

    # 보기: 아무 그림이나 만든다(참 그림 길로 갈음하라)
    print("\n보기 들임을 만든다(참 그림으로 갈음하라)...")
    input_tensor = torch.randn(1, 3, 224, 224)

    # CAM을 만든다
    print("Grad-CAM 열 그림을 만드는 중...")
    cam = gradcam.generate_cam(input_tensor, target_class=None)

    print(f"CAM 꼴: {cam.shape}")
    print(f"CAM 가장 작음/큼: {cam.min():.4f} / {cam.max():.4f}")

    # 그림으로 보인다
    print("그림을 만드는 중...")
    visualization = gradcam.visualize_cam(input_tensor)

    # 보여 준다
    plt.figure(figsize=(8, 8))
    plt.imshow(visualization)
    plt.title('Grad-CAM 그림')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('gradcam_example.png', dpi=150, bbox_inches='tight')
    print("그림을 'gradcam_example.png'에 갈무리했다")
    plt.close()


def example_with_real_image(image_path: str):
    """
    보기: 참 그림에 거는 Grad-CAM

    Args:
        image_path: 들임 그림의 길
    """
    print("\n" + "=" * 60)
    print("보기 2: 참 그림에 거는 Grad-CAM")
    print("=" * 60)

    # 모형을 부른다
    print("ResNet50을 부르는 중...")
    model = models.resnet50(pretrained=True)
    model.eval()

    # 그림을 부르고 미리 다듬는다
    print(f"{image_path}에서 그림을 부르는 중...")
    input_tensor, original_image = load_and_preprocess_image(image_path)

    # 겨눈 켜를 집는다
    target_layer = model.layer4[-1]

    # Grad-CAM의 첫자리를 잡는다
    gradcam = GradCAM(model, target_layer)

    # 앞으로 걸음으로 미루어 봄을 얻는다
    with torch.no_grad():
        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()
        pred_prob = torch.softmax(output, dim=1)[0, pred_class].item()

    print(f"미루어 본 클래스: {pred_class} (자신함: {pred_prob:.2%})")

    # Grad-CAM을 만든다
    print("Grad-CAM을 만드는 중...")
    visualization = gradcam.visualize_cam(input_tensor,
                                         target_class=pred_class,
                                         original_image=original_image)

    # 견줌 그림을 만든다
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_image)
    axes[0].set_title('본디 그림')
    axes[0].axis('off')

    cam = gradcam.generate_cam(input_tensor, target_class=pred_class)
    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('Grad-CAM 열 그림')
    axes[1].axis('off')

    axes[2].imshow(visualization)
    axes[2].set_title(f'겹쳐 보이기 (클래스: {pred_class})')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('gradcam_comparison.png', dpi=150, bbox_inches='tight')
    print("견줌 그림을 'gradcam_comparison.png'에 갈무리했다")
    plt.close()


def example_gradcam_plusplus():
    """
    보기: Grad-CAM과 Grad-CAM++ 견주기
    """
    print("\n" + "=" * 60)
    print("보기 3: Grad-CAM 대 Grad-CAM++")
    print("=" * 60)

    # 모형을 부른다
    model = models.resnet50(pretrained=True)
    model.eval()
    target_layer = model.layer4[-1]

    # 아무 들임이나 만든다
    input_tensor = torch.randn(1, 3, 224, 224)

    # 여느 Grad-CAM
    print("여느 Grad-CAM을 만드는 중...")
    gradcam = GradCAM(model, target_layer)
    cam1 = gradcam.generate_cam(input_tensor)
    vis1 = gradcam.visualize_cam(input_tensor)

    # Grad-CAM++
    print("Grad-CAM++을 만드는 중...")
    gradcam_pp = GradCAMPlusPlus(model, target_layer)
    cam2 = gradcam_pp.generate_cam(input_tensor)
    vis2 = gradcam_pp.visualize_cam(input_tensor)

    # 견준다
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    axes[0, 0].imshow(cam1, cmap='jet')
    axes[0, 0].set_title('Grad-CAM 열 그림')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(vis1)
    axes[0, 1].set_title('Grad-CAM 그림')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(cam2, cmap='jet')
    axes[1, 0].set_title('Grad-CAM++ 열 그림')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(vis2)
    axes[1, 1].set_title('Grad-CAM++ 그림')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('gradcam_comparison_methods.png', dpi=150, bbox_inches='tight')
    print("방법 견줌을 'gradcam_comparison_methods.png'에 갈무리했다")
    plt.close()


def example_multiple_classes():
    """
    보기: 여러 클래스에 대한 Grad-CAM 그리기
    """
    print("\n" + "=" * 60)
    print("보기 4: 여러 클래스에 대한 Grad-CAM")
    print("=" * 60)

    # 모형을 부른다
    model = models.resnet50(pretrained=True)
    model.eval()
    target_layer = model.layer4[-1]

    # Grad-CAM의 첫자리를 잡는다
    gradcam = GradCAM(model, target_layer)

    # 들임을 만든다
    input_tensor = torch.randn(1, 3, 224, 224)

    # 앞선 5개 미루어 봄을 얻는다
    with torch.no_grad():
        output = model(input_tensor)
        probs, classes = torch.softmax(output, dim=1).topk(5)

    # 앞선 클래스마다 CAM을 만든다
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for idx, (prob, cls) in enumerate(zip(probs[0], classes[0])):
        cam = gradcam.generate_cam(input_tensor, target_class=cls.item())
        axes[idx].imshow(cam, cmap='jet')
        axes[idx].set_title(f'클래스 {cls.item()}\n({prob.item():.2%})')
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('gradcam_multiple_classes.png', dpi=150, bbox_inches='tight')
    print("여러 클래스 그림을 'gradcam_multiple_classes.png'에 갈무리했다")
    plt.close()


def example_different_architectures():
    """
    보기: 여러 구조에 Grad-CAM 걸기
    """
    print("\n" + "=" * 60)
    print("보기 5: 여러 구조에 거는 Grad-CAM")
    print("=" * 60)

    architectures = {
        'ResNet50': (models.resnet50(pretrained=True), 'layer4'),
        'VGG16': (models.vgg16(pretrained=True), 'features'),
        'MobileNetV2': (models.mobilenet_v2(pretrained=True), 'features'),
    }

    input_tensor = torch.randn(1, 3, 224, 224)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (name, (model, layer_name)) in enumerate(architectures.items()):
        print(f"\n{name}을 다루는 중...")
        model.eval()

        # 겨눈 켜를 집는다
        if layer_name == 'features':
            target_layer = list(model.features.children())[-1]
        else:
            target_layer = get_target_layer(model, layer_name)

        # Grad-CAM을 만든다
        gradcam = GradCAM(model, target_layer)
        cam = gradcam.generate_cam(input_tensor)

        axes[idx].imshow(cam, cmap='jet')
        axes[idx].set_title(name)
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('gradcam_different_architectures.png', dpi=150, bbox_inches='tight')
    print("\n구조 견줌을 'gradcam_different_architectures.png'에 갈무리했다")
    plt.close()


if __name__ == "__main__":
    print("Grad-CAM 보기\n")

    # 보기를 돌린다
    example_resnet_gradcam()
    example_gradcam_plusplus()
    example_multiple_classes()
    example_different_architectures()

    # 참 그림에 쓰려면 주석을 푼다
    # example_with_real_image('path/to/your/image.jpg')

    print("\n" + "=" * 60)
    print("보기를 모두 마쳤다!")
    print("=" * 60)
```

## 2. 논의

그림으로 보이기는 모델의 액션을 알고 학습의 탈을 짚어내는 데 큰 몫을 한다. 그리는 코드는 배운 나타냄, 모여 가는 피처, 따짐 자를 들여다보게 해서 손에 잡히지 않던 계산을 눈에 보이게 한다.

여기서 보인 피처는 더 까다로운 자리로도 자연스레 넓혀진다. 하이퍼파라미터, 구조의 클래스, 여러 데이터를 바꿔 가며 해 보면 이해가 깊어지고 모델 해석에 대한 감이 몸에 붙는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
코드를 읽고 핵심이 되는 설계 판단을 짚어라. 짜기에서 고른 것 셋을 들고, 저마다 왜 그래디언트 기반 풀이에 알맞은지 밝혀라.

</div>

??? success "연습문제 1 풀이"
    설계 판단은 구현마다 다르나 흔히 이런 것이 있다. (1) 엑티베이션 함수 고르기 -- ReLU 클래스는 그래디언트가 잦아들지 않아 학습이 빠르다. (2) 고르게 하는 꾀 -- 배치 고르게 하기가 안쪽 함께 바뀌는 옮겨감을 줄여 학습을 든든하게 한다. (3) 나머지 이음 -- 있으면 건너뛰는 길을 주어 깊은 네트워크에서 그래디언트가 흐르게 한다. 고른 것마다 나타내는 힘, 계산 비용, 학습의 든든함 사이의 트레이드오프를 드러낸다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력의 꼴과 데이터 클래스가 바라는 대로인지 살피는 입력 살피기를 으뜸 함수나 클래스에 더하여라. 올바르지 않은 입력에는 알아듣기 쉬운 어긋남 알림을 띄워라.

</div>

??? success "연습문제 2 풀이"
    `forward` 방법(또는 알맞은 함수)의 소개에 `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`이나 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'` 같은 살핌을 더한다. 꼴을 살피려면 중요한 차원을 본다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 알아듣기 쉬운 어긋남 알림은 디버깅을 크게 앞당기고 코드를 되쓰기 든든하게 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 구현이 무너질 만한 피처 둘을 밝히고, 저마다 어떻게 짚어내고 고칠지 밝혀라.

</div>

??? success "연습문제 3 풀이"
    흔히 무너지는 피처는 이렇다. (1) **그래디언트가 사라지거나 터짐** -- 그래디언트 크기를 지켜보아 짚어낸다(`torch.nn.utils.clip_grad_norm_`이나 레이어마다 `param.grad.norm()` 적기). 그래디언트 자르기, 더 나은 첫값 잡기(Xavier/Kaiming), 구조 고치기(나머지 이음, 고르게 하기)로 고친다. (2) **지나치게 맞추기** -- 학습 로스는 줄어드는데 살핌 로스가 오르면 짚어낸다. 정규화(드롭아웃, 웨이트 줄이기, 데이터 늘리기)나 모델 크기 줄이기로 고친다. 학습과 살핌 자를 늘 함께 지켜보아 이를 일찍 잡아야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
쓰는 보기 구현을 살피는 두루 갖춘 시험 함수를 써라. 빈 입력, 원소 하나짜리 입력, 아주 큰 입력, 그리고 끝자락 값(0, 아주 큰 수)이 든 입력 같은 가장자리 자리를 시험하여라.

</div>

??? success "연습문제 4 풀이"
    금 언저리 조건을 두루 건드리는 시험 함수를 짓는다.
    ```python
    def test_example usage():
        model = 쓰는 보기(...)
        # 여느 들임
        assert model(normal_input).shape == expected_shape
        # 원소 하나짜리 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값(넘침을 살핀다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 그래디언트 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    구조가 끝에서 끝까지 학습을 받치는지 알려면 그래디언트 흐름을 시험하는 것이 특히 중요하다.

## 정리하며

**다룬 것** — 쓰는 보기

그림으로 보이기는 모델의 액션을 알고 학습의 탈을 짚어내는 데 큰 몫을 한다.

앞의 연습문제 4개로 스스로 따져 볼 수 있다.
