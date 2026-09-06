# 데이터 증강

데이터 불리기는 자기 지도 대조 학습의 주춧돌이며, 모형이 어떤 정보에 대해 변하지 않도록 배워야 하는지를 정하는 얼개 노릇을 한다. SimCLR, MoCo, BYOL, MAE, 그리고 DINO 같은 여러 조각 자르기 방법은 저마다 무엇이 뜻있는 시각적 비슷함인지에 관한 귀납 편향을 담은, 정성껏 설계한 불리기 파이프라인을 쓴다. 어떤 불리기를 어떻게 짜맞추느냐가 배우는 표현의 질과 성격을 곧바로 정한다.

## 코드

```python
"""
자기 지도 학습을 위한 데이터 불리기 방법
SimCLR, MoCo를 비롯한 대조 방법이 쓰는 불리기를 담는다.
"""

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
from PIL import ImageFilter, ImageOps
import numpy as np

# ========================================================================
# 메인
# ========================================================================


class GaussianBlur:
    """가우스 흐리기 불리기"""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarization:
    """태양화 불리기"""
    def __init__(self, threshold=128):
        self.threshold = threshold

    def __call__(self, img):
        return ImageOps.solarize(img, self.threshold)


class SimCLRAugmentation:
    """
    SimCLR가 쓰는 데이터 불리기 파이프라인
    같은 그림의 서로 얽힌 시야 둘을 만든다
    """
    def __init__(self, image_size=224, s=1.0):
        """
        인수:
            image_size: 마지막 그림 크기
            s: 색 뒤틀기의 세기
        """
        # 색 흔들기 매개변수
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(sigma=[.1, 2.]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        """시야 둘을 만들려고 불리기를 두 번 적용한다"""
        return self.transform(x), self.transform(x)


class MoCoAugmentation:
    """
    MoCo v2가 쓰는 데이터 불리기 파이프라인
    """
    def __init__(self, image_size=224):
        # 질의 불리기 (더 세다)
        self.query_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 열쇠 불리기 (MoCo v2에서는 질의와 같다)
        self.key_transform = self.query_transform

    def __call__(self, x):
        """질의와 열쇠를 만들려고 불리기를 적용한다"""
        return self.query_transform(x), self.key_transform(x)


class MoCoV3Augmentation:
    """
    MoCo v3이 쓰는 데이터 불리기 파이프라인
    불리기를 더 담는다
    """
    def __init__(self, image_size=224):
        self.transform1 = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.transform2 = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
            transforms.RandomApply([Solarization()], p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        """서로 다른 불리기 둘을 적용한다"""
        return self.transform1(x), self.transform2(x)


class BYOLAugmentation:
    """
    BYOL이 쓰는 데이터 불리기 파이프라인
    비대칭 불리기를 쓴다
    """
    def __init__(self, image_size=224):
        # 시야 1: 더 센 불리기
        self.transform1 = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(sigma=[.1, 2.]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 시야 2: 더 약한 불리기
        self.transform2 = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        """비대칭 불리기를 적용한다"""
        return self.transform1(x), self.transform2(x)


class MAEAugmentation:
    """
    MAE의 데이터 불리기
    MAE는 대조 방법에 견주어 불리기를 아주 적게 쓴다
    """
    def __init__(self, image_size=224):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        """간단한 불리기를 적용한다"""
        return self.transform(x)


class MultiCropAugmentation:
    """
    SwAV와 DINO가 쓰는 여러 조각 불리기
    크기가 다른 조각 여럿을 만든다
    """
    def __init__(
        self,
        image_size=224,
        n_global_crops=2,
        n_local_crops=6,
        global_scale=(0.4, 1.0),
        local_scale=(0.05, 0.4)
    ):
        self.n_global_crops = n_global_crops
        self.n_local_crops = n_local_crops

        # 전역 조각 불리기
        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=global_scale),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(sigma=[.1, 2.]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 국소 조각 불리기
        local_size = int(image_size * 0.4)  # 국소 조각은 더 작은 크기
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(local_size, scale=local_scale),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(sigma=[.1, 2.]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        """전역 조각과 국소 조각을 만든다"""
        crops = []

        # 전역 조각
        for _ in range(self.n_global_crops):
            crops.append(self.global_transform(x))

        # 국소 조각
        for _ in range(self.n_local_crops):
            crops.append(self.local_transform(x))

        return crops


class TwoCropsTransform:
    """
    불린 시야 둘을 만드는 두루 쓰는 감싸개
    """
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return self.base_transform(x), self.base_transform(x)


def get_augmentation(method='simclr', image_size=224):
    """
    정한 방법의 불리기 파이프라인을 얻는다

    인수:
        method: 'simclr', 'moco', 'mocov3', 'byol', 'mae', 'multicrop'
        image_size: 그림 크기

    반환값:
        불리기 변환
    """
    if method == 'simclr':
        return SimCLRAugmentation(image_size)
    elif method == 'moco':
        return MoCoAugmentation(image_size)
    elif method == 'mocov3':
        return MoCoV3Augmentation(image_size)
    elif method == 'byol':
        return BYOLAugmentation(image_size)
    elif method == 'mae':
        return MAEAugmentation(image_size)
    elif method == 'multicrop':
        return MultiCropAugmentation(image_size)
    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    from PIL import Image

    # 임시 그림을 만든다
    img = Image.new('RGB', (256, 256), color='red')

    # SimCLR 불리기를 시험한다
    print("Testing SimCLR augmentation...")
    simclr_aug = SimCLRAugmentation(image_size=224)
    view1, view2 = simclr_aug(img)
    print(f"View 1 shape: {view1.shape}")
    print(f"View 2 shape: {view2.shape}")

    # MoCo 불리기를 시험한다
    print("\nTesting MoCo augmentation...")
    moco_aug = MoCoAugmentation(image_size=224)
    query, key = moco_aug(img)
    print(f"Query shape: {query.shape}")
    print(f"Key shape: {key.shape}")

    # MAE 불리기를 시험한다
    print("\nTesting MAE augmentation...")
    mae_aug = MAEAugmentation(image_size=224)
    aug_img = mae_aug(img)
    print(f"Augmented image shape: {aug_img.shape}")

    # 여러 조각 불리기를 시험한다
    print("\nTesting Multi-crop augmentation...")
    multicrop_aug = MultiCropAugmentation(image_size=224, n_global_crops=2, n_local_crops=4)
    crops = multicrop_aug(img)
    print(f"Number of crops: {len(crops)}")
    print(f"Global crop shapes: {[crops[i].shape for i in range(2)]}")
    print(f"Local crop shapes: {[crops[i].shape for i in range(2, 6)]}")

    print("\nAll augmentation tests passed!")
```

## 논의

자기 지도 학습에서 불리기 파이프라인의 설계는 시각적 이해에 어떤 불변성이 중요한지에 관한 사전 지식을 담는다. **SimCLR**는 무작위 크기 조정 자르기, 색 흔들기, 흑백 바꾸기, 가우스 흐리기를 아우르는 센 불리기를 쓰며, 자르기와 색 뒤틀기의 짜맞춤이 가장 중요한 조합임을 밝혔다. 어느 하나를 빼도 성능이 크게 떨어진다. 자르기만 쓰면 모형이 여전히 색 히스토그램을 지름길로 쓸 수 있고, 색 뒤틀기만 쓰면 고정된 공간 영역의 결 무늬에 기댈 수 있기 때문이다.

방법마다 두 시야에 불리기를 적용하는 방식이 다르다. **대칭** 방식(SimCLR, MoCo v2)은 두 시야에 같은 불리기 분포를 적용한다. **비대칭** 방식(BYOL, MoCo v3)은 한 시야에 더 센 불리기를, 다른 시야에 더 약한 불리기를 적용한다. 이를테면 BYOL은 둘째 시야에서 가우스 흐리기를 빼고, MoCo v3은 둘째 시야에만 태양화를 더한다. 이 비대칭이 드러난 음성 예를 쓰지 않는 방법에서 표현이 무너지는 것을 막아 준다. **MAE**는 놀랍도록 적은 불리기(무작위 자르기와 뒤집기뿐)를 쓰는데, 가리기 자체가 넉넉한 학습 신호를 주기 때문이다.

SwAV와 DINO가 쓰는 **여러 조각 자르기** 방법은 (그림의 40~100%를 덮는) 전역 조각과 (5~40%를 덮는) 국소 조각을 함께 만든다. 전역 조각은 장면 수준의 맥락을 잡고 국소 조각은 더 작은 영역에 집중한다. 이것이 모형에 국소 조각이 전역 시야와 들어맞는 표현을 배우게 하여, 같은 그림의 서로 다른 공간 크기가 뜻을 함께 나눈다는 것을 가르친다. 이 여러 크기의 한결같음이 DINO의 표현이 지도 없는 물체 분할 같은 강한 창발 성질을 보이는 한 이유이다.

## 연습문제

**연습문제 1.**
SimCLR와 MAE의 불리기 파이프라인을 견주어라. 저마다 쓰는 불리기를 모두 들고, MAE가 훨씬 약한 불리기로도 좋은 성능을 내는 까닭을 설명하라.

??? success "연습문제 1 풀이"
    **SimCLR의 불리기**: RandomResizedCrop(scale 0.2~1.0), RandomHorizontalFlip(p=0.5), ColorJitter(밝기, 대비, 채도, 색상, p=0.8), RandomGrayscale(p=0.2), GaussianBlur, ImageNet 정규화.

    **MAE의 불리기**: RandomResizedCrop(scale 0.2~1.0), RandomHorizontalFlip, ImageNet 정규화.

    MAE는 색 흔들기, 흑백 바꾸기, 가우스 흐리기를 아주 뺀다. 적은 불리기로도 좋은 성능을 내는 것은 **조각의 75%를 무작위로 가리는 것** 자체가 매우 강력한 불리기 노릇을 하기 때문이다. 되살리기 목표가 모형에 보이는 맥락에서 가려진 영역을 맞히려고 공간 관계와 결과 물체의 짜임을 이해하게 만든다. 대조 방법에서는 불리기가 학습 신호(무엇이 변하지 않아야 하는가)의 유일한 원천이므로 정성껏 설계해야 한다. MAE에서는 가리기가 모형에 시각적 내용을 곧바로 가르치는 보완 신호를 주어 불리기로 만든 불변성이 덜 필요해진다.

---

**연습문제 2.**
여러 방법에서 `RandomResizedCrop`의 `scale` 매개변수가 하는 몫을 설명하라. SimCLR는 왜 `scale=(0.2, 1.0)`을 쓰고 BYOL은 `scale=(0.08, 1.0)`을, SwAV의 국소 조각은 `scale=(0.05, 0.4)`을 쓰는가?

??? success "연습문제 2 풀이"
    `scale` 매개변수는 자른 조각이 본디 그림 넓이의 얼마를 지니는지를 다스린다. $(0.2, 1.0)$이면 조각이 그림 넓이의 20%에서 100%까지이다.

    **SimCLR**는 두 시야가 그림의 뜻있는 몫을 잡으면서도 대조 학습에 넉넉한 공간적 다양함을 주도록 $(0.2, 1.0)$이라는 알맞은 범위를 쓴다.

    **BYOL**은 더 센 $(0.08, 1.0)$을 써서 그림의 8%까지 작은 조각을 허락한다. BYOL은 드러난 음성을 쓰지 않으므로 (같은 그림의 더 다른 조각인) 더 어려운 양성 쌍의 덕을 보는데, 이것이 시야 사이의 일치를 지키려고 모형에 더 튼튼한 표현을 배우게 만든다.

    **SwAV·DINO의 국소 조각**은 일부러 작게 (그림의 5~40%인) $(0.05, 0.4)$을 쓴다. 이 조각은 국소적인 세부와 결을 잡는다. 여러 조각 자르기 방법은 이를 전역 조각 $(0.4, 1.0)$과 짝지어 국소 시야와 전역 시야의 한결같음을 지키게 하여, 부분과 전체의 관계가 표현 공간에서 지켜져야 함을 모형에 가르친다.

---

**연습문제 3.**
배치에서 그림 둘을 받아 직사각 영역을 무작위로 골라 그 영역을 둘 사이에 맞바꾸는 `CutMixAugmentation`이라는 불리기 클래스를 직접 구현하라. 이 클래스는 고친 두 그림과 섞음 비율(본디 그림에서 온 화소의 비율)을 함께 돌려주어야 한다. 반지도 대조 학습에 쓸모 있다.

??? success "연습문제 3 풀이"
    ```python
    class CutMixAugmentation:
        """대조 학습을 위한 CutMix 불리기."""

        def __init__(self, image_size=224, beta=1.0):
            self.image_size = image_size
            self.beta = beta

        def _rand_bbox(self, lam):
            """무작위 테두리 상자를 만든다."""
            W, H = self.image_size, self.image_size
            cut_ratio = np.sqrt(1.0 - lam)
            cut_w = int(W * cut_ratio)
            cut_h = int(H * cut_ratio)

            cx = np.random.randint(W)
            cy = np.random.randint(H)

            x1 = np.clip(cx - cut_w // 2, 0, W)
            y1 = np.clip(cy - cut_h // 2, 0, H)
            x2 = np.clip(cx + cut_w // 2, 0, W)
            y2 = np.clip(cy + cut_h // 2, 0, H)

            return x1, y1, x2, y2

        def __call__(self, img1, img2):
            """
            그림 둘 사이에 CutMix를 적용한다.

            반환값:
                mixed1: img2의 조각을 넣은 img1
                mixed2: img1의 조각을 넣은 img2
                lam: 섞음 비율 (mixed1에서 img1 화소의 비율)
            """
            lam = np.random.beta(self.beta, self.beta)
            x1, y1, x2, y2 = self._rand_bbox(lam)

            mixed1 = img1.clone()
            mixed2 = img2.clone()

            mixed1[:, y1:y2, x1:x2] = img2[:, y1:y2, x1:x2]
            mixed2[:, y1:y2, x1:x2] = img1[:, y1:y2, x1:x2]

            # 잘라 낸 뒤의 실제 섞음 비율
            actual_lam = 1 - (x2 - x1) * (y2 - y1) / (self.image_size ** 2)

            return mixed1, mixed2, actual_lam
    ```

    `beta` 매개변수는 섞음 비율을 뽑는 베타 분포를 다스린다. `beta=1.0`이면 비율이 $[0, 1]$에서 고르고, 값이 작으면 극단적인 비율(거의 한쪽 그림) 쪽으로 치우친다. 돌려주는 `lam`은 반지도 상황에서 대조 목표의 사이를 채우는 데 쓸 수 있다.
