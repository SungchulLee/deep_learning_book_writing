# 영상 기초: 읽어 들이고 다루기

---

## 1. 학습 목표

이 절을 마치면 다음을 할 수 있게 된다.

- 영상 자료를 틀의 때 차례로 나타내는 법을 이해한다
- 여러 뒷단(torchvision, OpenCV)으로 영상을 읽어 들이고 다룬다
- 서로 다른 영상 텐서 꼴 사이를 옮긴다
- 틀 뽑기와 표집 전략을 짠다
- 영상 신경망에 알맞은 앞손질을 한다

---

## 2. 수학적 바탕

### 때 차례로서의 영상

영상 $V$은 근본적으로 틀 $T$개의 차례이다:

$$V = \{I_1, I_2, \ldots, I_T\}$$

여기서 틀 $I_t \in \mathbb{R}^{H \times W \times C}$은 때 $t$의 그림을 나타내며 높이는 $H$, 너비는 $W$, 갈래는 $C$개다(RGB이면 흔히 3이다).

### 텐서로 나타내기

PyTorch에서 영상은 4차원이나 5차원 텐서로 나타낸다:

**영상 하나(4차원):**

$$V \in \mathbb{R}^{T \times C \times H \times W}$$

**영상 묶음(5차원):**

$$V \in \mathbb{R}^{B \times T \times C \times H \times W}$$

여기서 $B$은 묶음 크기이다.

### 틀 비율과 길이

때에 관한 성질 사이의 관계:

$$\text{Duration (seconds)} = \frac{T}{\text{FPS}}$$

$$T = \text{Duration} \times \text{FPS}$$

여기서 초당 틀 수(FPS)가 때의 해상도를 정한다.

---

## 3. 영상 읽어 들이기 뒷단

### OpenCV 뒷단

OpenCV는 유연하고 실전에 쓸 수 있는 영상 읽어 들이기를 준다:

```python
import cv2
import numpy as np
import torch

class VideoLoader:
    """두루 쓰는 영상 읽어 들이기 도구."""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        
    def load_video_opencv(self):
        """
        OpenCV 뒷단으로 영상을 읽어 들인다.
        
        반환값:
            frames: 꼴이 (T, H, W, C)인 NumPy 배열
            info: 영상 메타자료를 담은 사전
        """
        cap = cv2.VideoCapture(self.video_path)
        
        # 영상의 성질 뽑아내기
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # BGR(OpenCV 붙박이)을 RGB로 바꾸기
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        frames = np.array(frames)  # 꼴: (T, H, W, 3)
        
        info = {
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration': frame_count / fps
        }
        
        return frames, info
```

### Torchvision 뒷단

torchvision으로 PyTorch에 곧바로 녹아든다:

```python
from torchvision.io import read_video, write_video

def load_video_torchvision(video_path: str):
    """
    torchvision 뒷단으로 영상을 읽어 들인다.
    
    반환값:
        video: 값이 [0, 255]이고 꼴이 (T, H, W, C)인 텐서
        audio: 있으면 소리 텐서
        info: 영상 메타자료를 담은 사전
    """
    video, audio, info = read_video(
        video_path,
        pts_unit='sec'  # 때 도장에 초를 쓴다
    )
    
    return video, audio, info
```

### 꼴 바꾸기

틀이 다를 때는 꼴 사이를 옮기는 일이 꼭 필요하다:

```python
def convert_to_pytorch_format(frames: np.ndarray) -> torch.Tensor:
    """
    NumPy 영상을 PyTorch 텐서 꼴로 바꾼다.
    
    인수:
        frames: 값이 [0, 255]인 NumPy 배열 (T, H, W, C)
        
    반환값:
        video_tensor: 값이 [0, 1]인 PyTorch 텐서 (T, C, H, W)
    """
    # 텐서로 바꾸기
    video_tensor = torch.from_numpy(frames).float()
    
    # 차원 다시 늘어놓기: (T, H, W, C) → (T, C, H, W)
    video_tensor = video_tensor.permute(0, 3, 1, 2)
    
    # [0, 1]로 고르게 맞추기
    video_tensor = video_tensor / 255.0
    
    return video_tensor
```

---

## 4. 틀 표집 전략

표집 전략은 어떤 틀을 다룰지 정하며, 셈 값과 때의 덮음 사이 균형을 잡는다.

### 고른 표집

영상 전체에 고루 퍼지도록 틀을 뽑는다:

$$i_k = \left\lfloor \frac{k \cdot T}{n} \right\rfloor \quad \text{for } k = 0, 1, \ldots, n-1$$

여기서 $n$은 뽑을 틀의 개수이다.

```python
def uniform_sampling(video: torch.Tensor, num_frames: int) -> torch.Tensor:
    """
    영상 전체에서 틀을 고루 뽑는다.
    
    인수:
        video: 들임 텐서 (T, C, H, W)
        num_frames: 뽑을 틀의 개수
        
    반환값:
        뽑은 텐서 (num_frames, C, H, W)
    """
    T = video.shape[0]
    indices = torch.linspace(0, T - 1, num_frames).long()
    return video[indices]
```

### 때 성큼 표집

$s$번째 틀마다 뽑는다:

```python
def stride_sampling(video: torch.Tensor, stride: int) -> torch.Tensor:
    """
    붙박이 때 성큼으로 틀을 뽑는다.
    
    인수:
        video: 들임 텐서 (T, C, H, W)
        stride: 때 성큼(stride-1개 틀씩 건너뛴다)
        
    반환값:
        때 차원을 줄여 뽑은 텐서
    """
    return video[::stride]
```

### 촘촘한 표집

때 그물을 위해 겹치는 토막을 여럿 뽑는다:

```python
def dense_sampling(video: torch.Tensor, 
                   clip_length: int, 
                   num_clips: int) -> list:
    """
    촘촘한 어림을 위해 영상에서 토막을 여럿 뽑는다.
    
    인수:
        video: 들임 텐서 (T, C, H, W)
        clip_length: 토막마다의 틀 개수
        num_clips: 뽑을 토막의 개수
        
    반환값:
        토막 텐서의 목록. 저마다 (clip_length, C, H, W)
    """
    T = video.shape[0]
    
    if T < clip_length:
        # 영상이 토막 길이보다 짧으면 덧대기
        padding = torch.zeros(clip_length - T, *video.shape[1:])
        video = torch.cat([video, padding], dim=0)
        T = clip_length
    
    # 토막 시작 자리 셈하기
    max_start = T - clip_length
    if num_clips == 1:
        starts = [max_start // 2]
    else:
        starts = torch.linspace(0, max_start, num_clips).long().tolist()
    
    clips = [video[start:start + clip_length] for start in starts]
    return clips
```

### 마구잡이 표집

익히는 동안 자료 불리기에 쓸모 있다:

```python
def random_sampling(video: torch.Tensor, num_frames: int) -> torch.Tensor:
    """
    틀을 마구잡이로 뽑는다(자료 불리기용).
    
    인수:
        video: 들임 텐서 (T, C, H, W)
        num_frames: 뽑을 틀의 개수
        
    반환값:
        뽑은 텐서 (num_frames, C, H, W)
    """
    T = video.shape[0]
    
    # 마구잡이 번호를 뽑고 때 차례를 지키려 정렬
    indices = torch.randint(0, T, (num_frames,))
    indices, _ = torch.sort(indices)
    
    return video[indices]
```

---

## 5. 영상 앞손질

### 고르게 맞추기

옮겨 배우기를 위해 ImageNet 방식으로 고르게 맞춘다:

```python
class VideoPreprocessor:
    """영상 자료를 위한 앞손질 도구."""
    
    # ImageNet 통계량
    MEAN = torch.tensor([0.485, 0.456, 0.406])
    STD = torch.tensor([0.229, 0.224, 0.225])
    
    def normalize(self, video: torch.Tensor) -> torch.Tensor:
        """
        영상에 ImageNet 방식 고르게 맞추기를 쓴다.
        
        인수:
            video: 값이 [0, 1]인 들임 텐서 (T, C, H, W)
            
        반환값:
            평균 0, 흩어짐 1로 고르게 맞춘 텐서
        """
        # 퍼뜨리기를 위해 꼴 바꾸기: (1, C, 1, 1)
        mean = self.MEAN.view(1, -1, 1, 1)
        std = self.STD.view(1, -1, 1, 1)
        
        return (video - mean) / std
    
    def denormalize(self, video: torch.Tensor) -> torch.Tensor:
        """그려 보려고 고르게 맞추기를 되돌린다."""
        mean = self.MEAN.view(1, -1, 1, 1)
        std = self.STD.view(1, -1, 1, 1)
        
        return video * std + mean
```

### 자리 잘라내기

모든 틀에 한결같은 자리 잘라내기를 한다:

```python
def spatial_crop(video: torch.Tensor, 
                 crop_size: tuple,
                 position: str = 'center') -> torch.Tensor:
    """
    모든 틀에 자리 잘라내기를 한다.
    
    인수:
        video: 들임 텐서 (T, C, H, W)
        crop_size: (자를 높이, 자를 너비)
        position: 'center', 'random', 또는 'top_left'
        
    반환값:
        잘라낸 텐서 (T, C, crop_h, crop_w)
    """
    T, C, H, W = video.shape
    crop_h, crop_w = crop_size
    
    if position == 'center':
        top = (H - crop_h) // 2
        left = (W - crop_w) // 2
    elif position == 'random':
        top = torch.randint(0, H - crop_h + 1, (1,)).item()
        left = torch.randint(0, W - crop_w + 1, (1,)).item()
    else:  # top_left
        top, left = 0, 0
    
    return video[:, :, top:top+crop_h, left:left+crop_w]
```

### 크기 바꾸기

틀을 목표 해상도로 바꾼다:

```python
import torch.nn.functional as F

def resize_video(video: torch.Tensor, 
                 target_size: tuple) -> torch.Tensor:
    """
    모든 틀을 목표 크기로 바꾼다.
    
    인수:
        video: 들임 텐서 (T, C, H, W)
        target_size: (과녁 높이, 과녁 너비)
        
    반환값:
        크기를 바꾼 텐서 (T, C, target_h, target_w)
    """
    T, C, H, W = video.shape
    target_h, target_w = target_size
    
    # 묶음 처리를 위해 꼴 바꾸기
    video_flat = video.view(T, C, H, W)
    
    # 두 줄 사이 끼움 쓰기
    resized = F.interpolate(
        video_flat,
        size=(target_h, target_w),
        mode='bilinear',
        align_corners=False
    )
    
    return resized
```

---

## 6. 시각화

### 틀 보여 주기

```python
import matplotlib.pyplot as plt

def visualize_frames(video: torch.Tensor, num_frames: int = 8):
    """
    영상에서 뽑은 틀을 보여 준다.
    
    인수:
        video: 영상 텐서 (T, C, H, W)
        num_frames: 보여 줄 틀의 개수
    """
    T = video.shape[0]
    indices = torch.linspace(0, T - 1, num_frames).long()
    
    fig, axes = plt.subplots(1, num_frames, figsize=(16, 4))
    
    for i, ax in enumerate(axes):
        frame = video[indices[i]].permute(1, 2, 0)  # (C, H, W) → (H, W, C)
        frame = torch.clamp(frame, 0, 1)
        
        ax.imshow(frame.cpu().numpy())
        ax.axis('off')
        ax.set_title(f'Frame {indices[i].item()}')
    
    plt.tight_layout()
    plt.show()
```

---

## 7. 다음 걸음

영상 기초를 다졌으니 이제 다음을 살펴볼 수 있다:

1. **3차원 누비기** — 자리와 때에 걸친 특징 뽑기
2. **때 나타내기** — 움직임과 흐름 이해하기
3. **두 갈래 그물** — 겉모습과 움직임 아우르기

---

## 연습문제

**연습문제 1.**
영상 이해를 위한 두 갈래 그물의 핵심 눈썰미를 밝히고, 날 틀 위의 한 갈래만으로는 왜 모자란지 설명하여라.

??? success "연습문제 1 풀이"
    두 갈래 그물은 **자리**(겉모습) 앎과 **때**(움직임) 앎을 서로 다른 갈래에서 다룬다. RGB 한 갈래만으로도 겉모습은 담아내지만 움직임에는 약한데, 그 까닭은 이렇다. (1) 때의 무늬는 여러 틀에 걸쳐 있어 넓은 받는 자리가 필요하다. (2) 움직임은 화소 바뀜 속에 숨어 있어 날 자료에서 배우기 어렵다. 빛 흐름 갈래는 움직임을 드러내어 부호로 담아 서로 채워 주는 신호를 준다. 두 갈래는 보통 (늦게 또는 중간에서) 녹여 붙여 겉모습과 움직임 이해를 아우르며, 한 갈래 방식보다 훨씬 낫다.

---

**연습문제 2.**
느림빠름 얼개를 설명하여라. 영상을 두 가지 틀 비율로 다루면 왜 알아보기가 나아지는가?

??? success "연습문제 2 풀이"
    SlowFast은 길 둘을 쓴다. **느린** 길은 낮은 틀 빠르기(예: 2 FPS)로 돌며 갈래를 넉넉히 두어 자리의 뜻을 잘게 담고, **빠른** 길은 높은 틀 빠르기(예: 16 FPS)로 돌며 갈래를 적게 두어 빠른 때의 움직임을 담는다. 자리의 뜻은 더디게 바뀌고(높은 틀 빠르기가 필요 없다) 움직임은 잔 때 잣대에서 일어나므로 이 꾸밈이 잘 든다. 빠른 길은 가볍고(셈의 $\sim$20%) 때의 결을 주며, 느린 길은 자리의 넉넉함을 준다. 옆으로 잇는 이음이 두 길 사이의 소식을 녹여 아우른다.

---

**연습문제 3.**
그림 가르기 얼개(보기로 ResNet)를 영상 이해로 넓힐 때의 주된 어려움은 무엇인가?

??? success "연습문제 3 풀이"
    고갱이 어려움은 이렇다. (1) **셈 값**: 때 차수를 더하면 자료가 $T\times$($T$은 틀 수)만큼 늘어 3차원 누비기가 값비싸진다. (2) **때 모형 짓기**: 2차원 누비기는 틀 하나만 보아 때의 무늬를 놓친다. 2차원 알갱이를 손쉽게 3차원으로 부풀리면(I3D 따위) 값이 비싸다. (3) **길이가 바뀌는 들임**: 영상마다 길이가 달라 때 모으기나 뽑기 꾀가 든다. (4) **멀리 걸친 매임**: 종요로운 일이 수백 틀에 걸칠 수 있어 그 자리 누비기의 받는 밭을 넘어선다. (5) **익힘 자료**: 영상 자료 묶음이 그림 자료 묶음보다 작아 지나치게 맞춰질 걱정이 있다.

---

**연습문제 4.**
영상을 나타내는 데 쓰는 3차원 누비기, (2+1)차원으로 쪼갠 누비기, 때에 걸친 스스로 눈길을 견주어라.

??? success "연습문제 4 풀이"
    | 방식 | 셈 | 때의 범위 | 익히기 |
    |----------|-------------|----------------|----------|
    | **3차원 누비기** | $O(k^3 C^2 THW)$ | 그 자리(틀 $k$개) | 값비싸고 미리 익히기가 든다 |
    | **(2+1)차원 누비기** | $O(k^2 C^2 THW + k C^2 THW)$ | 그 자리 | 다듬기 쉽고 매개변수가 적다 |
    | **때 눈길** | $O(T^2 CHW)$ | 두루 | $T$에 이차이며 너그럽다 |

    3차원 누비기는 힘세지만 값이 비싸다. (2+1)차원 쪼개기는 자리 다루기와 때 다루기를 갈라 정확도를 지키면서 매개변수를 줄인다. 때에 걸친 스스로 눈길은 멀리 떨어진 얽힘을 담아내지만 차례 길이의 제곱으로 늘어난다. 요즘 얼개(보기로 Video Swin Transformer)는 흔히 가까운 자리의 눈길과 층진 꾸밈을 아우른다.

## 정리하며

| 살필 점 | 핵심 |
|--------|------------|
| **영상 꼴** | PyTorch에서는 $V \in \mathbb{R}^{T \times C \times H \times W}$ |
| **읽어 들이기** | 유연함에는 OpenCV, PyTorch에 곧바로 쓰려면 torchvision |
| **표집** | 고루 덮으려면 고른 표집, 불리기에는 마구잡이 |
| **앞손질** | 옮겨 배우기에는 ImageNet 방식 고르게 맞추기 |
| **헤아릴 점** | 모든 틀에서 때의 한결같음을 지킨다 |
