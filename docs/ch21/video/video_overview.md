# 비디오 입력: 읽기·샘플링·전처리

비디오는 순서가 있는 프레임 열 V=(I_1,…,I_T)이며 각 프레임은 높이 H·너비 W·채널 C를 갖는다. C는 클래스 수가 아니라 RGB 같은 입력 채널 수다. 어떤 프레임을 어떤 시간 간격으로 골라 어떤 축 순서로 모델에 넣는지를 명확히 해야 한다.

## 1. 프레임 수와 실제 시간

고정 프레임률 f에서 T개 프레임의 재생 길이는 대략 T/f초다. 첫 프레임과 마지막 프레임 timestamp 사이 간격은 보통 (T−1)/f이며 두 표현의 기준이 다르다. 가변 프레임률이나 디코딩 누락이 있으면 실제 timestamp를 확인해야 한다. 같은 프레임 수여도 stride s에 따라 관찰 간격 s/f와 전체 시간 범위가 달라진다.

비디오 하나는 THWC·TCHW·CTHW, 배치는 BTCHW·BCTHW 등으로 표현할 수 있다. ‘PyTorch의 비디오는 항상 BTCHW’라는 고정 규칙은 없다. Conv3d는 일반적으로 BCTHW이고 아래 전처리 함수는 TCHW를 기준으로 한다. 축의 이름과 입력 모델의 계약을 함께 기록한다.

## 2. OpenCV와 torchvision 읽기

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

OpenCV의 프레임은 BGR이므로 RGB로 바꾸어 T×H×W×3 NumPy 배열로 모은다. 이 구현에는 파일 열기·빈 프레임·fps0 검사나 메타데이터와 실제 프레임 수의 일치 확인이 없다. 전체 프레임을 메모리에 올리므로 긴 영상은 디코딩·저장 비용도 고려한다. 고정 fps 메타데이터로 계산한 duration을 모든 파일의 정확한 timestamp 길이라고 가정하지 않는다.

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

이 torchvision I/O 형식은 기본 THWC 비디오와 오디오·정보를 반환하는 API를 가정한다. 지원 API·코덱 의존성은 설치된 버전에서 확인한다. 이 메서드와 OpenCV 메서드의 정보 사전 키·오디오 처리도 같지 않다. 읽기 backend를 바꿀 때 반환 형태까지 같은 것으로 취급하면 안 된다.

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

`convert_to_pytorch_format`은 uint8 RGB THWC를 float TCHW로 옮기고255로 나누는 함수다. 이미0~1인 입력에 다시 적용하면 값이255분의1로 줄어든다. BGR변환은 포함하지 않고, 채널·차원 검증도 없다. permute로 축을 옮기는 것과 reshape로 모양만 바꾸는 것은 다르다.

## 3. 전체 구간 샘플링과 stride

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

위 uniform_sampling은 linspace(0,T−1,n)을 정수로 바꿔 양 끝을 포함한다. n>1에서 인덱스는 floor(k(T−1)/(n−1))로 표현할 수 있고, n=1이면 첫 프레임이다. floor(kT/n)과는 다른 규칙이다. 짧은 영상에서 요청 수가 많으면 인덱스가 중복된다. 빈 입력·유효하지 않은 요청 수는 별도 규칙이 필요하다.

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

stride_sampling은 0번부터 양의 s 간격으로 선택한다. 영상 전체를 일정 수로 압축하는 uniform과 달리 출력 프레임 수가 원본 길이에 따라 달라진다. 큰 stride는 빠른 액션을 놓칠 수 있으며 s=0·음수 등 허용 범위를 검증해야 한다.

## 4. 여러 clip과 무작위 프레임

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

이 dense_sampling은 연속 clip의 시작점을 균일하게 두며 clip 하나면 중앙 시작점을 사용한다. 여러 clip은 입력 길이·clip 수에 따라 겹칠 수도 있고 떨어질 수도 있다. 원본이 짧으면 검은 프레임을 뒤에 추가해 길이를 맞춘다. 다만 padding은 기본 CPU float32로 생성되어 원본 GPU·다른 dtype과 일치하지 않을 수 있다. 마스크가 없어 추가 프레임과 실제 검은 장면을 모델이 자동 구분하지도 않는다.

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

random_sampling은 복원 추출한 인덱스를 정렬한다. 중복 없는 부분집합이나 연속 구간 선택이 아니다. 실제 프레임 순서를 유지해도 인접 표본의 간격은 불규칙하며 같은 프레임이 반복될 수 있다. 학습 어그멘테이션과 평가의 고정 샘플링 규칙을 구분한다.

## 5. 정규화·crop·resize

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

정규화는 TCHW의 RGB 각 채널에 (x−μ)/σ를 적용한다. ImageNet 통계가 모든 비디오 사전 학습 모델의 통계와 같지는 않다. mean·std가 CPU에 있어 CUDA 입력과 바로 연산하면 장치 오류가 날 수 있고, 임의 dtype도 확인해야 한다. denormalize는 역선형변환만 하며 표시 범위 clamp는 포함하지 않는다. 정규화 앞뒤의 0 패딩은 서로 다른 값을 뜻한다.

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

한 crop 위치를 모든 프레임에 적용해 우발적 흔들림을 줄인다. crop이 원본보다 크면 random 범위가 유효하지 않거나 슬라이싱이 예상과 다르게 작동한다. 시간 범위·공간 크기와 지원 position을 검증해야 한다. 목표 상자·마스크·flow가 있다면 같은 기하 변환과 각각의 레이블·벡터 규칙도 필요하다.

```python
import torch.nn.functional as F

def resize_video(video: torch.Tensor, 
                 target_size: tuple) -> torch.Tensor:
    """
    모든 틀을 목표 크기로 바꾼다.
    
    인수:
        video: 들임 텐서 (T, C, H, W)
        target_size: (타깃 높이, 타깃 너비)
        
    반환값:
        크기를 바꾼 텐서 (T, C, target_h, target_w)
    """
    T, C, H, W = video.shape
    target_h, target_w = target_size
    
    # 배치 처리를 위해 꼴 바꾸기
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

resize는 T를 이미지 배치처럼 두고 공간 축만 bilinear 보간한다. 시간 프레임 수는 바꾸지 않으며 지정한 가로세로 크기로 늘이면 원본 종횡비가 변형될 수 있다. pad·crop을 사용하는 사전 학습 전처리와 동일하지 않다. 정수 클래스 마스크에는 이 보간을 그대로 적용하지 않는다.

## 6. 표시와 다음 단계

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

표시할 때 정규화 값을 바로 0~1로 clamp하면 색이 달라질 수 있으므로 필요하면 먼저 역정규화한다. 위 함수는 num_frames=1이면 단일 Axes를 반복하려 해 오류가 날 수 있고, 빈 영상도 검증하지 않는다. 그림 출력은 프레임 선택을 이해하는 자료이며 학습·정확도 평가는 아니다.

이후 [3D 컨볼루션](02_3d_convolution.md), [Two-Stream](two_stream.md), [SlowFast](slowfast.md)에 연결할 때 각 모델의 축 순서·입력 범위·샘플링 계약을 다시 확인한다. 같은 전처리 이름이라도 다른 절의 짧은 영상 처리·장치 이동 구현이 같다는 보장은 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
THWC 배열을 Conv3d 배치 입력으로 바꾸는 과정을 설명하라.

</div>

??? success "풀이"

    RGB 색상과 0~255 범위를 확인한 뒤 THWC→CTHW로 축을 옮기고 배치를 추가해 BCTHW를 만든다. 본문 함수는 TCHW까지만 반환하므로 추가 축 정렬이 필요하다. permute와 reshape의 의미를 구분한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
uniform_sampling의 n>1 인덱스와 n>T일 때의 결과를 설명하라.

</div>

??? success "풀이"

    인덱스는 $\lfloor k(T-1)/(n-1)\rfloor$로 양 끝을 포함한다. n>T이면 중복이 생길 수 있다. n=1은 첫 프레임이고 빈 영상은 별도 처리해야 한다. 일정 시간 간격의 stride 샘플링과는 출력 수·관찰 간격의 계약이 다르다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span>
짧은 GPU 영상의 dense_sampling과 normalize에 어떤 장치 문제가 있는가?

</div>

??? success "풀이"

    padding과 정규화 통계가 CPU 기본 텐서여서 GPU 입력과 일치하지 않는다. 입력 device·dtype에 맞게 만들어야 한다. 패딩의 유효성 마스크와 정규화 전후의 패딩 값도 명시한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
TCHW resize가 시간 축도 줄이는가? 시각화 함수의 경계 사례도 설명하라.

</div>

??? success "풀이"

    resize는 T를 배치로 취급해 H·W만 바꾸고 T는 그대로다. 시각화는 num_frames=1에서 Axes가 목록이 아니라 반복 오류가 날 수 있고 빈 영상도 검증하지 않는다. 정규화된 프레임은 표시 전에 필요한 역변환을 해야 한다.



## 정리하며

비디오 입력은 축·색상·범위와 함께 실제 시간·샘플링·패딩을 정의한다. 균일·stride·연속 clip·복원 추출은 서로 다르며, 장치·dtype·전처리 통계와 목표 변환도 모델의 계약에 맞춰야 한다.
