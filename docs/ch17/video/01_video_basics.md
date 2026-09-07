# 단원 34: 영상 이해

단원 34: 영상 이해 — 첫걸음 수준. 파일 01: 영상 기초 — 영상 읽어 들이고 다루기

이 단원은 셈틀 보기라는 더 넓은 맥락 안에서 영상 이해를 살펴본다. 여기 짠 것은 요즘 체계에서 쓰는 얼개의 새로움과 익히기 전략을 보여 주는 실전 PyTorch 코드이다.

## 코드

```python
"""
단원 34: 영상 이해 — 첫걸음 수준
파일 01: 영상 기초 — 영상 읽어 들이고 다루기

이 파일은 영상 자료를 다루는 근본을 다룬다:
- 영상을 틀의 때 차례로 이해하기
- 여러 방법으로 영상 읽어 들이기
- 기본 영상 앞손질 연산
- PyTorch에서 영상 자료 나타내기
- 틀 뽑기와 표집 전략

수학적 바탕:
- 영상 V은 틀 T개의 차례이다: V = {I_1, I_2, ..., I_T}
- 틀마다 I_t ∈ ℝ^(H×W×C), 여기서 H=높이, W=너비, C=채널
- 영상 텐서: PyTorch에서 V ∈ ℝ^(T×C×H×W)((T, C, H, W) 꼴)
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.io import read_video, write_video
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from typing import Tuple, List, Optional
import warnings

# ========================================================================
# 메인
# ========================================================================
warnings.filterwarnings('ignore')


#=============================================================================
# 1부: 영상 나타내기와 읽어 들이기
#=============================================================================

class VideoLoader:
    """
    여러 뒷단을 받쳐 주는 두루 쓰는 영상 읽어 들이기 도구.
    
    속성:
        video_path: 영상 파일의 경로
        backend: 읽어 들이기 뒷단('torchvision', 'opencv', 'decord')
    """
    
    def __init__(self, video_path: str, backend: str = 'opencv'):
        """
        영상 읽개를 첫자리매김한다.
        
        인수:
            video_path: 영상 파일의 경로
            backend: 쓸 뒷단('torchvision', 'opencv', 'decord')
        """
        self.video_path = video_path
        self.backend = backend
        
    def load_video_torchvision(self) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        torchvision 뒷단으로 영상을 읽어 들인다.
        
        반환값:
            video: 값이 [0, 255]이고 꼴이 (T, H, W, C)인 영상 텐서
            audio: 있으면 소리 텐서
            info: 영상 메타자료(초당 틀 수, 길이 등)를 담은 사전
            
        수학으로 나타내기:
            V ∈ ℝ^(T×H×W×3), 여기서 T = 틀의 개수
        """
        # torchvision으로 영상 읽어 들이기
        # 유의: torchvision은 (T, H, W, C) 꼴로 돌려준다
        video, audio, info = read_video(
            self.video_path,
            pts_unit='sec'  # 때 도장에 초를 쓴다
        )
        
        print(f"Video shape: {video.shape}")  # (T, H, W, C)
        print(f"FPS: {info['video_fps']}")
        print(f"Duration: {video.shape[0] / info['video_fps']:.2f} seconds")
        
        return video, audio, info
    
    def load_video_opencv(self) -> Tuple[np.ndarray, dict]:
        """
        OpenCV 뒷단으로 영상을 읽어 들인다(더 유연하고 널리 받쳐 준다).
        
        반환값:
            frames: 꼴이 (T, H, W, C)인 NumPy 배열
            info: 영상 메타자료를 담은 사전
            
        OpenCV는 효율적이고 여러 꼴을 받쳐 주어 실전에 좋다.
        """
        # 영상 파일 열기
        cap = cv2.VideoCapture(self.video_path)
        
        # 영상의 성질 얻기
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 모든 틀 읽기
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # BGR을 RGB로 바꾸기(OpenCV는 BGR로 읽는다)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        # numpy 배열로 바꾸기
        frames = np.array(frames)  # 꼴: (T, H, W, 3)
        
        info = {
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration': frame_count / fps
        }
        
        print(f"Loaded {len(frames)} frames")
        print(f"Shape: {frames.shape}, FPS: {fps:.2f}")
        
        return frames, info
    
    def load_video_to_tensor(self, 
                            num_frames: Optional[int] = None,
                            target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
        """
        영상을 읽어 들여 PyTorch 텐서 꼴로 바꾼다.
        
        인수:
            num_frames: 뽑을 틀의 개수(None이면 모든 틀)
            target_size: 틀을 (H, W)로 바꾼다
            
        반환값:
            video_tensor: 꼴 (T, C, H, W) — PyTorch 표준 꼴
            
        꼴 바꾸기:
            OpenCV: 값이 [0, 255]인 (T, H, W, C)
            → PyTorch: 값이 [0, 1]인 (T, C, H, W)
        """
        frames, info = self.load_video_opencv()
        
        # 정해졌으면 틀 뽑기
        if num_frames is not None and num_frames < len(frames):
            indices = np.linspace(0, len(frames) - 1, num_frames).astype(int)
            frames = frames[indices]
        
        # 틀 크기 바꾸기
        resized_frames = []
        for frame in frames:
            # cv2.resize는 (H, W)가 아니라 (W, H)를 바란다
            resized = cv2.resize(frame, (target_size[1], target_size[0]))
            resized_frames.append(resized)
        
        # 텐서로 바꾸고 고르게 맞추기
        # 원래: 값이 [0, 255]인 (T, H, W, C) numpy 배열
        # 결과: 값이 [0, 1]인 (T, C, H, W) 텐서
        video_tensor = torch.from_numpy(np.array(resized_frames)).float()
        video_tensor = video_tensor.permute(0, 3, 1, 2)  # (T, H, W, C) → (T, C, H, W)
        video_tensor = video_tensor / 255.0  # [0, 1]로 고르게 맞추기
        
        return video_tensor


#=============================================================================
# 2부: 틀 표집 전략
#=============================================================================

class FrameSampler:
    """
    영상 이해를 위한 여러 틀 표집 전략을 짠다.
    
    표집은 다음에 결정적이다:
    1. 셈 값 다스리기(영상에는 틀이 많다)
    2. 잣수마다의 때에 걸친 움직임 담아내기
    3. 길이가 들쭉날쭉한 영상 다루기
    """
    
    @staticmethod
    def uniform_sampling(video: torch.Tensor, num_frames: int) -> torch.Tensor:
        """
        영상 전체에서 틀을 고루 뽑는다.
        
        인수:
            video: 들임 영상 텐서 (T, C, H, W)
            num_frames: 뽑을 틀의 개수
            
        반환값:
            뽑은 영상 텐서 (num_frames, C, H, W)
            
        수학으로 나타내기:
            뽑는 번호: k = 0, 1, ..., num_frames-1에 대해 i_k = floor(k * T / num_frames)
            이러면 영상 길이 전체에 고루 떨어지게 된다
        """
        T = video.shape[0]
        
        # 고루 떨어진 번호 만들기
        # linspace는 [0, T-1]에서 고루 떨어진 num_frames개 점을 준다
        indices = torch.linspace(0, T - 1, num_frames).long()
        
        # 영상 텐서에서 번호로 뽑기
        sampled_video = video[indices]
        
        print(f"Uniform sampling: {T} → {num_frames} frames")
        print(f"Sampled indices: {indices.tolist()[:10]}...")
        
        return sampled_video
    
    @staticmethod
    def random_sampling(video: torch.Tensor, num_frames: int) -> torch.Tensor:
        """
        틀을 마구잡이로 뽑는다(자료 불리기에 쓸모 있다).
        
        인수:
            video: 들임 영상 텐서 (T, C, H, W)
            num_frames: 뽑을 틀의 개수
            
        반환값:
            뽑은 영상 텐서 (num_frames, C, H, W)
            
        마구잡이 표집은 모델이 때에 안 바뀜을 배우도록 돕는다
        """
        T = video.shape[0]
        
        # 되넣으며 마구잡이 번호 뽑기
        indices = torch.randint(0, T, (num_frames,))
        
        # 때 차례를 지키려 정렬
        indices, _ = torch.sort(indices)
        
        sampled_video = video[indices]
        
        return sampled_video
    
    @staticmethod
    def temporal_stride_sampling(video: torch.Tensor, 
                                 stride: int) -> torch.Tensor:
        """
        stride번째 틀마다 뽑는다.
        
        인수:
            video: 들임 영상 텐서 (T, C, H, W)
            stride: 때 성큼(보기로 stride=2이면 두 틀마다 하나)
            
        반환값:
            뽑은 영상 텐서
            
        보기:
            stride=1: 모든 틀 [0, 1, 2, 3, 4, 5, 6, 7, 8]
            stride=2: 두 틀마다  [0, 2, 4, 6, 8]
            stride=4: 네 틀마다  [0, 4, 8]
        """
        # 성큼을 준 자르기 쓰기
        sampled_video = video[::stride]
        
        print(f"Stride sampling (stride={stride}): "
              f"{video.shape[0]} → {sampled_video.shape[0]} frames")
        
        return sampled_video
    
    @staticmethod
    def dense_sampling(video: torch.Tensor, 
                      clip_length: int,
                      num_clips: int = 1) -> List[torch.Tensor]:
        """
        영상에서 촘촘한 때 토막을 뽑는다.
        
        인수:
            video: 들임 영상 텐서 (T, C, H, W)
            clip_length: 토막마다 잇닿은 틀의 개수
            num_clips: 뽑을 토막의 개수
            
        반환값:
            영상 토막의 목록. 저마다 꼴이 (clip_length, C, H, W)
            
        TSN(때 토막 그물)과 비슷한 얼개에 쓰인다
        영상을 토막으로 나누고 토막마다 뽑는다
        """
        T = video.shape[0]
        
        if T < clip_length:
            # 영상이 clip_length보다 짧으면 덧대거나 원본을 돌려준다
            return [video]
        
        clips = []
        
        # 토막 길이 셈하기
        segment_length = (T - clip_length) // num_clips
        
        for i in range(num_clips):
            # 토막의 시작
            start_idx = i * segment_length
            
            # 자료 불리기를 위해 토막 안에서 마구잡이로 뽑기
            random_offset = torch.randint(0, segment_length + 1, (1,)).item()
            clip_start = start_idx + random_offset
            clip_end = clip_start + clip_length
            
            # 토막 뽑아내기
            if clip_end <= T:
                clip = video[clip_start:clip_end]
                clips.append(clip)
        
        return clips


#=============================================================================
# 3부: 영상 앞손질과 불리기
#=============================================================================

class VideoPreprocessor:
    """
    영상에 맞춘 앞손질 연산.
    
    앞손질은 다음에 결정적이다:
    1. 들임 고르게 맞추기(평균, 표준편차)
    2. 해상도가 들쭉날쭉한 것 다루기
    3. 튼튼함을 위한 자료 불리기
    """
    
    def __init__(self, 
                 mean: List[float] = [0.485, 0.456, 0.406],
                 std: List[float] = [0.229, 0.224, 0.225]):
        """
        고르게 맞추기 매개변수로 앞손질기를 첫자리매김한다.
        
        인수:
            mean: 채널마다의 평균(ImageNet 붙박이)
            std: 채널마다의 표준편차
            
        고르게 맞추기 식:
            x_고르게맞춤 = (x - 평균) / 표준편차
        """
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
    
    def normalize(self, video: torch.Tensor) -> torch.Tensor:
        """
        영상 텐서를 고르게 맞춘다.
        
        인수:
            video: 값이 [0, 1]인 들임 텐서 (T, C, H, W)
            
        반환값:
            고르게 맞춘 영상 텐서
            
        고르게 맞추기는 익히기를 든든하게 하고 더 빨리 모이게 한다
        """
        # 퍼뜨리기를 위해 차원 늘리기: (1, 3, 1, 1) → (T, 3, H, W)
        mean = self.mean.to(video.device)
        std = self.std.to(video.device)
        
        normalized = (video - mean) / std
        
        return normalized
    
    def denormalize(self, video: torch.Tensor) -> torch.Tensor:
        """
        그려 보려고 고르게 맞추기를 되돌린다.
        
        인수:
            video: 고르게 맞춘 텐서 (T, C, H, W)
            
        반환값:
            고르게 맞추기를 되돌려 [0, 1]로 만든 영상 텐서
        """
        mean = self.mean.to(video.device)
        std = self.std.to(video.device)
        
        denormalized = video * std + mean
        denormalized = torch.clamp(denormalized, 0, 1)
        
        return denormalized
    
    def temporal_crop(self, 
                     video: torch.Tensor,
                     start_frame: int,
                     num_frames: int) -> torch.Tensor:
        """
        때 잘라내기(잇닿은 틀의 차례)를 뽑는다.
        
        인수:
            video: 들임 텐서 (T, C, H, W)
            start_frame: 시작 틀 번호
            num_frames: 뽑을 틀의 개수
            
        반환값:
            잘라낸 영상 텐서 (num_frames, C, H, W)
        """
        end_frame = start_frame + num_frames
        cropped = video[start_frame:end_frame]
        
        return cropped
    
    def spatial_crop(self,
                    video: torch.Tensor,
                    crop_size: Tuple[int, int],
                    position: str = 'center') -> torch.Tensor:
        """
        모든 틀에 자리 잘라내기를 한다.
        
        인수:
            video: 들임 텐서 (T, C, H, W)
            crop_size: (자를 높이, 자를 너비)
            position: 'center', 'random', 또는 'top_left'
            
        반환값:
            자리를 잘라낸 영상
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
        
        cropped = video[:, :, top:top+crop_h, left:left+crop_w]
        
        return cropped


#=============================================================================
# 4부: 영상 그려 보기
#=============================================================================

def visualize_frames(video: torch.Tensor,
                    num_frames: int = 8,
                    figsize: Tuple[int, int] = (16, 4)):
    """
    영상에서 뽑은 틀을 그려 본다.
    
    인수:
        video: 영상 텐서 (T, C, H, W)
        num_frames: 보여 줄 틀의 개수
        figsize: 그림 크기
    """
    T = video.shape[0]
    
    # 틀을 고루 뽑기
    if T > num_frames:
        indices = torch.linspace(0, T - 1, num_frames).long()
        frames_to_show = video[indices]
    else:
        frames_to_show = video
        num_frames = T
    
    # 아래 그림 만들기
    fig, axes = plt.subplots(1, num_frames, figsize=figsize)
    if num_frames == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        # 보여 줄 수 있는 꼴로 바꾸기
        frame = frames_to_show[i].permute(1, 2, 0)  # (C, H, W) → (H, W, C)
        frame = torch.clamp(frame, 0, 1)  # [0, 1] 범위 보장
        
        ax.imshow(frame.cpu().numpy())
        ax.axis('off')
        ax.set_title(f'Frame {indices[i].item() if T > num_frames else i}')
    
    plt.tight_layout()
    plt.savefig('/home/claude/34_video_understanding/01_video_frames.png', 
                dpi=150, bbox_inches='tight')
    print(f"Visualization saved to 01_video_frames.png")
    plt.close()


def visualize_optical_flow(flow: np.ndarray,
                          figsize: Tuple[int, int] = (12, 4)):
    """
    빛 흐름을 RGB로 그려 본다(뒤 단원 미리보기).
    
    인수:
        flow: (u, v) 성분을 갖는 빛 흐름 배열 (H, W, 2)
        figsize: 그림 크기
    """
    # 그려 보려고 흐름을 HSV로 바꾸기
    # 각 → 색상, 크기 → 밝기
    h, w = flow.shape[:2]
    
    # 크기와 각 셈하기
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # HSV 그림 만들기
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2  # 각 → 색상
    hsv[..., 1] = 255  # 가득 찬 채도
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # 크기 → 밝기
    
    # RGB로 바꾸기
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    plt.figure(figsize=figsize)
    plt.imshow(rgb)
    plt.title('Optical Flow Visualization')
    plt.axis('off')
    plt.savefig('/home/claude/34_video_understanding/01_optical_flow.png',
                dpi=150, bbox_inches='tight')
    print(f"Flow visualization saved to 01_optical_flow.png")
    plt.close()


#=============================================================================
# 5부: 쓰는 보기와 보임
#=============================================================================

def demonstrate_video_loading():
    """
    여러 가지 영상 읽어 들이기와 다루기 재주를 보여 준다.
    """
    print("="*80)
    print("VIDEO BASICS DEMONSTRATION")
    print("="*80)
    
    # 보여 주려고 인공 영상 만들기
    print("\n1. Creating synthetic video...")
    # 꼴: (T, C, H, W) = (30, 3, 128, 128)
    # 128x128 RGB 영상 30틀
    T, C, H, W = 30, 3, 128, 128
    
    # 움직이는 무늬가 있는 영상 만들기
    video = torch.zeros(T, C, H, W)
    for t in range(T):
        # 움직이는 세로 막대 만들기
        bar_position = int((t / T) * W)
        video[t, :, :, max(0, bar_position-5):min(W, bar_position+5)] = 1.0
        
        # 실제 같도록 잡음 더하기
        video[t] += torch.randn(C, H, W) * 0.1
    
    video = torch.clamp(video, 0, 1)
    
    print(f"Created synthetic video: {video.shape}")
    print(f"Min value: {video.min():.3f}, Max value: {video.max():.3f}")
    
    # 표집 전략 보여 주기
    print("\n2. Testing sampling strategies...")
    sampler = FrameSampler()
    
    # 고른 표집
    uniform_sampled = sampler.uniform_sampling(video, num_frames=10)
    print(f"After uniform sampling: {uniform_sampled.shape}")
    
    # 성큼 표집
    stride_sampled = sampler.temporal_stride_sampling(video, stride=3)
    print(f"After stride sampling: {stride_sampled.shape}")
    
    # 촘촘한 표집
    clips = sampler.dense_sampling(video, clip_length=8, num_clips=3)
    print(f"Dense sampling produced {len(clips)} clips")
    print(f"Each clip shape: {clips[0].shape}")
    
    # 앞손질 보여 주기
    print("\n3. Testing preprocessing...")
    preprocessor = VideoPreprocessor()
    
    normalized = preprocessor.normalize(video)
    print(f"After normalization - Mean: {normalized.mean():.3f}, Std: {normalized.std():.3f}")
    
    denormalized = preprocessor.denormalize(normalized)
    print(f"After denormalization - Mean: {denormalized.mean():.3f}")
    
    # 자리 잘라내기
    cropped = preprocessor.spatial_crop(video, crop_size=(96, 96), position='center')
    print(f"After spatial crop: {cropped.shape}")
    
    # 틀 그려 보기
    print("\n4. Visualizing frames...")
    visualize_frames(video, num_frames=8)
    
    # 인공 빛 흐름 만들고 그려 보기
    print("\n5. Creating synthetic optical flow...")
    flow = np.zeros((H, W, 2), dtype=np.float32)
    # 가로 흐름 무늬 만들기
    flow[:, :, 0] = 5.0  # u 성분(가로)
    flow[:, :, 1] = 0.0  # v 성분(세로)
    visualize_optical_flow(flow)
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)


def main():
    """
    영상 기초를 보여 주는 주된 실행 함수.
    """
    print(__doc__)
    
    # 재현성을 위해 난수 씨앗 고정
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 시연 실행
    demonstrate_video_loading()
    
    # 요약
    print("\n" + "="*80)
    print("KEY TAKEAWAYS")
    print("="*80)
    print("""
    1. 영상 꼴: 영상은 틀의 차례 V = {I_1, I_2, ..., I_T}이다
       - PyTorch 꼴: (T, C, H, W), 여기서 T=때, C=채널, H=높이, W=너비
    
    2. 읽어 들이는 방법:
       - torchvision: PyTorch에 곧바로 붙고 쓰기 쉽다
       - OpenCV: 유연하고 널리 받쳐 주며 실전에 쓸 수 있다
       - 필요와 이미 갖춘 얼거리에 따라 고른다
    
    3. 틀 표집:
       - 고름: 영상 전체에 고루 떨어짐(가장 흔함)
       - 마구잡이: 자료 불리기용
       - 성큼: 효율을 위해 틀을 건너뛴다
       - 촘촘: 때 그물을 위한 여러 토막
    
    4. 앞손질:
       - 고르게 맞추기: 평균/표준편차로 익히기를 든든하게 한다
       - 잘라내기: 때(틀)와 자리(구역)
       - 불리기: 두루 통함을 낫게 한다
    
    5. 영상과 그림:
       - 영상에는 때 차원이 있다(복잡함이 더해진다)
       - 표집 전략이 모델이 무엇을 배우는지를 바꾼다
       - 앞손질은 때의 한결같음을 지켜야 한다
    
    다음: 자리와 때의 특징을 배우는 3차원 누비기를 살펴본다!
    """)


if __name__ == "__main__":
    main()```

## 논의

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 넓혀진다. 초매개변수와 구조의 변형, 다른 데이터셋으로 실험해 보면 이해가 깊어지고 컴퓨터 비전 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
코드를 끝까지 읽고 핵심 꾸밈 결정을 가려내어라. 구체적인 짜기 고름 세 가지를 들고 저마다 왜 영상 이해에 알맞은지 설명하여라.

??? success "연습문제 1 풀이"
    설계 결정은 구현마다 다르지만 흔히 다음이 포함된다. (1) 활성화 함수의 선택 — ReLU 계열은 포화되지 않는 경사를 주어 학습을 빠르게 한다. (2) 정규화 전략 — 배치 정규화는 내부 공변량 이동을 줄여 학습을 안정시킨다. (3) 잔차 연결 — 있을 경우 건너뛰는 경로를 제공하여 깊은 신경망에서도 경사가 흐르게 한다. 각 선택은 표현력, 계산 비용, 학습 안정성 사이의 절충을 반영한다.

---

**연습문제 2.**
입력이 기대하는 모양과 자료형을 갖는지 확인하도록 주 함수나 클래스에 입력 검증을 추가하라. 잘못된 입력에는 유익한 오류 메시지를 내라.

??? success "연습문제 2 풀이"
    `forward` 메서드(또는 해당 함수)의 첫머리에 다음과 같은 검사를 추가한다. `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`와 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'`. 모양을 검증할 때는 중요한 차원을 확인한다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 유익한 오류 메시지는 디버깅 속도를 크게 높이고 코드를 재사용하기에도 더 견고하게 만든다.

---

**연습문제 3.**
이 구현에서 생길 수 있는 실패 양상 두 가지를 서술하고, 각각을 어떻게 진단하고 고칠지 설명하라.

??? success "연습문제 3 풀이"
    흔한 실패 양상은 다음과 같다. (1) **경사 소실/폭발** — 경사의 노름을 지켜보아 진단한다(`torch.nn.utils.clip_grad_norm_`을 쓰거나 층마다 `param.grad.norm()`을 기록한다). 경사 자르기, 더 나은 초기화(Xavier/Kaiming), 또는 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** — 학습 손실은 줄어드는데 검증 손실이 늘어나면 진단된다. 정칙화(드롭아웃, 가중치 감쇠, 데이터 증강)나 모델 용량 축소로 고친다. 이런 문제를 일찍 잡아내려면 언제나 학습 지표와 검증 지표를 함께 살펴라.

---

**연습문제 4.**
단원 34: 영상 이해의 짜기를 확인하는 두루 살피는 시험 함수를 쓰라. 빈 들임, 원소가 하나인 들임, 아주 큰 들임, 값이 극단인 들임(0, 아주 큰 수)을 비롯한 모서리 경우를 시험하여라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_videoloader():
        model = VideoLoader(...)
        # 보통의 입력
        assert model(normal_input).shape == expected_shape
        # 원소가 하나인 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값 (넘침을 확인한다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 경사의 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    경사의 흐름을 시험하는 것은 그 구조가 처음부터 끝까지 이어지는 학습을 지원하는지 확인하는 데 특히 중요하다.
