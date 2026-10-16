# 비디오 기초: 프레임 읽기와 샘플링

비디오는 시간 순서가 있는 이미지 묶음이다. 파일에서 읽는 배열, 모델이 받는 텐서와 실제 시간 간격을 연결하는 것이 출발점이다. 이 예제는 VideoLoader·FrameSampler·VideoPreprocessor로 읽기·샘플링·전처리를 설명한다. 기본 시연은 파일을 읽는 대신 움직이는 막대의 합성 프레임을 만든다.

## 1. 코드

??? note "코드"

    ```python
    """
    영상 이해 — 첫스텝 수준
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
        2. 잣수마다의 때에 걸친 액션 담아내기
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

            TSN(때 토막 넷트웍)과 비슷한 구조에 쓰인다
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
            고르게 맞추기 파라메터로 앞손질기를 첫자리매김한다.

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
        plt.savefig('01_video_frames.png', 
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
        plt.savefig('01_optical_flow.png',
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
           - 촘촘: 때 넷트웍을 위한 여러 토막

        4. 앞손질:
           - 고르게 맞추기: 평균/표준편차로 익히기를 든든하게 한다
           - 잘라내기: 때(틀)와 자리(구역)
           - 불리기: 두루 통함을 낫게 한다

        5. 영상과 그림:
           - 영상에는 때 차원이 있다(복잡함이 더해진다)
           - 표집 전략이 모델이 무엇을 배우는지를 바꾼다
           - 앞손질은 때의 한결같음을 지켜야 한다

        다음: 자리와 때의 피처를 배우는 3차원 컨볼루션을 살펴본다!
        """)


    if __name__ == "__main__":
        main()
    ```

## 2. 차원·색상·자료형

OpenCV로 읽은 프레임은 H×W×3 BGR이므로 코드에서 RGB로 변환한 뒤 T×H×W×3으로 모은다. `load_video_to_tensor`는 resize 후 float로 바꾸고 255로 나누어 **T×C×H×W**의 0~1 텐서를 반환한다. Conv3d의 일반적인 배치 입력은 **B×C×T×H×W**이므로 배치 차원과 축 순서를 따로 맞춰야 한다. reshape만으로 시간과 채널을 서로 바꾸면 안 된다.

`load_video_torchvision`의 기본 반환 비디오도 T×H×W×C이며 이 메서드는 자체적으로 채널 우선·0~1 변환을 하지 않는다. torchvision과 OpenCV 메서드의 반환 정보 키도 video_fps와 fps처럼 다르다. 생성자 backend는 저장만 하므로 이름만 바꿔 메서드 호출이 자동 전환되지는 않는다. 설치 버전의 비디오 I/O 지원과 코덱 의존성을 확인한다.

파일 열기 실패·빈 비디오·fps=0은 현재 검사하지 않는다. 메타데이터의 frame_count와 실제 디코딩한 프레임 수가 다를 수 있으며 frame_count/fps는 고정 프레임률을 가정한 길이 추정이다. 가변 프레임률 영상은 프레임 인덱스와 timestamp를 구분해야 한다. 전체 프레임을 메모리에 올리므로 긴 비디오에서는 저장량도 커진다.

## 3. 프레임 수와 시간 간격

`uniform_sampling`은 처음부터 끝까지 균일한 인덱스를 선택하고, 요청 수가 원래 프레임보다 많으면 정수 인덱스가 반복될 수 있다. `random_sampling`도 복원 추출 뒤 정렬하므로 중복 없는 무작위 부분집합은 아니다. 원래 순서는 유지하지만 프레임 간 간격은 일정하지 않을 수 있다.

반면 `load_video_to_tensor(num_frames=...)`는 요청 수가 더 작을 때만 줄인다. 짧은 영상에 프레임을 채워 같은 길이로 만드는 기능은 없다. dense_sampling도 원본이 clip_length보다 짧으면 원본 하나를 그대로 반환하므로 고정 길이 배치를 보장하지 않는다. 프레임 반복·패딩·마스크 등 짧은 영상 처리 규칙이 별도로 필요하다.

stride 샘플링은 양의 간격을 가정한다. 고정 fps에서 stride s로 선택하면 인접 표본의 시간 간격은 s/fps다. 같은 16프레임이라도 fps·stride가 다르면 덮는 시간과 액션의 의미가 다르다. 균일 샘플링은 전체 비디오, 연속 clip은 짧은 구간을 보는 차이가 있으므로 과제에 맞게 선택한다.

## 4. 정규화와 일관된 crop

`normalize`는 RGB 채널별 고정 평균·표준편차로 (x−μ)/σ를 계산한다. ImageNet 통계가 모든 비디오 모델의 사전 학습 전처리와 일치하지는 않는다. 코드의 mean 텐서는 1×3×1×1이므로 T×3×H×W 계약을 기준으로 사용한다. 통계는 장치로 옮기지만 임의 입력 dtype까지 자동 일치시키는 코드는 아니다.

공간 crop은 한 번 뽑은 위치를 모든 프레임에 적용해 인위적인 카메라 흔들림을 줄인다. crop이 원본보다 크거나 시간 범위를 벗어나면 현재 슬라이싱은 기대보다 작은 결과·빈 결과를 만들거나 random crop의 범위 오류를 낼 수 있다. 유효 범위 검증이 필요하다. denormalize의 clamp는 표시 범위를 자르는 처리이며 범위 밖 값까지 완전히 역변환하는 것은 아니다.

## 5. 합성 시연과 시각화의 한계

기본 시연의 optical flow는 프레임에서 추정하지 않고 수평 성분 5·수직 성분 0으로 직접 만든다. 따라서 학습이나 흐름 추정 정확도를 보여 주지 않는다. HSV 표시에서는 크기를 영상 내부 min-max로 정규화하므로 모든 크기가 같은 이 예제는 밝기가 0이 되어 검게 보일 수 있다. 서로 다른 그림의 밝기를 절대 이동량으로 비교하지 않는다.

이 절에는 학습 가능한 분류기나 역전파가 없다. 샘플링 결과와 그림은 입력 표현을 이해하기 위한 것이며 기본 main은 명시적인 실행 조건문 아래에서 호출된다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
파일에서 읽은 T×H×W×C와 Conv3d 입력의 차이를 설명하라.

</div>

??? success "연습문제 1 풀이"

    색상 순서와 0~255/0~1 범위를 확인한 뒤 축을 B×C×T×H×W로 배치한다. 이 loader의 텐서 메서드는 T×C×H×W이므로 배치 추가와 시간·채널 순서 정렬이 필요하다. 단순 view로 축을 바꾸면 원소 의미가 달라진다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
짧은 영상에서 모든 함수가 정확히 num_frames 프레임을 반환하는가?

</div>

??? success "연습문제 2 풀이"

    아니다. uniform sampling은 중복 선택으로 요청 수를 만들 수 있지만 load_video_to_tensor는 원본보다 많은 요청을 채우지 않는다. dense sampling도 짧으면 원본 하나를 반환한다. 빈 입력·양의 요청 수·패딩 또는 반복 폴리시를 명시해야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
동일한 flow 크기 5의 시각화가 검게 나올 수 있는 이유를 설명하라.

</div>

??? success "연습문제 3 풀이"

    cv2.normalize의 min-max 정규화에서 모든 크기가 같으면 상대 대비가 없어 밝기를 0으로 만들 수 있다. 검은 그림이 반드시 실제 이동량 0을 뜻하지 않는다. 절대 크기를 비교하려면 공통 표시 스케일이 필요하다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 VideoLoader 시험 조각의 한계를 검토하고 적절한 확인 항목을 설명하라.

</div>

??? success "연습문제 4 풀이"

    VideoLoader는 nn.Module이 아니고 __call__·parameters·학습 경로가 없다. 아래 미정의 입력과 backward 검사는 이 클래스 계약에 맞지 않는다. 읽기 실패·fps=0·실제 프레임 수·RGB 변환·짧은 영상·샘플링 중복·crop 범위와 텐서 축 순서를 확인하는 명세가 적절하다. 기존 조각은 완성된 시험이 아니다.

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


## 정리하며

비디오 입력은 축 순서·색상·범위와 함께 fps·샘플링 간격을 정의해야 한다. 이 예제의 함수들은 짧은 영상과 요청 프레임 수를 서로 다르게 처리한다. 정규화·crop의 계약과 합성 flow 표시의 한계를 확인하고 모델 입력에 연결한다.
