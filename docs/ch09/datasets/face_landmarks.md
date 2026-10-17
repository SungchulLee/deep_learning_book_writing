# 얼굴 랜드마크

이미지 파일과 CSV의 얼굴 랜드마크 좌표를 함께 불러오는 사용자 정의 Dataset이다. 공간 변환을 이미지와 좌표에 동시에 적용하는 것이 핵심이다.

## 1. 코드

```python
"""
얼굴 랜드마크 — 사용자 정의 Dataset, 변환, DataLoader
============================================================
얼굴 랜드마크 데이터를 위해 ``Dataset``을 상속하여 만들고, ``Rescale``,
``RandomCrop``, ``ToTensor`` 변환을 직접 쓰는 법을 보인다.

PyTorch 공식 데이터 적재 튜토리얼에 바탕을 두었다.

사용법
-----
    python face_landmarks.py
"""

import os
import urllib.request
import zipfile

# ========================================================================
# 메인
# ========================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from skimage import io, transform
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

DOWNLOAD_DIR = "./data"

# ---------------------------------------------------------------------------
# 내려받기
# ---------------------------------------------------------------------------


def download():
    dest = os.path.join(DOWNLOAD_DIR, "faces")
    if os.path.isdir(dest):
        return
    url = "https://download.pytorch.org/tutorial/faces.zip"
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    zip_path = os.path.join(DOWNLOAD_DIR, "faces.zip")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DOWNLOAD_DIR)
    os.remove(zip_path)
    print(f"Dataset downloaded and extracted to: {DOWNLOAD_DIR}")


# ---------------------------------------------------------------------------
# 사용자 정의 데이터셋
# ---------------------------------------------------------------------------


class FaceLandmarksDataset(Dataset):
    """얼굴 랜드마크 데이터셋."""

    def __init__(self, csv_file, root_dir, transform=None):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = np.array([self.landmarks_frame.iloc[idx, 1:]])
        landmarks = landmarks.astype("float").reshape(-1, 2)
        sample = {"image": image, "landmarks": landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample


# ---------------------------------------------------------------------------
# 사용자 정의 변환
# ---------------------------------------------------------------------------


class Rescale:
    """가로세로비를 지키며 이미지를 주어진 크기로 조정한다."""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]
        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        landmarks = landmarks * [new_w / w, new_h / h]

        return {"image": img, "landmarks": landmarks}


class RandomCrop:
    """표본의 이미지를 무작위로 잘라 낸다."""

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top : top + new_h, left : left + new_w]
        landmarks = landmarks - [left, top]

        return {"image": image, "landmarks": landmarks}


class ToTensor:
    """표본의 ndarray를 텐서로 바꾼다."""

    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]
        # 넘파이: H × W × C  →  토치: C × H × W
        image = image.transpose((2, 0, 1))
        return {
            "image": torch.from_numpy(image),
            "landmarks": torch.from_numpy(landmarks),
        }


# ---------------------------------------------------------------------------
# 데이터 적재
# ---------------------------------------------------------------------------


def load_data():
    dataset = FaceLandmarksDataset(
        csv_file="data/faces/face_landmarks.csv",
        root_dir="data/faces/",
        transform=transforms.Compose([
            Rescale(256),
            RandomCrop(224),
            ToTensor(),
        ]),
    )
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=0)
    return dataloader


# ---------------------------------------------------------------------------
# 시각화
# ---------------------------------------------------------------------------


def show_one_image(df_landmarks, n=65):
    img_name = df_landmarks.iloc[n, 0]
    image = io.imread(os.path.join("data/faces/", img_name))
    landmarks = np.asarray(df_landmarks.iloc[n, 1:]).astype("float").reshape(-1, 2)

    _, ax = plt.subplots()
    ax.imshow(image)
    ax.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker=".", c="r")
    ax.axis("off")
    plt.show()


def show_four_images(dataloader):
    _, axes = plt.subplots(1, 4, figsize=(12, 3))
    i = -1
    for d in dataloader:
        images = d["image"]
        landmarks = d["landmarks"]
        for image, landmark in zip(images, landmarks):
            i += 1
            axes[i].imshow(image.permute(1, 2, 0))
            axes[i].scatter(landmark[:, 0], landmark[:, 1], s=10, marker=".", c="r")
            axes[i].axis("off")
            if i == 3:
                break
        if i == 3:
            break
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------


def main():
    download()

    df = pd.read_csv(f"{DOWNLOAD_DIR}/faces/face_landmarks.csv")
    show_one_image(df, n=65)

    dataloader = load_data()
    show_four_images(dataloader)


if __name__ == "__main__":
    main()
```

**출력:**

```
Dataset downloaded and extracted to: ./data
```

## 2. 논의

FaceLandmarksDataset은 이미지와 `(점 수, 2)` 좌표를 사전으로 반환한다. Rescale은 정수 크기일 때 짧은 변을 기준으로 비율을 유지하지만 튜플 크기일 때는 지정한 높이·너비로 바꾼다. 좌표의 x에는 너비 비율, y에는 높이 비율을 곱한다.

RandomCrop은 이미지의 시작 위치를 빼 좌표 원점을 옮긴다. 일부 점은 잘린 영역 밖에 남을 수 있으므로 실제 과제에서는 유효 점 마스크나 별도 처리 규칙이 필요하다. 현재 코드의 randint 상한은 제외되므로 같은 크기 자르기나 마지막 유효 위치의 처리도 점검해야 한다.

ToTensor는 HWC 이미지를 CHW로 전치하지만 torchvision ToTensor처럼 자동으로 픽셀 스케일과 자료형을 모두 맞추는 연산은 아니다. 변환 결과의 dtype과 값 범위를 모델 입력에 맞춘다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
이미지 너비를 두 배, 높이를 세 배로 바꾸면 좌표는 어떻게 되는가?

</div>

??? success "풀이"
    각 점 $(x,y)$를 $(2x,3y)$로 옮긴다. 이미지의 높이·너비 순서와 좌표의 x·y 순서를 혼동하지 않는다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
왼쪽 10, 위 20에서 자르면 좌표와 유효성을 어떻게 처리하는가?

</div>

??? success "풀이"
    좌표는 $(x-10,y-20)$이다. 새 이미지 범위 밖의 점을 단순히 올바른 내부 좌표로 간주하면 안 된다. 제외·가림 표시 등 과제의 규칙이 필요하다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
사용자 정의 ToTensor가 출력하는 자료형을 확인해야 하는 이유를 설명하라.

</div>

??? success "풀이"
    torch.from_numpy는 원본 배열의 dtype을 반영한다. resize나 좌표 처리에서 float64가 만들어질 수 있으므로 모델의 자료형과 맞추고 픽셀 스케일도 확인한다.

## 정리하며

공간 어그멘테이션은 이미지와 랜드마크를 같은 좌표계로 변환해야 한다. 크기 조정과 자르기의 규칙, 범위 밖 점 처리, 텐서 자료형과 값 범위를 함께 확인한다.
