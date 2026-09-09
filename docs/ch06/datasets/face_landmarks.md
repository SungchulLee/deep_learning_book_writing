# 얼굴 랜드마크

표준이 아닌 데이터 형식을 다룰 때에는 Dataset을 상속하여 직접 만들어야 한다. 이 스크립트는 얼굴 랜드마크 검출을 위한 데이터셋을 만드는 법을 보이며, 크기 조정, 무작위 잘라내기, NumPy 배열을 텐서로 바꾸는 사용자 정의 변환도 함께 다룬다. PyTorch 공식 데이터 적재 튜토리얼의 방식을 따른다.

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

## 2. 논의

사용자 정의 데이터셋은 세 메서드를 구현해야 한다. 준비를 위한 `__init__`, 데이터셋의 크기를 주는 `__len__`, 표본을 꺼내는 `__getitem__`이다. 이 데이터셋은 이미지 파일 이름과 랜드마크 좌표가 담긴 CSV 파일을 읽고, scikit-image로 이미지를 불러온 뒤, 사용자 정의 변환을 차례로 적용한다.

사용자 정의 변환(`Rescale`, `RandomCrop`, `ToTensor`)은 이미지와 랜드마크를 함께 담은 사전에 작용하여 공간 변환이 둘 모두에 한결같이 적용되게 한다. 크기 조정은 가로세로비를 지키고, 무작위 잘라내기는 공간 증강을 넣으며, 텐서 변환은 HWC 형식을 PyTorch가 요구하는 CHW 형식으로 바꾼다.

`transforms.Compose` 파이프라인은 함수형 프로그래밍의 합성 방식을 본떠 이 연산들을 차례로 이어 붙인다. 이런 설계 덕분에 데이터셋 클래스를 건드리지 않고도 변환을 넣거나 빼거나 순서를 바꾸기 쉽다.

## 연습문제

**연습문제 1.**
코드를 따라가며 쓰인 주요 자료 구조를 찾아라. 각각에 대해 자료형, (해당한다면) 모양, 파이프라인에서의 구실을 적어라.

??? success "연습문제 1 풀이"
    코드를 꼼꼼히 읽으며 변수 대입마다 살펴본다. 텐서는 `.shape`과 `.dtype`을 확인하고, 클래스는 `__init__`의 매개변수와 `forward`/`__call__`의 서명을 확인한다. 이름, 자료형, 모양, 구실을 열로 하는 표에 정리한다.

---


**연습문제 2.**
오류 처리와 입력 검증을 넣도록 코드를 고쳐라. 이 코드를 실전에 쓸 수 있게 하려면 어떤 검사를 더하겠는가?

??? success "연습문제 2 풀이"
    입력에 자료형 검사(`isinstance`), 모양 검증(`assert tensor.dim() == expected`), 값 범위 검사(예: 확률이 [0,1] 안인지)를 넣고, 입출력 연산은 try-except로 감싼다. 빈 배치나 NaN 같은 경계 상황에는 경고를 남긴다. 매개변수와 반환값의 자료형을 적은 독스트링을 붙인다.

---


**연습문제 3.**
직접 고른 새로운 쓰임새를 지원하도록 코드를 확장하라. 무엇을 왜 바꿀지 설명하라.

??? success "연습문제 3 풀이"
    알맞은 확장을 하나 고른다(예: 다른 데이터셋, 지표 추가, 새 모델 변형). 필요한 변경을 설명한다. 새 임포트, 클래스 정의 수정, 초매개변수 갱신, 새로운 시각화나 기록 등이다. 핵심 변경을 구현하고 간단한 시험으로 올바름을 확인한다.

## 정리하며

**다룬 것** — 얼굴 랜드마크

사용자 정의 데이터셋은 세 메서드를 구현해야 한다.

핵심 클래스는 `FaceLandmarksDataset`, `Rescale`, `RandomCrop`, `ToTensor`이며 앞의 연습문제 3개로 직접 확인할 수 있다.
