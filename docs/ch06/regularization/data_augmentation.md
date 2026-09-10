# 데이터 증강

데이터 증강은 기존 표본을 변형한 판본을 만들어 학습 데이터셋을 인위적으로 넓히는 정칙화 기법이다. 의미를 보존하는 변환된 데이터를 모델에게 보여 줌으로써 무관한 변이에 대한 불변성을 가르치고 일반화를 크게 개선한다.

---

## 1. 개념적 토대

### 데이터 증강이 통하는 이유

데이터 증강은 학습 데이터가 적다는 근본적인 문제를 다룬다.

1. **실효 데이터셋 크기 증가**: 레이블을 더 붙이지 않고도 학습 예가 늘어난다
2. **불변성 학습**: 모델이 변환에 견고한 특징을 배운다
3. **과적합 감소**: 증강되어 변화하는 데이터는 외우기 더 어렵다
4. **암묵적 정칙화**: 가설 공간을 변환에 불변인 해로 제약한다

### 수학적 관점

증강은 손실에 정칙화 항을 더하는 것으로 볼 수 있다. 확률 $p$으로 적용되는 변환 $T$에 대해 다음과 같다.

$$
\mathcal{L}_{\text{aug}} = \mathbb{E}_{x, y \sim \mathcal{D}} \left[ \mathbb{E}_{T \sim \mathcal{T}} \left[ \ell(f(T(x)), y) \right] \right]
$$

이는 다음을 이끈다.

$$
f(T(x)) \approx f(x) \quad \forall T \in \mathcal{T}
$$

---

## 2. 이미지 증강

### 기하 변환

```python
import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
import numpy as np
from PIL import Image

class GeometricAugmentation:
    """이미지를 위한 표준 기하 증강."""
    
    def __init__(
        self,
        rotation_range: float = 15,
        translate_range: float = 0.1,
        scale_range: tuple = (0.9, 1.1),
        shear_range: float = 10,
        flip_horizontal: bool = True,
        flip_vertical: bool = False
    ):
        transforms = []
        
        # 무작위 아핀 변환 (회전, 평행이동, 크기, 전단).
        # 넷을 한 변환에 몰아 넣는다. 따로 걸면 보간이 네 번 일어나
        # 이미지가 그만큼 뭉개지지만, 아핀 행렬을 한 번에 합쳐 쓰면
        # 보간이 한 번으로 끝난다.
        # 주의: 기본값에서는 이 any(...)가 언제나 참이다. 네 인자를
        # 모두 0이나 항등으로 주어야만 거짓이 되는데, scale_range의
        # 기본값이 (0.9, 1.1)이라 그럴 일이 드물다
        if any([rotation_range, translate_range, scale_range != (1, 1), shear_range]):
            transforms.append(T.RandomAffine(
                degrees=rotation_range,
                translate=(translate_range, translate_range) if translate_range else None,
                scale=scale_range,
                shear=shear_range
            ))
        
        # 뒤집기
        # 좌우 뒤집기만 기본으로 켜져 있고 위아래는 꺼져 있다. 자연
        # 사진에는 중력이라는 방향이 있어 뒤집힌 자동차는 현실에 거의
        # 없지만, 좌우가 바뀐 자동차는 흔하기 때문이다.
        # 위성 사진이나 현미경 사진처럼 위아래가 뜻이 없는 데이터라면
        # flip_vertical을 켜도 된다
        if flip_horizontal:
            transforms.append(T.RandomHorizontalFlip(p=0.5))
        if flip_vertical:
            transforms.append(T.RandomVerticalFlip(p=0.5))
        
        self.transform = T.Compose(transforms)
    
    def __call__(self, image):
        return self.transform(image)

# PyTorch의 표준 기하 변환
geometric_transforms = T.Compose([
    T.RandomRotation(degrees=15),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
        shear=10
    ),
    T.RandomPerspective(distortion_scale=0.2, p=0.5),
    T.RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
])
```

### 색/광도 변환

```python
class PhotometricAugmentation:
    """색과 조명 증강."""
    
    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1
    ):
        # 기하 증강(자르기, 뒤집기)이 "어디에 있는가"를 흔든다면,
        # 광도 증강은 "어떻게 보이는가"를 흔든다. 조명과 카메라가
        # 제각각인 실제 사진에 견디게 하려는 것이다.
        self.transform = T.Compose([
            T.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                # 색상(hue)만 범위를 훨씬 작게 잡는다. 색상을 크게 돌리면
                # 물체의 정체가 바뀌어 버린다(빨간 사과가 초록이 된다).
                # 갈래를 가르는 데 색이 중요한 과제라면 아예 0으로 두어야 한다
                hue=hue
            ),
            # 확률 0.1로 흑백으로 만든다. 색에만 기대어 판단하는 것을 막고
            # 모양과 결도 보게 한다
            T.RandomGrayscale(p=0.1),
            T.RandomAdjustSharpness(sharpness_factor=2, p=0.3),   # 초점 차이
            T.RandomAutocontrast(p=0.3),                          # 자동 대비 보정
            T.RandomEqualize(p=0.3),   # 히스토그램 평활화. uint8 이미지만 받으므로
                                       # ToTensor 앞에 두어야 한다
        ])

    def __call__(self, image):
        return self.transform(image)

# 표준 색 증강. 위보다 세기를 두 배로 키운 설정이며,
# 대조학습(SimCLR 계열)에서 흔히 쓰는 값이다. 자기지도학습에서는
# 같은 사진의 두 모습을 크게 다르게 만들수록 배우는 표현이 좋아진다
color_transforms = T.Compose([
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.RandomGrayscale(p=0.2),
])
```

### 잡음과 흐리기

```python
class NoiseAugmentation:
    """이미지에 여러 종류의 잡음을 더한다."""
    
    def __init__(self, noise_types=['gaussian', 'blur', 'jpeg']):
        self.noise_types = noise_types
    
    def __call__(self, image):
        # PIL 이미지로 들어올 수도 있으므로 텐서로 맞춰 둔다.
        # 아래 연산이 모두 [0,1] 범위의 텐서를 가정하기 때문이다
        if isinstance(image, Image.Image):
            image = T.ToTensor()(image)

        # 매번 하나만 고른다. 셋을 겹쳐 걸면 원본이 너무 망가져
        # 학습에 도움이 되기보다 방해가 된다
        noise_type = np.random.choice(self.noise_types)

        if noise_type == 'gaussian':
            # 센서 잡음을 흉내 낸다. 표준편차 0.05는 [0,1] 눈금에서
            # 눈에 거의 띄지 않을 만큼 작은 값이다
            noise = torch.randn_like(image) * 0.05
            # clamp가 반드시 필요하다. 잡음을 더하면 화솟값이 [0,1]을
            # 벗어나는데, 그대로 두면 뒤의 정규화가 엉뚱한 값을 받는다
            image = torch.clamp(image + noise, 0, 1)

        elif noise_type == 'blur':
            # 초점이 나갔거나 흔들린 사진을 흉내 낸다.
            # sigma를 범위로 주면 흐림 정도가 매번 무작위로 뽑힌다
            image = T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))(image)

        elif noise_type == 'jpeg':
            # JPEG 압축 잡티 모의실험.
            # 실제로 압축했다 푸는 까닭은, JPEG 특유의 블록 무늬를
            # 손으로 흉내 내기 어렵기 때문이다. 웹에서 긁어모은
            # 사진으로 추론할 모델에는 이 잡티에 대한 내성이 중요하다
            quality = np.random.randint(30, 95)   # 낮을수록 심하게 뭉갠다
            pil_img = T.ToPILImage()(image)
            import io
            buffer = io.BytesIO()   # 파일 대신 메모리에 쓴다
            pil_img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)          # 방금 쓴 자리를 다시 읽으려면 처음으로 되돌린다
            image = T.ToTensor()(Image.open(buffer))

        return image
```

### 고급 증강

표준적인 기하 및 광도 변환을 넘어서는 강력한 증강 전략이 여럿 있다. 각각은 별도의 절에서 자세히 다룬다.

- **[컷아웃](cutout.md)** — 직사각형 영역을 무작위로 가려, 모델이 국소 조각에 기대는 대신 물체의 전체 공간 범위를 쓰도록 강제한다. PyTorch에서는 `transforms.RandomErasing`으로 쓸 수 있다.
- **[믹스업](mixup.md)** — 학습 이미지 쌍과 그 레이블을 볼록 결합으로 섞어, 예 사이에서 선형적인 행동을 이끌고 보정을 개선한다.
- **[컷믹스](cutmix.md)** — 한 이미지의 조각을 잘라 다른 이미지에 붙이고 레이블을 비례해서 섞음으로써, (컷아웃 같은) 공간적 가림과 (믹스업 같은) 표본 섞기를 결합한다.

이 기법들은 서로 보완적이며 표준 증강 파이프라인과 함께 쓸 수 있다. 이러한 데이터 증강과 나란히 자주 쓰이는 목표 쪽 정칙화 기법은 **[레이블 평활화](label_smoothing.md)**도 함께 보라.

### 완전한 이미지 증강 파이프라인

```python
def get_train_transforms(image_size: int = 224, augment_level: str = 'standard'):
    """
    증강 수준에 따른 학습용 변환을 얻는다.
    
    인수:
        image_size: 목표 이미지 크기
        augment_level: 'minimal', 'standard', 'aggressive'
    """
    # 세 수준 모두 순서가 같다. 기하 변환 → 색 변환 → ToTensor →
    # Normalize다. ToTensor 앞의 변환들은 PIL 이미지를 다루고 뒤의
    # 변환들은 텐서를 다루므로, 이 경계를 넘나들면 형 오류가 난다.
    # RandomErasing만 예외로 ToTensor 뒤에 와야 한다
    if augment_level == 'minimal':
        return T.Compose([
            T.Resize((image_size, image_size)),
            # 좌우 뒤집기는 거의 언제나 안전한 증강이다. 다만 글자나
            # 숫자를 읽는 과제에서는 뜻을 망가뜨리므로 쓰면 안 된다
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            # ImageNet 전체의 채널별 평균과 표준편차다. 사전 학습 모델을
            # 쓸 때는 그 모델이 학습된 눈금과 맞추어야 하므로 이 값을
            # 그대로 써야 한다
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    elif augment_level == 'standard':
        return T.Compose([
            # Resize와 달리 RandomResizedCrop은 매번 다른 조각을 잘라
            # 크기를 맞춘다. scale=(0.8, 1.0)이니 원본 넓이의 80~100%를
            # 남긴다. 이것 하나가 이 파이프라인에서 가장 크게 일하는
            # 증강이다
            T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            # 조명과 카메라 차이를 흉내 낸다. hue만 값이 작은데, 색조는
            # 조금만 틀어도 물체의 정체성이 바뀌기 때문이다
            # (빨간 사과를 초록으로 만들면 다른 것이 된다)
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # RandomErasing은 텐서에만 동작하므로 반드시 ToTensor 뒤에
            # 온다. 정규화 뒤라 지운 자리가 0으로 채워지는데, 정규화
            # 공간에서 0은 검정이 아니라 그 채널의 평균값이다
            T.RandomErasing(p=0.25)
        ])
    
    elif augment_level == 'aggressive':
        # 데이터가 아주 적을 때만 쓴다. 증강이 세면 학습 분포가 시험
        # 분포에서 멀어져, 정칙화로 얻는 것보다 잃는 것이 커질 수 있다.
        # 학습 정확도가 시험 정확도보다 낮아지면 지나치다는 신호다
        return T.Compose([
            # 넓이의 절반까지 잘라 낸다. 물체가 통째로 잘려 나가
            # 이름표와 맞지 않는 표본이 생길 수도 있다
            T.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            # 위아래 뒤집기는 확률이 낮다. 자연 사진에는 중력이라는
            # 방향이 있어, 뒤집힌 자동차는 현실에 거의 없기 때문이다.
            # 위성 사진이나 현미경 사진이라면 0.5로 올려도 된다
            T.RandomVerticalFlip(p=0.2),
            T.RandomRotation(degrees=30),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            # 다섯 번에 한 번은 색을 없앤다. 모델이 색에만 기대어
            # 판단하지 못하게 하고 모양을 보도록 떠민다
            T.RandomGrayscale(p=0.2),
            # degrees=0인 까닭은 바로 위 RandomRotation이 이미 회전을
            # 맡고 있어서다. 여기서는 평행 이동과 기울임만 더한다
            T.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=15),
            T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.RandomErasing(p=0.5, scale=(0.02, 0.4))
        ])
    
    raise ValueError(f"Unknown augment_level: {augment_level}")

def get_val_transforms(image_size: int = 224):
    """검증/시험용 변환 (증강 없음)."""
    # 무작위 변환이 하나도 없다는 점이 핵심이다. 평가에 증강을 넣으면
    # 같은 이미지에 매번 다른 점수가 나와 견줄 수가 없다. 다만 Resize와
    # Normalize는 남는다. 학습 때와 같은 크기·같은 눈금으로 맞추는
    # 전처리이지 증강이 아니기 때문이다
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
```

---

## 3. 텍스트 증강

### 기본적인 텍스트 증강

```python
import random
import nltk
from nltk.corpus import wordnet

class TextAugmentation:
    """텍스트 증강 기법."""
    
    def __init__(self, aug_prob: float = 0.3):
        self.aug_prob = aug_prob
        # 필요한 NLTK 데이터 내려받기
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
    
    def synonym_replacement(self, text: str, n: int = 1) -> str:
        """낱말 n개를 유의어로 바꾼다."""
        words = text.split()
        new_words = words.copy()
        
        # 유의어가 있는 낱말 얻기
        # set으로 중복을 없애므로 같은 낱말이 두 번 뽑히지 않는다.
        # 다만 아래 치환이 문장 전체에서 그 낱말을 모두 바꾸므로,
        # 같은 낱말이 여러 번 나오면 한꺼번에 바뀐다.
        # 또 set은 순서를 보장하지 않아, 섞기 전부터 이미 실행마다
        # 차례가 달라질 수 있다
        random_word_list = list(set([w for w in words if self._get_synonyms(w)]))
        random.shuffle(random_word_list)
        
        num_replaced = 0
        for random_word in random_word_list:
            synonyms = self._get_synonyms(random_word)
            if synonyms:
                synonym = random.choice(synonyms)
                new_words = [synonym if w == random_word else w for w in new_words]
                num_replaced += 1
            if num_replaced >= n:
                break
        
        return ' '.join(new_words)
    
    def _get_synonyms(self, word: str) -> list:
        """WordNet에서 유의어를 얻는다."""
        # 주의: 품사와 문맥을 보지 않는다. WordNet에서 그 철자에 딸린
        # 뜻을 모조리 긁어 오므로, "bank"가 강둑인지 은행인지 가리지
        # 못한다. 이미지의 좌우 뒤집기와 달리 텍스트 증강은 뜻을
        # 바꿔 놓기 쉬워, 바뀐 문장이 원래 이름표와 맞는지
        # 눈으로 확인해 보는 편이 좋다
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower():
                    synonyms.add(synonym)
        return list(synonyms)
    
    def random_insertion(self, text: str, n: int = 1) -> str:
        """무작위 낱말의 유의어 n개를 무작위 위치에 넣는다."""
        words = text.split()
        new_words = words.copy()
        
        for _ in range(n):
            word = random.choice(words)
            synonyms = self._get_synonyms(word)
            if synonyms:
                insert_pos = random.randint(0, len(new_words))
                new_words.insert(insert_pos, random.choice(synonyms))
        
        return ' '.join(new_words)
    
    def random_swap(self, text: str, n: int = 1) -> str:
        """낱말 n쌍을 무작위로 맞바꾼다."""
        words = text.split()
        new_words = words.copy()
        
        for _ in range(n):
            if len(new_words) >= 2:
                idx1, idx2 = random.sample(range(len(new_words)), 2)
                new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        
        return ' '.join(new_words)
    
    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """확률 p으로 낱말을 무작위로 지운다."""
        words = text.split()
        if len(words) == 1:
            return text
        
        new_words = [w for w in words if random.random() > p]
        if not new_words:
            return random.choice(words)
        
        return ' '.join(new_words)
    
    def augment(self, text: str) -> str:
        """무작위 증강을 적용한다."""
        if random.random() > self.aug_prob:
            return text
        
        aug_type = random.choice(['synonym', 'insert', 'swap', 'delete'])
        
        if aug_type == 'synonym':
            return self.synonym_replacement(text)
        elif aug_type == 'insert':
            return self.random_insertion(text)
        elif aug_type == 'swap':
            return self.random_swap(text)
        else:
            return self.random_deletion(text)
```

### 역번역

```python
class BackTranslation:
    """
    다른 언어로 번역했다가 되돌려 텍스트를 증강한다.
    transformers 라이브러리가 필요하다.
    """
    
    def __init__(self, intermediate_lang: str = 'de'):
        from transformers import MarianMTModel, MarianTokenizer
        
        # 영어에서 중간 언어로
        self.en_to_lang = MarianMTModel.from_pretrained(
            f'Helsinki-NLP/opus-mt-en-{intermediate_lang}'
        )
        self.en_to_lang_tokenizer = MarianTokenizer.from_pretrained(
            f'Helsinki-NLP/opus-mt-en-{intermediate_lang}'
        )
        
        # 중간 언어에서 영어로
        self.lang_to_en = MarianMTModel.from_pretrained(
            f'Helsinki-NLP/opus-mt-{intermediate_lang}-en'
        )
        self.lang_to_en_tokenizer = MarianTokenizer.from_pretrained(
            f'Helsinki-NLP/opus-mt-{intermediate_lang}-en'
        )
    
    def translate(self, text: str, model, tokenizer) -> str:
        inputs = tokenizer(text, return_tensors='pt', padding=True)
        translated = model.generate(**inputs)
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    
    def augment(self, text: str) -> str:
        # 중간 언어로 번역
        intermediate = self.translate(text, self.en_to_lang, self.en_to_lang_tokenizer)
        # 영어로 되돌려 번역
        back_translated = self.translate(intermediate, self.lang_to_en, self.lang_to_en_tokenizer)
        return back_translated
```

---

## 4. 시계열 증강

```python
import numpy as np
import torch

class TimeSeriesAugmentation:
    """시계열 데이터를 위한 증강."""
    
    # 아래 메서드들은 모두 x의 모양을 (시간, 채널)로 가정한다.
    # 축 0이 시간이고 축 1이 변수다. 배치 축이 붙은 텐서를 그대로
    # 넘기면 시간과 배치가 뒤바뀌어 엉뚱한 결과가 나온다.
    @staticmethod
    def jittering(x: np.ndarray, sigma: float = 0.03) -> np.ndarray:
        """정규 잡음을 더한다."""
        # 시각마다 독립인 잡음이라 계열이 거칠어진다. 센서의 측정
        # 오차를 흉내 내는 셈이다. sigma가 크면 신호가 잡음에 묻히므로
        # 데이터의 표준편차에 견주어 작게 잡아야 한다
        return x + np.random.normal(0, sigma, x.shape)
    
    @staticmethod
    def scaling(x: np.ndarray, sigma: float = 0.1) -> np.ndarray:
        """무작위 인수로 배율을 조정한다."""
        # 모양이 (1, 채널)이라 시간축으로 방송된다. 즉 채널마다 배율
        # 하나를 뽑아 계열 전체에 똑같이 곱한다. jittering과 달리
        # 계열의 모양은 그대로 두고 크기만 바꾸는 것이다.
        # 평균이 1인 정규분포에서 뽑으므로 늘어나기와 줄어들기가 반반이다
        factor = np.random.normal(1, sigma, (1, x.shape[1]))
        return x * factor
    
    @staticmethod
    def magnitude_warping(x: np.ndarray, sigma: float = 0.2, 
                          knot: int = 4) -> np.ndarray:
        """매끄러운 곡선으로 크기를 뒤튼다."""
        from scipy.interpolate import CubicSpline
        
        # scaling의 확장이다. 배율 하나를 계열 전체에 곱하는 대신,
        # knot + 2개의 매듭에서만 배율을 뽑고 그 사이를 삼차 스플라인으로
        # 이어 시각마다 다른 배율을 만든다. 스플라인이라 배율이 매끄럽게
        # 변해, 계열의 큰 흐름은 지키면서 굴곡만 달라진다
        orig_steps = np.arange(x.shape[0])
        random_warps = np.random.normal(1.0, sigma, (knot + 2, x.shape[1]))
        warp_steps = np.linspace(0, x.shape[0] - 1, knot + 2)
        
        warper = np.zeros((x.shape[0], x.shape[1]))
        for i in range(x.shape[1]):
            warper[:, i] = CubicSpline(warp_steps, random_warps[:, i])(orig_steps)
        
        return x * warper
    
    @staticmethod
    def time_warping(x: np.ndarray, sigma: float = 0.2, 
                     knot: int = 4) -> np.ndarray:
        """매끄러운 곡선으로 시간축을 뒤튼다."""
        from scipy.interpolate import CubicSpline
        
        orig_steps = np.arange(x.shape[0])
        random_warps = np.random.normal(1.0, sigma, knot + 2)
        warp_steps = np.linspace(0, x.shape[0] - 1, knot + 2)
        
        # magnitude_warping이 세로축(값)을 늘였다면 이쪽은 가로축(시간)을
        # 늘인다. 어떤 구간은 빠르게, 어떤 구간은 느리게 흐르는 셈이다.
        # 걸음걸이나 손동작처럼 같은 동작을 사람마다 다른 속도로 하는
        # 데이터에서 특히 잘 맞는다
        time_warp = CubicSpline(warp_steps, warp_steps * random_warps)(orig_steps)
        # 뒤튼 시각이 계열 밖으로 나가지 않도록 자른다. 여기서 잘리면
        # 그 구간은 시간이 멈춘 것처럼 같은 값이 이어진다
        time_warp = np.clip(time_warp, 0, x.shape[0] - 1)
        
        warped = np.zeros_like(x)
        for i in range(x.shape[1]):
            # 뒤튼 시각 위의 값을 원래 격자 위로 되읽어 온다. 길이가
            # 그대로 유지되어야 배치로 묶을 수 있기 때문이다.
            # np.interp는 두 번째 인자가 오름차순이라고 가정하는데,
            # sigma가 크면 뒤튼 시각이 뒷걸음질 쳐 그 가정이 깨질 수 있다
            warped[:, i] = np.interp(orig_steps, time_warp, x[:, i])
        
        return warped
    
    @staticmethod
    def window_slicing(x: np.ndarray, reduce_ratio: float = 0.9) -> np.ndarray:
        """이어진 구간을 무작위로 잘라 내어 크기를 맞춘다."""
        target_len = int(x.shape[0] * reduce_ratio)
        if target_len < 1:
            return x
        
        # 이어진 한 토막만 남기고 나머지를 버린다. 이미지의 무작위
        # 자르기에 해당하며, 계열의 일부만 보고도 판단하도록 떠민다
        start = np.random.randint(0, x.shape[0] - target_len + 1)
        sliced = x[start:start + target_len]
        
        # 원래 길이로 되돌리기
        indices = np.linspace(0, target_len - 1, x.shape[0])
        resized = np.zeros_like(x)
        for i in range(x.shape[1]):
            resized[:, i] = np.interp(indices, np.arange(target_len), sliced[:, i])
        
        return resized
    
    @staticmethod
    def permutation(x: np.ndarray, max_segments: int = 5) -> np.ndarray:
        """계열의 구간들을 무작위로 뒤섞는다."""
        n_segments = np.random.randint(2, max_segments + 1)
        segment_len = x.shape[0] // n_segments
        
        segments = []
        for i in range(n_segments):
            start = i * segment_len
            # 마지막 토막만 끝까지 가져간다. 나눗셈에서 생긴 나머지를
            # 여기에 몰아주는 것이라, 마지막 토막이 다른 것보다 길 수 있다
            end = (i + 1) * segment_len if i < n_segments - 1 else x.shape[0]
            segments.append(x[start:end])
        
        # 앞의 증강들과 성격이 다르다. 이쪽은 시간 순서를 대놓고
        # 깨뜨린다. 순서가 곧 뜻인 과제(추세 예측 등)에서는 이름표와
        # 맞지 않는 표본을 만들어 내므로 쓰면 안 된다. 어떤 무늬가
        # 들어 있는지만 중요한 과제에 맞는 기법이다
        np.random.shuffle(segments)
        return np.concatenate(segments, axis=0)
```

---

## 5. 표 형식 데이터의 증강

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

class TabularAugmentation:
    """표 형식 데이터를 위한 증강 기법."""
    
    @staticmethod
    def add_gaussian_noise(X: np.ndarray, noise_scale: float = 0.1) -> np.ndarray:
        """특징에 정규 잡음을 더한다."""
        # axis=0으로 열마다 표준편차를 따로 구한다. 표 데이터는 나이,
        # 소득처럼 눈금이 제각각이라 고정된 잡음을 쓰면 어떤 열은
        # 잡음에 묻히고 어떤 열은 꿈쩍도 하지 않는다. 열의 산포에
        # 비례해 흔들어야 모든 특징이 고르게 증강된다.
        # 주의: 이 방식은 열을 서로 독립으로 흔들므로 특징 사이의
        # 상관을 깨뜨린다. 키와 몸무게처럼 얽힌 열이 있으면
        # 있을 수 없는 표본이 만들어질 수 있다
        std = np.std(X, axis=0) * noise_scale
        noise = np.random.normal(0, std, X.shape)
        return X + noise
    
    @staticmethod
    def feature_dropout(X: np.ndarray, dropout_rate: float = 0.1) -> np.ndarray:
        """특징을 무작위로 그 열의 평균으로 바꾼다."""
        X_aug = X.copy()
        mask = np.random.random(X.shape) < dropout_rate
        col_means = np.mean(X, axis=0)
        X_aug[mask] = np.tile(col_means, (X.shape[0], 1))[mask]
        return X_aug
    
    @staticmethod
    def smote(X: np.ndarray, y: np.ndarray, 
              minority_class: int = 1,
              k_neighbors: int = 5,
              n_synthetic: int = None) -> tuple:
        """
        SMOTE: 합성 소수 클래스 과표집 기법.
        
        소수 클래스를 위한 합성 표본을 만든다.
        """
        minority_mask = y == minority_class
        X_minority = X[minority_mask]
        
        if n_synthetic is None:
            majority_count = np.sum(~minority_mask)
            minority_count = np.sum(minority_mask)
            n_synthetic = majority_count - minority_count
        
        if n_synthetic <= 0 or len(X_minority) < k_neighbors:
            return X, y
        
        # 가장 가까운 이웃 k개 찾기
        nn = NearestNeighbors(n_neighbors=k_neighbors + 1)
        nn.fit(X_minority)
        
        synthetic_samples = []
        for _ in range(n_synthetic):
            # 무작위 소수 클래스 표본
            idx = np.random.randint(len(X_minority))
            sample = X_minority[idx]
            
            # 무작위 이웃
            distances, indices = nn.kneighbors([sample])
            neighbor_idx = np.random.choice(indices[0][1:])  # 자기 자신 제외
            neighbor = X_minority[neighbor_idx]
            
            # 보간
            alpha = np.random.random()
            synthetic = sample + alpha * (neighbor - sample)
            synthetic_samples.append(synthetic)
        
        X_synthetic = np.array(synthetic_samples)
        y_synthetic = np.full(n_synthetic, minority_class)
        
        return np.vstack([X, X_synthetic]), np.concatenate([y, y_synthetic])

class MixupTabular:
    """표 형식 데이터를 위한 믹스업."""
    
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
    
    def __call__(self, X: np.ndarray, y: np.ndarray) -> tuple:
        n = len(X)
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha, n)
        else:
            lam = np.ones(n)
        
        lam = lam.reshape(-1, 1)
        index = np.random.permutation(n)
        
        X_mixed = lam * X + (1 - lam) * X[index]
        y_mixed = lam.squeeze() * y + (1 - lam.squeeze()) * y[index]
        
        return X_mixed, y_mixed
```

---

## 6. AutoAugment와 학습된 증강

```python
class RandAugment:
    """
    RandAugment: 간략화한 학습된 증강 정책.
    
    참고: Cubuk 등, "RandAugment: Practical Automated Data Augmentation"
    """
    
    def __init__(self, n_ops: int = 2, magnitude: int = 9):
        """
        인수:
            n_ops: 적용할 증강 연산의 수
            magnitude: 증강의 세기 (0~30)
        """
        self.n_ops = n_ops
        self.magnitude = magnitude
        
        # 쓸 수 있는 연산 정의
        self.ops = [
            'identity', 'autocontrast', 'equalize', 'rotate',
            'solarize', 'color', 'posterize', 'contrast',
            'brightness', 'sharpness', 'shear_x', 'shear_y',
            'translate_x', 'translate_y'
        ]
    
    def __call__(self, image):
        # 무작위 연산 고르기.
        # RandAugment의 요점은 손잡이를 둘로 줄인 데 있다. 연산마다
        # 확률과 세기를 따로 맞추던 앞 세대(AutoAugment)와 달리,
        # n_ops와 magnitude 두 값만 정하면 되므로 탐색이 훨씬 싸다.
        # sample은 중복 없이 뽑으므로 한 이미지에 같은 연산이 두 번
        # 걸리지 않는다. 'identity'가 목록에 들어 있는 덕에
        # 아무것도 하지 않는 선택지도 확률을 갖는다
        ops = random.sample(self.ops, self.n_ops)
        
        for op in ops:
            image = self._apply_op(image, op, self.magnitude)
        
        return image
    
    def _apply_op(self, img, op: str, magnitude: int):
        """증강 연산 하나를 적용한다."""
        # 세기를 실제 값으로
        mag = magnitude / 30.0  # 0~1로 정규화
        
        if op == 'identity':
            return img
        elif op == 'autocontrast':
            return F.autocontrast(img)
        elif op == 'equalize':
            return F.equalize(img)
        elif op == 'rotate':
            angle = mag * 30  # 최대 30도
            return F.rotate(img, angle)
        elif op == 'solarize':
            threshold = int((1 - mag) * 255)
            return F.solarize(img, threshold)
        elif op == 'posterize':
            bits = int(8 - mag * 4)
            return F.posterize(img, bits)
        elif op == 'contrast':
            factor = 1 + mag * 0.9 * random.choice([-1, 1])
            return F.adjust_contrast(img, factor)
        elif op == 'brightness':
            factor = 1 + mag * 0.9 * random.choice([-1, 1])
            return F.adjust_brightness(img, factor)
        elif op == 'sharpness':
            factor = 1 + mag * 0.9 * random.choice([-1, 1])
            return F.adjust_sharpness(img, factor)
        elif op == 'shear_x':
            shear = mag * 0.3 * random.choice([-1, 1])
            return F.affine(img, angle=0, translate=[0, 0], scale=1, shear=[shear, 0])
        elif op == 'shear_y':
            shear = mag * 0.3 * random.choice([-1, 1])
            return F.affine(img, angle=0, translate=[0, 0], scale=1, shear=[0, shear])
        elif op == 'translate_x':
            shift = int(mag * img.size[0] * 0.3) * random.choice([-1, 1])
            return F.affine(img, angle=0, translate=[shift, 0], scale=1, shear=[0, 0])
        elif op == 'translate_y':
            shift = int(mag * img.size[1] * 0.3) * random.choice([-1, 1])
            return F.affine(img, angle=0, translate=[0, shift], scale=1, shear=[0, 0])
        
        return img
```

---

## 7. 실무 지침

### 증강 고르기

| 데이터 종류 | 권장 증강 |
|-----------|--------------------------|
| 자연 이미지 | 뒤집기, 잘라내기, 색 흔들기, RandAugment |
| 의료 영상 | 회전, 크기 조정 (뒤집기는 조심) |
| 문서 | 약간의 회전, 잡음, 흐리기 |
| 시계열 | 흔들기, 크기 조정, 시간 뒤틀기 |
| 텍스트 | 유의어 치환, 역번역 |
| 표 형식 | 잡음 주입, 불균형에는 SMOTE |

### 증강의 강도

- **너무 약하면**: 정칙화의 이점이 거의 없다
- **너무 강하면**: 의미 정보를 없앨 수 있다
- **좋은 관행**: 적당한 세기로 시작하고 과적합이 계속되면 올린다

### 검증 집합

**검증/시험 데이터는 결코 증강하지 마라.** 공정한 평가를 위해 원래 표본을 쓴다.

---

## 연습문제

**연습문제 1.**
이미지 분류를 위한 표준 데이터 증강 기법을 열거하고 각각이 북돋우는 불변성을 설명하라.

??? success "연습문제 1 풀이"
    좌우 뒤집기는 좌우 불변성, 무작위 잘라내기는 평행이동 불변성, 색 흔들기는 조명 불변성, 회전은 회전 불변성, 무작위 지우기는 가림에 대한 견고성을 준다. 각 증강은 어떤 변환이 레이블을 바꾸지 말아야 하는지에 대한 사전 지식을 담는다.

---

**연습문제 2.**
`torchvision.transforms`를 써서 PyTorch에서 사용자 정의 증강 파이프라인을 구현하라.

??? success "연습문제 2 풀이"
    ```python
    from torchvision import transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    ```

---

**연습문제 3.**
데이터 증강이 손실 함수에 정칙화 항을 더하는 것과 동등한 이유를 설명하라.

??? success "연습문제 3 풀이"
    입력 $x$을 변환 $T$으로 증강하면 $f(T(x)) \approx f(x)$이라는 제약이 더해진다. 이는 손실에 $\mathbb{E}_T[L(f(T(x)), y)]$을 더하는 것과 동등하며, 증강 변환에 대한 민감도에 벌점을 준다.

---

**연습문제 4.**
(a) 의료 영상, (b) 위성 영상, (c) 텍스트 데이터에 알맞은 증강을 설계하라.

??? success "연습문제 4 풀이"
    (a) 의료: 회전, 탄성 변형, 명암 크기 조정(좌우가 구분되는 장기에는 좌우 뒤집기를 쓰지 않는다). (b) 위성: 회전, 뒤집기, 색 흔들기, 크기 변화. (c) 텍스트: 유의어 치환, 무작위 삽입/삭제, 역번역, 문장 섞기.

## 정리하며

이 마당은 개념적 토대、이미지 증강、텍스트 증강、시계열 증강을 차례로 짚었다.

**참고 문헌**

1. Shorten, C., & Khoshgoftaar, T. M. (2019). A Survey on Image Data Augmentation for Deep Learning. *Journal of Big Data*, 6(1), 60.
2. Cubuk, E. D., et al. (2020). RandAugment: Practical Automated Data Augmentation. *NeurIPS*.
3. Cubuk, E. D., et al. (2019). AutoAugment: Learning Augmentation Strategies from Data. *CVPR*.
4. Zhang, H., et al. (2018). mixup: Beyond Empirical Risk Minimization. *ICLR*.
5. DeVries, T., & Taylor, G. W. (2017). Improved Regularization of CNNs with Cutout. *arXiv*.
6. Yun, S., et al. (2019). CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features. *ICCV*.
