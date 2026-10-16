# wav2vec 2.0과 음성 표현의 자기지도 학습

[wav2vec 2.0](https://arxiv.org/abs/2006.11477)은 원시 음성에서 잠재 피처를 얻고 일부 구간을 마스킹한 문맥 표현으로 양자화 목표를 구별하는 자기지도 학습 방법이다. 전사 없이 사전 학습한 뒤 레이블이 있는 음성으로 인식 head를 파인튜닝한다. 다음은 그 전체 목적을 구현한 모델이 아니라 컨볼루션·트랜스포머와 선형 출력의 축약 구조다.

## 1. 코드와 기존 출력

```python
#!/usr/bin/env python3
'''
Wav2Vec 2.0 — 말소리 알아듣기를 위한 스스로 살피는 러닝
논문: "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations" (2020)
핵심: 말소리를 위한 맞대어 배우기, 양자화, 변환기 인코더
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class FeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 512, 10, stride=5, bias=False),
            nn.GroupNorm(512, 512),
            nn.GELU(),
            nn.Conv1d(512, 512, 3, stride=2, bias=False),
            nn.GroupNorm(512, 512),
            nn.GELU(),
            nn.Conv1d(512, 512, 3, stride=2, bias=False),
            nn.GroupNorm(512, 512),
            nn.GELU(),
        )
    
    def forward(self, x):
        return self.conv_layers(x)

class Wav2Vec2(nn.Module):
    def __init__(self, d_model=768, n_layers=12, n_heads=12):
        super().__init__()
        self.feature_extractor = FeatureEncoder()
        self.feature_projection = nn.Linear(512, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        self.quantizer = nn.Linear(d_model, 320)
    
    def forward(self, x):
        # x: [batch, 1, time]
        features = self.feature_extractor(x)
        features = features.transpose(1, 2)
        features = self.feature_projection(features)
        
        context = self.transformer(features)
        
        quantized = self.quantizer(context)
        
        return context, quantized

if __name__ == "__main__":
    model = Wav2Vec2()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    x = torch.randn(2, 1, 16000)
    print(f"Input: {x.shape}")
```

**기존 출력:**

```
Parameters: 87,275,584
Input: torch.Size([2, 1, 16000])
```

## 2. 원시 파형과 시간 해상도

입력은 B×1×L의 모노 파형이며 Conv1d가 시간 축을 줄인다. 현재 FeatureEncoder는 kernel10·3·3과 stride5·2·2의 세 컨볼루션, GroupNorm·GELU를 사용한다. BatchNorm이 아니다. GroupNorm512그룹은512채널 각각의 시간 축 통계를 사용하는 구성이다.

L16000이면 출력 길이는3199→1599→799다. 총 stride20이고 리셉티브 필드는40샘플이다. 실제16kHz 파형일 때 명목상 간격은1.25ms로 원 논문의7컨볼루션·총 stride320·약20ms 간격과 다르다. 단순히 입력 숫자가16000개라는 사실만으로 원래 녹음의 샘플링률이16kHz로 정해지지는 않는다. 파일의 실제 주파수와 resampling·정규화를 맞춰야 한다.

## 3. 양자화·마스킹과 실제 출력

코드는 컨볼루션 피처를 B×T×512로 바꾸고 d_model차원으로 사영해 TransformerEncoder에 넣는다. 기본 입력 B2·L16000에서는 context가2×799×768, 두 번째 출력은2×799×320이다. `quantizer`라는 이름은 **문맥 피처의 선형 레이어**일 뿐 이산 codebook·product quantization·Gumbel 선택이 없다. 두 번째 텐서도 연속 로짓이며 양자화된 음성 단위가 아니다.

원래 사전 학습의 시간 마스킹·잠재 피처에서의 양자화 목표·negative sampling·대조 로스·codebook 다양성 로스도 없다. 명시적인 시간 위치 embedding·padding mask도 제공하지 않는다. 따라서 현재 순전파를 원 논문의 자기지도 사전 학습이나 전사 모델로 설명하면 안 된다. main 시연은 모델을 만들고 파라메터·입력 모양을 출력할 뿐 forward를 호출하지도 않는다.

마스킹 대조 목적은 정답 잠재 목표를 다른 후보와 구별하도록 학습하는 선택이다. 파형의 모든 세부를 복원하는 목적과 다르지만, 음성 내용만 남기거나 화자·잡음을 완전히 제거한다는 보장은 없다. 복원 기반 자기지도 학습이 언제나 열등하거나 대조 학습이 항상 더 효율적이라고 일반화하지 않는다.

## 4. 전사 head와 CTC 디코딩

파인튜닝에서는 문맥 표현에 전사 어휘의 선형 head를 붙일 수 있다. 레이블 있는 파형과 전사·실제 입력/목표 길이, CTC의 log-probability·blank 계약을 맞춰 학습해야 한다. 이 파일의 파인튜닝 예제도 head만 선언하고 학습을 수행하지 않는다.

greedy CTC는 위치별 argmax의 연속 반복을 합친 뒤 blank를 제거하는 경로 축약이다. 예를 들어 a·a·blank·a는 aa가 되어 blank가 구분한 반복은 남는다. 이는 전체 가능한 CTC 경로를 합한 최적 전사 탐색과는 다르다. padding 프레임의 실제 길이도 적용해야 한다. 아래 연습문제의 무작위 입력·웨이트에서 나온 문자열은 인식 정확도의 근거가 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
아래 StereoFeatureEncoder의 입력 계약과 원래 FeatureEncoder 대비 파라메터 증가를 계산하라.

</div>

??? success "연습문제 1 풀이"

    스테레오는 B×2×L을 입력하며 첫 컨볼루션의 웨이트가512×1×10개 더 생겨5120개 증가한다. 세 GroupNorm의 γ·β와 뒤 컨볼루션은 같다. 모노 FeatureEncoder는1581056개, 스테레오는1586176개다. 전체 Wav2Vec2에 쓰려면 feature_extractor도 실제 교체해야 한다.

    ```python
    class StereoFeatureEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_layers = nn.Sequential(
                nn.Conv1d(2, 512, 10, stride=5, bias=False),  # 들임 채널 2개
                nn.GroupNorm(512, 512),
                nn.GELU(),
                nn.Conv1d(512, 512, 3, stride=2, bias=False),
                nn.GroupNorm(512, 512),
                nn.GELU(),
                nn.Conv1d(512, 512, 3, stride=2, bias=False),
                nn.GroupNorm(512, 512),
                nn.GELU(),
            )
        
        def forward(self, x):
            return self.conv_layers(x)
    
    # 첫 컨볼루션 레이어의 파라메터가 1*512*10 = 5120에서 2*512*10 = 10240으로 바뀐다
    # 첫 레이어에서만 파라메터가 5120개 늘어난다.
    ```


---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
대조 학습의 역할과 현재 quantizer가 원 논문의 양자화 모듈과 다른 점을 설명하라.

</div>

??? success "연습문제 2 풀이"

    대조 목적은 마스킹한 문맥에서 올바른 잠재 목표와 다른 후보를 구별한다. 코드의 quantizer는 문맥 출력의 선형 레이어고 이산 codebook이나 목표 선택·로스가 없어 그 학습을 구현하지 않는다. 대조 목적이 모든 복원 목적보다 우월하거나 화자·잡음을 완전히 제거한다고 단정할 수 없다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 head의 어휘·blank와 greedy CTC decode를 설명하고 학습·padding 처리가 포함되어 있는지 검토하라.

</div>

??? success "연습문제 3 풀이"

    기본 어휘는 알파벳26개·공백·apostrophe·UNK·blank의30개이며 blank ID는29다. decode는 연속 동일 ID를 합치고 blank를 제거하되 blank 뒤 반복은 다시 남긴다. vocab_size를 바꾸어도 문자열 목록은 고정되어 있어 따로 맞춰야 한다. CTC 학습·입력 길이별 padding 제거는 없고, 새 head와 무작위 입력의 출력은 실제 전사 성능이 아니다.

    ```python
    class Wav2Vec2ForASR(nn.Module):
        def __init__(self, d_model=768, n_layers=12, n_heads=12, vocab_size=30):
            super().__init__()
            self.wav2vec2 = Wav2Vec2(d_model, n_layers, n_heads)
            self.lm_head = nn.Linear(d_model, vocab_size)
            self.blank_id = vocab_size - 1  # CTC 빈 토막
            # 낱말 곳간: a-z (0-25), 사이 (26), ' (27), <unk> (28), <blank> (29)
            self.vocab = list("abcdefghijklmnopqrstuvwxyz '") + ["<unk>", "<blank>"]
        
        def forward(self, x):
            context, _ = self.wav2vec2(x)
            logits = self.lm_head(context)  # [batch, time, vocab_size]
            return logits
        
        def decode(self, logits):
            """욕심쟁이 CTC 풀기."""
            # logits: [batch, time, vocab_size]
            pred_ids = logits.argmax(dim=-1)  # [batch, time]
            texts = []
            for seq in pred_ids:
                chars = []
                prev = -1
                for idx in seq:
                    idx = idx.item()
                    if idx != self.blank_id and idx != prev:
                        chars.append(self.vocab[idx])
                    prev = idx
                texts.append("".join(chars))
            return texts
    
    # 쓰는 법:
    model = Wav2Vec2ForASR()
    x = torch.randn(1, 1, 16000)
    logits = model(x)
    decoded = model.decode(logits)
    print(f"Decoded text: {decoded}")
    ```


## 정리하며

wav2vec 2.0의 자기지도 목적과 단순한 컨볼루션·트랜스포머 구조는 구분해야 한다. 현재 코드는 양자화·마스킹·대조 학습이 없고 시간 해상도도 원 모델과 다르다. ASR에는 전사 head·CTC·길이·blank와 실제 파인튜닝을 연결해야 한다.
