# WAV2VEC

WAV2VEC은 2020년 글 "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations"에서 나왔다. 말소리를 위한 맞대어 배우기, 수 줄이기, 변환기 부호기를 쓴다.

여기 짜보기는 WAV2VEC을 짧고 배우기 좋게 보인 본이다. 코드는 고갱이 얼개와 앞으로 걸음에 마음을 두어, 고갱이 꾸밈새를 살펴보고 이리저리 바꾸어 보기 쉽다.

## 코드

```python
#!/usr/bin/env python3
'''
Wav2Vec 2.0 - Self-Supervised Learning for Speech Recognition
Paper: "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations" (2020)
Key: Contrastive learning for speech, quantization, transformer encoder
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
    print(f"Input: {x.shape}")```

## 논의

이 짜보기는 갈래 2개(`FeatureEncoder`, `Wav2Vec2`)를 매기고, 이들이 어울려 온전한 변환기 얼개를 이룬다. 갈래마다 남다른 몫을 담아 코드를 묶음으로 나누고 넓히기 쉽게 한다. `forward` 방법이 PyTorch이 절로 미분하는 데 쓰는 셈 그림을 매긴다.

여기 실린 코드는 본보기 짜보기라 다듬기보다 알아보기 쉬움을 앞세운다. 서비스 얼개라면 흔히 섞인 촘촘함 익히기, 흩은 자료 나란히, 더 정교한 자료 불리기를 더한다. 그래도 여기서 보인 얼개의 고갱이 깨침은 크기와 상관없이 그대로다.

## 익힘 문제

**익힘 1.**
`FeatureEncoder`의 앞으로 걸음을 따라 텐서의 꼴을 좇아라. 기본 매개변수로 들임 보기 4개를 묶어 넣었을 때, 큰 셈(엮음, 모으기, 선형 켜)마다 꼴이 어떻게 되는지 적어라.

??? success "익힘 1 풀이"
    들임의 꼴에서 비롯해 켜를 차례로 건다. `Conv2d(in_c, out_c, k)`마다 자리 차수는 $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌고(덧대기 없이) `padding=k//2`이면 그대로다. 알갱이 2로 모으면 자리 차수가 반이 된다. 선형 켜는 마지막 차수를 바꾼다. 묶음 차수는 내내 그대로임을 좇아라. 엮음 켜에서는 $(B, C, H, W)$, 편 뒤에는 $(B, F)$으로 가운데 꼴을 적어라.

---

**익힘 2.**
얼개를 크기 $64 \times 64$의 RGB 그림(들임 꼴: $3 \times 64 \times 64$)을 받도록 고쳐라. 켜의 차수를 모두 그에 맞게 손보고 모형이 어긋남 없이 도는지 따져라.

??? success "익힘 2 풀이"
    첫 엮음 켜의 `in_channels`을 이제 값에서 3으로 바꾼다. 엮음과 모으기 켜마다 $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$으로 자리 차수를 다시 셈한다. 첫 선형 켜의 `in_features`을 마지막 엮음/모으기 켜의 편 날임에 맞춘다. `model = FeatureEncoder(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`으로 따진다.

---

**익힘 3.**
들임과 날임의 차수가 같을 때 여느 엮음과 깊이별로 가른 엮음의 매개변수 수와 뜨는 셈 횟수를 견주어라. 셈을 가장 크게 아끼는 때는 언제인가?

??? success "익힘 3 풀이"
    여느 `Conv2d(C_in, C_out, k)`의 매개변수는 $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$개다. 깊이별로 가른 엮음은 이를 (1) 깊이별: $C_{{\text{{in}}}} \times k^2$개(들임 갈래마다 거르개 하나)와 (2) 점별: $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$개(1x1 엮음)로 나눈다. 매개변수의 견줌은 어림잡아 $1/C_{{\text{{out}}}} + 1/k^2$이다. $k=3$이고 $C_{{\text{{out}}}}=256$이면 매개변수가 약 $8{-}9\times$ 적다. $C_{{\text{{out}}}}$과 $k$이 모두 클 때 가장 크게 아낀다.

---

**익힘 4.**
`FeatureEncoder`을 켜나 덩이의 수를 골라 잡을 수 있게 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`으로 깊이를 바꿀 수 있는 얼개를 짜라. 켜 2개, 4개, 8개로 시험하여라.

??? success "익힘 4 풀이"
    박아 넣은 켜를 다음으로 갈음한다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 방법에서 `for layer in self.layers: x = layer(x)`으로 되돈다. 여느 파이썬 목록이 아니라 `nn.ModuleList`을 써야 PyTorch이 매개변수를 모두 다듬기에 올린다. `for n in [2, 4, 8]: model = FeatureEncoder(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`으로 시험한다.
