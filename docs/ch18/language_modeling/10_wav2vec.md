# Wav2Vec 2.0

Wav2Vec 2.0은 2020년 논문 "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations"에서 나온, 말소리 알아듣기를 위한 스스로 살피는 배움 얼거리이다. 이 모델은 맞대어 배우기에 양자화와 변환기 부호기를 곁들여 말소리 나타냄을 배우며, 이름표 붙인 자료가 적어도 좋은 말소리 알아듣기를 해낸다.

## 코드

```python
#!/usr/bin/env python3
'''
Wav2Vec 2.0 — 말소리 알아듣기를 위한 스스로 살피는 배움
Paper: "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations" (2020)
핵심: 말소리를 위한 맞대어 배우기, 양자화, 변환기 부호기
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

## 논의

Wav2Vec 2.0은 크게 두 단계로 돌아간다. 곧 날 소리 물결을 숨은 말소리 나타냄으로 다루는 누비기 특징 부호기와, 그 특징에서 맥락이 담긴 나타냄을 세우는 변환기 부호기이다. 특징 부호기는 묶음 고르게 맞추기와 GELU 깨어남을 곁들인 1차원 누비기를 이어 붙여 16kHz의 날 소리 신호를 다루기 좋은 때 해상도까지 차츰 줄여 뽑는다. 이 꾸밈 덕분에 멜 스펙트로그램 같은 손수 만든 특징 없이 날 소리를 곧바로 다룰 수 있다.

스스로 살피는 미리 익히기 목표는 맞대어 배우기를 쓴다. 곧 가린 때 걸음마다 모델이 헷갈리게 하는 것들 가운데서 맞는 양자화된 말소리 낱을 가려내야 한다. 양자화 단원은 이어진 숨은 나타냄을 유한한 부호책으로 띄엄띄엄하게 만들어 맞대어 배우기의 목표를 만든다. 이러면 이름표 없는 소리 자료를 잔뜩 써서 풍부한 말소리 나타냄을 배운 뒤, 이름표 붙인 적은 자료로 말소리 알아듣기에 곱게 다듬을 수 있다.

변환기 부호기는 숨은 특징의 온 차례를 다루어 모델이 말소리의 멀리 떨어진 얽힘을 담아내게 한다. 여기 보인 얼개는 온전한 Wav2Vec 2.0 모델을 간추린 판이며, 온전한 판은 상대 자리 부호와 여러 부호책 묶음에 걸친 곱 양자화라는 더 정교한 양자화를 쓴다. 실전에서 이 얼개는 말소리 알아듣기 잣대에서, 특히 이름표 붙인 자료가 귀한 곳에서 놀라운 성능을 낸다.

## 연습문제

**연습문제 1.**
`FeatureEncoder`가 스테레오 소리 들임(채널 1개가 아니라 2개)을 받도록 고치고, 고친 모델의 매개변수 전체 개수를 본디 것과 견주어 셈하여라.

??? success "연습문제 1 풀이"
    첫 `nn.Conv1d`의 들임 채널을 1에서 2로 바꾼다:
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
    
    # 첫 누비기 층의 매개변수가 1*512*10 = 5120에서 2*512*10 = 10240으로 바뀐다
    # 첫 층에서만 매개변수가 5120개 늘어난다.
    ```

---

**연습문제 2.**
Wav2Vec 2.0이 말소리의 스스로 살피는 미리 익히기에 (자기부호기 같은) 되살림 바탕 목표 대신 맞대어 배우기를 쓰는 까닭을 밝혀라. 좋은 점과 있을 수 있는 나쁜 점은 무엇인가?

??? success "연습문제 2 풀이"
    말소리에서 맞대어 배우기는 되살림보다 나은 점이 여럿 있다:
    
    1. **결이 고운 소리까지 나타내지 않아도 된다**: 되살림 목표는 말의 내용과 상관없는 물결의 세부(말하는 이, 뒷소리, 녹음 품질)까지 그대로 되살리게 만든다. 맞대어 배우기는 맞는 말소리 낱을 헷갈리게 하는 것들과 가려내기만 하면 되므로 소리값의 내용에 초점을 둔다.
    
    2. **더 효율적인 배움 신호**: 맞대어 배우기는 모든 때 걸음에서 낱낱의 되살림 어긋남을 줄이려 애쓰는 대신 양성 보기와 음성 보기를 맞대어 또렷한 배움 기울기를 준다.
    
    3. **뒤따르는 일로 더 잘 옮겨진다**: 맞대어 배워 얻은 나타냄은 낮은 수준의 소리 세부보다 높은 수준의 뜻과 소리값 앎을 담아내는 편이어서 뒤따르는 말소리 알아듣기 성능이 더 좋다.
    
    **나쁜 점**: 맞대어 배우기는 음성 보기를 조심스레 골라야 하고 그 개수에 민감하다. 또 뜻있는 띄엄띄엄한 목표를 만들려면 양자화 단원이 필요해 꾸밈이 복잡해진다.

---

**연습문제 3.**
맥락이 담긴 나타냄을 글자 30개(글자 26개 + 사이 + 홑따옴표 + 빈칸 + 모름)의 낱말 곳간으로 내리쬐어 말소리를 알아듣는 단순한 곱게 다듬기 머리를 `Wav2Vec2` 모델에 더하고, 욕심쟁이 CTC 풀기로 모델의 내놓음을 글로 바꾸는 `decode` 메서드를 짜라.

??? success "연습문제 3 풀이"
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
