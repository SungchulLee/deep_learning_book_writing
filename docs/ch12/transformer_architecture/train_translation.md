# 번역 학습

앞의 GRU Seq2Seq 모델, 교차 엔트로피 손실과 Adam 옵티마이저를 준비하는 짧은 예제다. 함수 이름과 달리 데이터 로딩, 손실 계산과 가중치 갱신을 수행하지 않는다.

## 1. 코드

```python
"""
주의를 쓰는 Seq2Seq 학습
"""
import torch
import torch.nn as nn
from seq2seq_with_attention import Seq2SeqWithAttention

# ========================================================================
# 메인
# ========================================================================

# 간단한 학습 보기
def train_simple_example():
    # 초매개변수
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    embed_size = 256
    hidden_size = 512
    batch_size = 32
    
    model = Seq2SeqWithAttention(src_vocab_size, tgt_vocab_size, embed_size, hidden_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print("Training script ready!")
    print("\nTo train: Create your own dataset and training loop")

if __name__ == '__main__':
    train_simple_example()
```

**출력:**

```
Model created with 4702697 parameters
Training script ready!

To train: Create your own dataset and training loop
```

## 2. 논의

src_vocab_size와 tgt_vocab_size는 각각 1000, 임베딩은 256, 은닉은 512다. 모델의 4,702,697개 매개변수는 임베딩·GRU·어텐션·출력층을 모두 합한 수이며, 보존된 출력도 준비 완료 메시지다. batch_size=32는 선언만 되어 사용되지 않는다.

실제 번역 학습에는 원문·목표문 쌍, 어휘·특수 토큰과 배치 패딩이 필요하다. 목표 입력은 BOS부터 시작하고 정답은 다음 토큰부터 EOS까지 맞춘다. 반환 로짓 $(B,T,V)$와 정수 정답 $(B,T)$를 손실 함수가 기대하는 축으로 배치해야 한다.

현재 CrossEntropyLoss의 기본값은 PAD 토큰을 별도로 제외하지 않는다. 패딩 마스크와 ignore_index를 맞추고, 역전파·옵티마이저 갱신·검증과 생성 평가를 추가해야 한다. 이 페이지에 실제 학습 손실, 번역 정확도나 BLEU 결과는 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
제시된 모델의 매개변수 4,702,697개를 구성 요소별로 나누어라.

</div>

??? success "연습문제 1 풀이"
    인코더 임베딩 256,000개와 GRU 1,182,720개, 디코더 임베딩 256,000개와 GRU 1,969,152개, 어텐션 525,825개, 출력층 513,000개다. 합계는 4,702,697개이며 손실 함수와 Adam 생성은 모델의 학습 매개변수를 늘리지 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
준비 코드만으로 학습이 이루어지지 않는 이유와 필요한 단계를 설명하라.

</div>

??? success "연습문제 2 풀이"
    모델 호출·손실 계산·zero_grad·backward·step이 모두 없다. 데이터를 구성하고 목표 시프트와 패딩 처리를 정한 뒤 그 단계들을 반복해야 한다. 평가 데이터는 학습·설정 선택에 쓴 분할과 역할을 구분한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 Seq2Seq의 손실에서 소프트맥스와 패딩을 어떻게 처리해야 하는가?

</div>

??? success "연습문제 3 풀이"
    CrossEntropyLoss는 로짓을 받으므로 모델 출력에 소프트맥스를 먼저 적용하지 않는다. 어휘 축을 맞추고 PAD 정답을 ignore_index로 제외한다. 손실 평균을 보고할 때는 유효 목표 토큰 수를 기준으로 집계한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래의 테스트 코드 조각을 검토하라. 정의되지 않은 클래스와 잘못된 이름, 입력 모양 가정을 찾아 이 페이지의 실제 함수에 맞는 점검 항목을 제안하라.

</div>

??? success "연습문제 4 풀이"
    아래 조각은 실제 실행 가능한 테스트가 아니다. 함수 이름과 모델 이름부터 이 예제에 맞게 작성해야 한다.
    ```python
    def test_train translation():
        model = Train Translation(...)
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
    이 페이지는 완성된 사용자 정의 모델 클래스를 제공하지 않는다. 실제 함수의 입력·출력 모양, 허용 자료형, 유한한 값과 경사의 유무를 기준으로 검사한다. 빈 입력을 처리하는 방식은 연산마다 다르며, 무조건 정상적인 손실이나 경사가 나와야 한다고 가정하지 않는다.

## 정리하며

이 페이지는 번역 학습의 구성 요소를 생성하는 준비 단계다. 출력의 매개변수 수를 학습 결과로 오해하지 말고, 목표 정렬·패딩·실제 갱신과 평가 과정을 별도로 구성해야 한다.
