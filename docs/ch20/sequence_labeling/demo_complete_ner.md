# 온전한 이름 알아보기 보임

두루 살피는 이름 알아보기 보임. 이 각본은 이 단원에서 다룬 모든 이름 알아보기 방식을 보여 준다:

자연어 다루기는 깊은 배움 방식으로 크게 달라졌다. 이 단원은 차례 이름표 붙이기 재주를 보여 주며, 신경망이 글을 어떻게 다루고 만들어 내는지 밝히는 실전 짜기를 준다.

## 1. 코드

```python
"""
두루 살피는 이름 알아보기 보임
================================

이 각본은 이 단원에서 다룬 모든 이름 알아보기 방식을 보여 준다:
1. 규칙 바탕 이름 알아보기
2. 사전 바탕 이름 알아보기
3. CRF 바탕 이름 알아보기
4. 깊은 배움 방식(두 방향 LSTM-CRF, 변환기)

방식마다의 온전한 보기를 보려면 이 각본을 돌려라.

지은이: 배움 목적
날짜: 2025
"""

import sys
import os

# ========================================================================
# 메인
# ========================================================================

# 들여오기를 위해 부모 디렉터리를 경로에 더하기
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def demo_rule_based():
    """규칙 바탕 이름 알아보기 보이기."""
    print("="*70)
    print("1. RULE-BASED NER DEMONSTRATION")
    print("="*70)
    
    from beginner.ner_basics_03_rule_based_ner import RuleBasedNER
    
    ner = RuleBasedNER()
    
    texts = [
        "Apple Inc. was founded by Steve Jobs in Cupertino, California.",
        "Microsoft CEO Satya Nadella announced new products in Seattle.",
        "The meeting is on January 15, 2025 at 2:30 PM.",
    ]
    
    for i, text in enumerate(texts, 1):
        print(f"\nExample {i}: {text}")
        entities = ner.extract_entities(text)
        print(f"Found {len(entities)} entities:")
        for entity in entities:
            print(f"  - {entity['text']} ({entity['type']})")
    
    print("\n" + "-"*70)
    print("Advantages: Fast, interpretable, no training needed")
    print("Disadvantages: Low recall, cannot handle variations")


def demo_dictionary_based():
    """사전 바탕 이름 알아보기 보이기."""
    print("\n\n" + "="*70)
    print("2. DICTIONARY-BASED NER DEMONSTRATION")
    print("="*70)
    
    from beginner.dictionary_04_dictionary_ner import DictionaryNER
    
    ner = DictionaryNER()
    
    text = "Steve Jobs founded Apple in California. Bill Gates started Microsoft."
    
    print(f"\nText: {text}")
    entities = ner.extract_entities(text)
    
    print(f"\nFound {len(entities)} entities:")
    for entity in entities:
        print(f"  - {entity['text']} ({entity['type']})")
    
    print("\n" + "-"*70)
    print("Advantages: Very fast lookup, perfect precision for known entities")
    print("Disadvantages: Zero recall for entities not in dictionary")


def demo_feature_extraction():
    """예로부터의 기계 배움을 위한 특징 뽑기 보이기."""
    print("\n\n" + "="*70)
    print("3. FEATURE EXTRACTION FOR TRADITIONAL ML")
    print("="*70)
    
    from intermediate.feature_extraction_05_feature_extraction import FeatureExtractor
    
    extractor = FeatureExtractor()
    tokens = ["Steve", "Jobs", "founded", "Apple", "Inc", "."]
    
    print(f"\nSentence: {' '.join(tokens)}")
    print("\nFeatures for each token:")
    
    for i, token in enumerate(tokens):
        features = extractor.extract_token_features(tokens, i, window_size=1)
        print(f"\n{token}:")
        print(f"  Word shape: {features['word_shape']}")
        print(f"  Is capitalized: {features['is_capitalized']}")
        print(f"  Prefix 2: {features.get('prefix_2', 'N/A')}")
        print(f"  Total features: {len(features)}")
    
    print("\n" + "-"*70)
    print("These features are used by CRF and other traditional ML models")


def demo_evaluation():
    """이름 알아보기 값매김 잣대 보이기."""
    print("\n\n" + "="*70)
    print("4. NER EVALUATION METRICS")
    print("="*70)
    
    from intermediate.evaluation_metrics_07_evaluation_metrics import NERMetrics
    
    # 어림 보기
    y_true = [["B-PER", "I-PER", "O", "B-ORG", "I-ORG"]]
    y_pred = [["B-PER", "I-PER", "O", "B-ORG", "O"]]  # I-ORG를 놓침
    
    print("\nTrue labels: ", y_true[0])
    print("Predicted:   ", y_pred[0])
    
    metrics = NERMetrics.compute_metrics(y_true, y_pred)
    
    print(f"\nToken-level metrics:")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1 Score:  {metrics['f1']:.3f}")
    
    print("\n" + "-"*70)
    print("Proper evaluation is crucial for comparing NER systems")


def demo_architectures():
    """깊은 배움 얼개 보이기."""
    print("\n\n" + "="*70)
    print("5. DEEP LEARNING ARCHITECTURES")
    print("="*70)
    
    print("\nBiLSTM-CRF Architecture:")
    print("  Input → Embedding → BiLSTM → Linear → CRF → Output")
    print("  - Captures context from both directions")
    print("  - CRF layer enforces valid tag sequences")
    print("  - State-of-the-art for sequence labeling")
    
    print("\nTransformer Architecture (BERT/RoBERTa):")
    print("  Input → BERT → Linear → Softmax → Tags")
    print("  - Uses pre-trained language understanding")
    print("  - Contextual embeddings")
    print("  - Current state-of-the-art performance")
    
    print("\n" + "-"*70)
    print("Modern NER systems typically use transformer-based models")


def print_summary():
    """간추림과 다음 걸음 찍기."""
    print("\n\n" + "="*70)
    print("SUMMARY: NER APPROACH COMPARISON")
    print("="*70)
    
    approaches = [
        ("Rule-based", "High", "Low", "Very Fast", "None", "Domain-specific patterns"),
        ("Dictionary", "High", "Low", "Very Fast", "None", "Known entities"),
        ("CRF", "Medium", "Medium", "Fast", "Moderate", "General NER"),
        ("BiLSTM-CRF", "High", "High", "Medium", "Large", "General NER"),
        ("Transformer", "Very High", "Very High", "Slow", "Very Large", "State-of-the-art"),
    ]
    
    print(f"\n{'Approach':<15} {'Precision':<12} {'Recall':<10} {'Speed':<12} {'Data Needed':<15} {'Best For'}")
    print("-" * 100)
    
    for approach, precision, recall, speed, data, best_for in approaches:
        print(f"{approach:<15} {precision:<12} {recall:<10} {speed:<12} {data:<15} {best_for}")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. For quick prototyping: Start with rule-based or dictionary NER")
    print("2. For better performance: Collect training data and use CRF or BiLSTM-CRF")
    print("3. For state-of-the-art: Fine-tune a transformer model (BERT/RoBERTa)")
    print("4. For production: Combine multiple approaches (ensemble)")
    
    print("\n" + "="*70)
    print("LEARNING PATH")
    print("="*70)
    print("\nWeek 1 (Beginner):")
    print("  - Understanding NER concepts and entity types")
    print("  - IOB tagging schemes")
    print("  - Rule-based and dictionary-based approaches")
    
    print("\nWeek 2 (Intermediate):")
    print("  - Feature extraction techniques")
    print("  - CRF for sequence labeling")
    print("  - Evaluation metrics and dataset creation")
    
    print("\nWeek 3-4 (Advanced):")
    print("  - BiLSTM-CRF architecture")
    print("  - Transformer-based NER (BERT, RoBERTa)")
    print("  - Fine-tuning and production deployment")
    
    print("\n" + "="*70)


def main():
    """모든 보임 돌리기."""
    print("\n" + "="*70)
    print("COMPREHENSIVE NER DEMONSTRATION")
    print("Module 38: Named Entity Recognition")
    print("="*70)
    
    try:
        demo_rule_based()
    except Exception as e:
        print(f"\nNote: Rule-based demo requires beginner modules. Error: {e}")
    
    try:
        demo_dictionary_based()
    except Exception as e:
        print(f"\nNote: Dictionary-based demo requires beginner modules. Error: {e}")
    
    try:
        demo_feature_extraction()
    except Exception as e:
        print(f"\nNote: Feature extraction demo requires intermediate modules. Error: {e}")
    
    try:
        demo_evaluation()
    except Exception as e:
        print(f"\nNote: Evaluation demo requires intermediate modules. Error: {e}")
    
    demo_architectures()
    print_summary()
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nExplore individual module files for detailed implementations:")
    print("  - beginner/     : Basic NER concepts and simple approaches")
    print("  - intermediate/ : Traditional ML and feature engineering")
    print("  - advanced/     : Deep learning architectures")
    print("  - utils/        : Helper utilities")
    print("  - data/         : Sample datasets")


if __name__ == "__main__":
    main()
```

## 2. 논의

여기 짠 것은 깔끔하고 읽기 좋은 PyTorch 코드로 차례 이름표 붙이기의 핵심 개념을 보여 준다. 단원별로 나뉜 짜임 덕분에 낱낱의 조각을 익히고 다른 일이나 자료 뭉치에 맞게 고치기 쉽다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 넓혀진다. 초매개변수와 구조의 변형, 다른 데이터셋으로 실험해 보면 이해가 깊어지고 자연어 처리 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
코드를 끝까지 읽고 핵심 꾸밈 결정을 가려내어라. 구체적인 짜기 고름 세 가지를 들고 저마다 왜 차례 이름표 붙이기에 알맞은지 설명하여라.

??? success "연습문제 1 풀이"
    설계 결정은 구현마다 다르지만 흔히 다음이 포함된다. (1) 활성화 함수의 선택 — ReLU 계열은 포화되지 않는 경사를 주어 학습을 빠르게 한다. (2) 정규화 전략 — 배치 정규화는 내부 공변량 이동을 줄여 학습을 안정시킨다. (3) 잔차 연결 — 있을 경우 건너뛰는 경로를 제공하여 깊은 신경망에서도 경사가 흐르게 한다. 각 선택은 표현력, 계산 비용, 학습 안정성 사이의 절충을 반영한다.

---

**연습문제 2.**
입력이 기대하는 모양과 자료형을 갖는지 확인하도록 주 함수나 클래스에 입력 검증을 추가하라. 잘못된 입력에는 유익한 오류 메시지를 내라.

??? success "연습문제 2 풀이"
    `forward` 메서드(또는 해당 함수)의 첫머리에 다음과 같은 검사를 추가한다. `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`와 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'`. 모양을 검증할 때는 중요한 차원을 확인한다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 유익한 오류 메시지는 디버깅 속도를 크게 높이고 코드를 재사용하기에도 더 견고하게 만든다.

---

**연습문제 3.**
은닉 크기가 $h$이고 입력 크기가 $x$로 같을 때 LSTM 셀과 GRU 셀의 매개변수 개수를 비교하라. 어느 쪽이 더 적으며 그 이유는 무엇인가?

??? success "연습문제 3 풀이"
    LSTM에는 4개의 게이트(입력, 망각, 셀, 출력)가 있고 각 게이트가 입력과 은닉 상태 양쪽에 대한 가중치 행렬을 가지므로 $4 \times (x \cdot h + h \cdot h + h) = 4(xh + h^2 + h)$개의 매개변수를 갖는다. GRU에는 3개의 게이트(재설정, 갱신, 새 상태)가 있어 $3 \times (x \cdot h + h \cdot h + h) = 3(xh + h^2 + h)$개이다. GRU는 게이트를 4개 대신 3개 쓰고 셀 상태와 은닉 상태를 합치므로 LSTM의 75%에 해당하는 매개변수를 갖는다. 실무에서 GRU는 매개변수가 적은데도 LSTM에 견줄 만한 성능을 내는 경우가 많다.

---

**연습문제 4.**
온전한 이름 알아보기 보임의 짜기를 확인하는 두루 살피는 시험 함수를 쓰라. 빈 들임, 원소가 하나인 들임, 아주 큰 들임, 값이 극단인 들임(0, 아주 큰 수)을 비롯한 모서리 경우를 시험하여라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_demo complete ner():
        model = Demo Complete NER(...)
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

## 정리하며

**다룬 것** — 온전한 이름 알아보기 보임

여기 짠 것은 깔끔하고 읽기 좋은 PyTorch 코드로 차례 이름표 붙이기의 핵심 개념을 보여 준다.

앞의 연습문제 4개로 스스로 따져 볼 수 있다.
