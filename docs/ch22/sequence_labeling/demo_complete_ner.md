# NER 종합 데모의 실행 범위와 결과 해석

이 데모는 다른 모듈의 규칙·사전 기반 추출, 수작업 특징과 토큰 지표를 호출하고 신경망 구조의 개념을 출력한다. 모든 NER 방식을 학습·평가하거나 하나의 운영 파이프라인으로 결합하는 코드는 아니다. 방법별 장단점 표도 통제 실험 결과가 아니라 고정된 설명 문구다.

## 1. 코드와 기존 출력

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
    """깊은 배움 구조 보이기."""
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

??? note "기존 전체 출력"

    ```

    ======================================================================
    COMPREHENSIVE NER DEMONSTRATION
    Module 38: Named Entity Recognition
    ======================================================================
    ======================================================================
    1. RULE-BASED NER DEMONSTRATION
    ======================================================================

    Note: Rule-based demo requires beginner modules. Error: No module named 'beginner'


    ======================================================================
    2. DICTIONARY-BASED NER DEMONSTRATION
    ======================================================================

    Note: Dictionary-based demo requires beginner modules. Error: No module named 'beginner'


    ======================================================================
    3. FEATURE EXTRACTION FOR TRADITIONAL ML
    ======================================================================

    Note: Feature extraction demo requires intermediate modules. Error: No module named 'intermediate'


    ======================================================================
    4. NER EVALUATION METRICS
    ======================================================================

    Note: Evaluation demo requires intermediate modules. Error: No module named 'intermediate'


    ======================================================================
    5. DEEP LEARNING ARCHITECTURES
    ======================================================================

    BiLSTM-CRF Architecture:
      Input → Embedding → BiLSTM → Linear → CRF → Output
      - Captures context from both directions
      - CRF layer enforces valid tag sequences
      - State-of-the-art for sequence labeling

    Transformer Architecture (BERT/RoBERTa):
      Input → BERT → Linear → Softmax → Tags
      - Uses pre-trained language understanding
      - Contextual embeddings
      - Current state-of-the-art performance

    ----------------------------------------------------------------------
    Modern NER systems typically use transformer-based models


    ======================================================================
    SUMMARY: NER APPROACH COMPARISON
    ======================================================================

    Approach        Precision    Recall     Speed        Data Needed     Best For
    ----------------------------------------------------------------------------------------------------
    Rule-based      High         Low        Very Fast    None            Domain-specific patterns
    Dictionary      High         Low        Very Fast    None            Known entities
    CRF             Medium       Medium     Fast         Moderate        General NER
    BiLSTM-CRF      High         High       Medium       Large           General NER
    Transformer     Very High    Very High  Slow         Very Large      State-of-the-art

    ======================================================================
    NEXT STEPS
    ======================================================================

    1. For quick prototyping: Start with rule-based or dictionary NER
    2. For better performance: Collect training data and use CRF or BiLSTM-CRF
    3. For state-of-the-art: Fine-tune a transformer model (BERT/RoBERTa)
    4. For production: Combine multiple approaches (ensemble)

    ======================================================================
    LEARNING PATH
    ======================================================================

    Week 1 (Beginner):
      - Understanding NER concepts and entity types
      - IOB tagging schemes
      - Rule-based and dictionary-based approaches

    Week 2 (Intermediate):
      - Feature extraction techniques
      - CRF for sequence labeling
      - Evaluation metrics and dataset creation

    Week 3-4 (Advanced):
      - BiLSTM-CRF architecture
      - Transformer-based NER (BERT, RoBERTa)
      - Fine-tuning and production deployment

    ======================================================================

    ======================================================================
    DEMONSTRATION COMPLETE
    ======================================================================

    Explore individual module files for detailed implementations:
      - beginner/     : Basic NER concepts and simple approaches
      - intermediate/ : Traditional ML and feature engineering
      - advanced/     : Deep learning architectures
      - utils/        : Helper utilities
      - data/         : Sample datasets
    ```

## 2. 가져오는 모듈과 실패 처리

beginner·intermediate 패키지에서 규칙·사전·특징·지표를 가져온다. 코드를 어느 파일 위치와 import 경로에서 사용하는지에 따라 이 패키지를 찾을 수 있어야 한다. sys.path에 상위 경로를 추가하는 것만으로 모든 실행 위치에서 import가 해결되는 것은 아니다. 예제 Markdown의 코드와 실제 별도 Python 모듈의 계약도 확인한다.

각 데모의 예외를 출력하고 다음으로 넘어가므로 마지막 완료 메시지가 앞의 모든 데모가 정상 실행되었다는 증거는 아니다. 어떤 단계가 실패했는지와 실제 결과를 구분해야 한다. 이 문서 편집에서는 모듈을 import하거나 예제를 실행하지 않고 정적으로 제공된 코드를 검토한다.

## 3. 규칙·사전 결과와 의미적 정확도

규칙 기반 추출은 정규식·우선순위의 결과다. 대소문자 무시와 기호의 단어 경계·탐욕적 일치 때문에 일반 구절이 이름으로 잡히거나 금액·접미사가 누락될 수 있다. 같은 조건에서 결정적으로 출력된다고 의미적 정확도가 보장되는 것은 아니다.

사전 데모의 ‘known entities에 완벽한 정밀도’ 문구도 부정확하다. Apple 같은 다의어의 유형을 문맥 없이 정할 수 없고, 공백 토큰화의 구두점·반복 언급의 첫 위치 검색 때문에 알려진 항목도 놓치거나 잘못 정렬할 수 있다. 사전 등재 여부와 실제 문맥의 정답을 분리해 평가해야 한다.

특징 추출은 Steve·Jobs 등의 철자·문맥 사전을 보여 줄 뿐 CRF나 신경망을 학습하지 않는다. 특징 수가 많거나 이름이 익숙하다는 사실을 정확도 비교로 바꾸어 설명하지 않는다.

## 4. 데모의 토큰 지표와 정확 span

평가 예는 정답 B-PER I-PER O B-ORG I-ORG에 대해 마지막 I-ORG를 O로 바꾼 예측을 사용한다. 호출하는 compute_metrics는 비O **토큰 태그**를 세므로TP3·FP0·FN1, P1·R0.75·F1약0.857이다. 이는 개체 수준의 정확 일치 F1이 아니다.

정확 span으로 보면 사람은 맞지만 조직은 짧게 예측되어TP1·FP1·FN1, P·R·F1은각각0.5다. 태그의 일부를 맞힌 것과 전체 언급의 경계·유형을 맞힌 것을 구분한다. 입력 목록 길이·문장 경계·패딩·무효 태그의 정책도 실제 평가에 명시해야 한다.

## 5. 구조 설명·비교표의 한계

demo_architectures는 BiLSTM-CRF·트랜스포머 구조를 **출력할 뿐 생성·학습·비교하지 않는다**. CRF가 항상 올바른 태그를 강제하는 것도 아니다. 명시적인 시작·종료·금지 전이 제약과 같은 점수의 정규화·디코딩이 필요하다. 관련 코드의 전이 방향·패딩·head 학습 한계는 개별 절의 설명을 참고한다.

정밀도·재현율·속도·필요 자료를 High·Low 등으로 적은 표는 데이터·분할·구조·사전 학습·장치 조건이 없는 질적 개요다. 실제 측정치나 모든 과제의 고정 순위로 인용하지 않는다. 앙상블 역시 운영에 항상 필요한 단계이거나 자동으로 더 정확한 해결책은 아니다.

실제 모델 선택에는 같은 주석 체계·관찰 자료·독립 분할·span 지표와 전체 처리 비용을 사용한다. 이 데모는 방법을 연결해 읽는 안내이며 최신 성능이나 배포 준비 상태를 검증한 실험 보고가 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
이 파일이 실제로 실행하는 학습 모델과 출력만 하는 구조를 구분하라.

</div>

??? success "연습문제 1 풀이"

    규칙·사전·특징 추출·토큰 지표를 다른 모듈에서 호출한다. BiLSTM-CRF·트랜스포머는 설명 문자열만 출력하며 학습하지 않는다. 완료 메시지도 예외 처리된 데모의 성공을 보장하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
외부 모듈·입력·결과에서 확인할 계약을 설명하라.

</div>

??? success "연습문제 2 풀이"

    패키지 import 경로와 실제 모듈, 원문 offset·유형·토큰 순서와 태그 길이를 맞춘다. 추출 개수와 정답 평가, 토큰·span 지표를 구분한다. 고정 영상 텐서 shape 검사로 확인할 수 있는 인터페이스가 아니다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
데모의 토큰 F1과 정확 span F1을 계산하라.

</div>

??? success "연습문제 3 풀이"

    비O 토큰은TP3·FP0·FN1이라P1·R0.75·F1=6/7≈0.857이다. 정확 개체는 사람 하나만 맞고 조직의 짧은 예측과 누락 정답이 있어TP1·FP1·FN1, F1=0.5다. 부분 토큰 정답은 정확 span의 부분 점수가 아니다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 기존 시험 조각이 이 종합 호출 예제에 맞는가?

</div>

??? success "연습문제 4 풀이"

    공백이 있는 식별자와 정의되지 않은 Demo Complete NER·입력 때문에 실행 가능한 시험이 아니다. 이 파일 자체는 신경망이 아니므로 backward·parameters 검사가 맞지 않는다. import 실패·단계별 결과·원문 offset·태그 길이·정확한 지표와 예외 후 완료 메시지를 명세로 확인해야 한다.

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


## 정리하며

종합 데모는 여러 NER 도구와 구조 설명을 연결한다. 실제 호출·설명 출력·고정 비교표를 구분하고 규칙·사전 일치와 의미적 정답, 토큰 F1과 정확 span F1을 혼동하지 않는다.
