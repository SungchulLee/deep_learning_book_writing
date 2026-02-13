"""
Comprehensive NER Demonstration
================================

This script demonstrates all NER approaches covered in this module:
1. Rule-based NER
2. Dictionary-based NER  
3. CRF-based NER
4. Deep learning approaches (BiLSTM-CRF, Transformers)

Run this script to see complete examples of each approach.

Author: Educational purposes
Date: 2025
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def demo_rule_based():
    """Demonstrate rule-based NER."""
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
    """Demonstrate dictionary-based NER."""
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
    """Demonstrate feature extraction for traditional ML."""
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
    """Demonstrate NER evaluation metrics."""
    print("\n\n" + "="*70)
    print("4. NER EVALUATION METRICS")
    print("="*70)
    
    from intermediate.evaluation_metrics_07_evaluation_metrics import NERMetrics
    
    # Example predictions
    y_true = [["B-PER", "I-PER", "O", "B-ORG", "I-ORG"]]
    y_pred = [["B-PER", "I-PER", "O", "B-ORG", "O"]]  # Missed I-ORG
    
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
    """Demonstrate deep learning architectures."""
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
    """Print summary and next steps."""
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
    """Run all demonstrations."""
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
