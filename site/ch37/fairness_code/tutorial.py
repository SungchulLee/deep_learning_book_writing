"""
Tutorial: Understanding Bias and Fairness in Deep Learning
A step-by-step guide with examples.
"""

import numpy as np
from typing import Dict


# ============================================================================
# PART 1: Understanding Bias in Data
# ============================================================================

def part1_data_bias():
    """
    Demonstrate how bias can exist in training data.
    """
    print("=" * 80)
    print("PART 1: Understanding Bias in Data")
    print("=" * 80)
    
    # Scenario: Credit approval dataset
    print("\nScenario: Credit Card Approval")
    print("-" * 80)
    
    # Simulate data for two groups
    np.random.seed(42)
    n_per_group = 500
    
    # Group A (e.g., one demographic) - higher approval rate
    group_a_scores = np.random.normal(650, 50, n_per_group)  # Credit scores
    group_a_approved = (group_a_scores > 600).astype(int)
    
    # Group B (e.g., another demographic) - lower approval rate due to bias
    group_b_scores = np.random.normal(640, 50, n_per_group)  # Similar scores
    group_b_approved = (group_b_scores > 650).astype(int)  # Higher threshold!
    
    print(f"Group A: Mean credit score = {group_a_scores.mean():.1f}, "
          f"Approval rate = {group_a_approved.mean():.2%}")
    print(f"Group B: Mean credit score = {group_b_scores.mean():.1f}, "
          f"Approval rate = {group_b_approved.mean():.2%}")
    
    print("\n⚠️  Despite similar credit scores, Group B has lower approval rate!")
    print("This is an example of HISTORICAL BIAS in the data.")


# ============================================================================
# PART 2: Types of Fairness
# ============================================================================

def part2_fairness_types():
    """
    Explain different types of fairness definitions.
    """
    print("\n\n" + "=" * 80)
    print("PART 2: Types of Fairness Definitions")
    print("=" * 80)
    
    print("\n1. DEMOGRAPHIC PARITY (Statistical Parity)")
    print("-" * 80)
    print("Definition: P(Ŷ=1|A=0) = P(Ŷ=1|A=1)")
    print("In plain English: Positive predictions should be equal across groups")
    print("\nExample: If 30% of loans are approved for Group A,")
    print("         then 30% should be approved for Group B")
    print("\nWhen to use: When you want equal outcomes regardless of ground truth")
    
    print("\n\n2. EQUAL OPPORTUNITY")
    print("-" * 80)
    print("Definition: P(Ŷ=1|Y=1, A=0) = P(Ŷ=1|Y=1, A=1)")
    print("In plain English: True positive rates should be equal across groups")
    print("\nExample: Among qualified applicants, approval rates should be equal")
    print("\nWhen to use: When false negatives are costly and should be equal")
    
    print("\n\n3. EQUALIZED ODDS")
    print("-" * 80)
    print("Definition: Both TPR and FPR should be equal across groups")
    print("In plain English: Both correct and incorrect predictions equal across groups")
    print("\nExample: Both approval of qualified AND rejection of unqualified")
    print("         should be equal across groups")
    print("\nWhen to use: When both false positives and false negatives matter")
    
    print("\n\n4. PREDICTIVE PARITY")
    print("-" * 80)
    print("Definition: P(Y=1|Ŷ=1, A=0) = P(Y=1|Ŷ=1, A=1)")
    print("In plain English: Precision should be equal across groups")
    print("\nExample: Among approved loans, default rates should be equal")
    print("\nWhen to use: When you care about the quality of positive predictions")


# ============================================================================
# PART 3: Measuring Bias - Hands-on Example
# ============================================================================

def part3_measuring_bias():
    """
    Hands-on example of measuring bias.
    """
    print("\n\n" + "=" * 80)
    print("PART 3: Measuring Bias - Hands-on Example")
    print("=" * 80)
    
    np.random.seed(42)
    
    # Create synthetic predictions
    n = 1000
    sensitive_attr = np.random.randint(0, 2, n)  # 0 or 1
    
    # Create biased predictions
    # Group 0: 70% positive rate
    # Group 1: 40% positive rate
    y_pred = np.where(
        sensitive_attr == 0,
        np.random.choice([0, 1], n, p=[0.3, 0.7]),
        np.random.choice([0, 1], n, p=[0.6, 0.4])
    )
    
    # Calculate metrics
    group_0_pos_rate = np.mean(y_pred[sensitive_attr == 0])
    group_1_pos_rate = np.mean(y_pred[sensitive_attr == 1])
    
    print("\nPositive Prediction Rates:")
    print(f"  Group 0: {group_0_pos_rate:.2%}")
    print(f"  Group 1: {group_1_pos_rate:.2%}")
    
    spd = abs(group_0_pos_rate - group_1_pos_rate)
    di_ratio = min(group_0_pos_rate, group_1_pos_rate) / max(group_0_pos_rate, group_1_pos_rate)
    
    print(f"\nStatistical Parity Difference: {spd:.4f}")
    print(f"Disparate Impact Ratio: {di_ratio:.4f}")
    
    print("\n📊 Interpretation:")
    if spd < 0.1:
        print("  ✓ Statistical Parity: LOW BIAS")
    elif spd < 0.2:
        print("  ⚠ Statistical Parity: MODERATE BIAS")
    else:
        print("  ✗ Statistical Parity: HIGH BIAS")
    
    if di_ratio >= 0.8:
        print("  ✓ Disparate Impact: ACCEPTABLE (>= 0.8, passes 80% rule)")
    else:
        print("  ✗ Disparate Impact: PROBLEMATIC (< 0.8, fails 80% rule)")


# ============================================================================
# PART 4: Bias Mitigation Strategies
# ============================================================================

def part4_mitigation_strategies():
    """
    Explain bias mitigation strategies.
    """
    print("\n\n" + "=" * 80)
    print("PART 4: Bias Mitigation Strategies")
    print("=" * 80)
    
    print("\n1. PRE-PROCESSING (Before Training)")
    print("-" * 80)
    print("Modify the training data before model training")
    
    print("\nA. Reweighing:")
    print("   - Assign different weights to training samples")
    print("   - Overweight under-represented combinations")
    print("   - Example: If (Group=A, Label=1) is rare, give it higher weight")
    
    print("\nB. Resampling:")
    print("   - Oversample minority groups or undersample majority")
    print("   - Balance representation in training data")
    
    print("\nC. Data Augmentation:")
    print("   - Generate synthetic samples for under-represented groups")
    print("   - Use techniques like SMOTE")
    
    print("\n\n2. IN-PROCESSING (During Training)")
    print("-" * 80)
    print("Modify the learning algorithm")
    
    print("\nA. Adversarial Debiasing:")
    print("   - Add an adversary that tries to predict sensitive attribute")
    print("   - Train model to fool the adversary")
    print("   - Result: Representations that don't encode sensitive info")
    
    print("\nB. Fairness Constraints:")
    print("   - Add fairness constraints to the loss function")
    print("   - Example: Penalize difference in TPR across groups")
    
    print("\nC. Fair Representation Learning:")
    print("   - Learn features that are predictive but fair")
    print("   - Remove sensitive information from representations")
    
    print("\n\n3. POST-PROCESSING (After Training)")
    print("-" * 80)
    print("Adjust model outputs")
    
    print("\nA. Threshold Optimization:")
    print("   - Use different classification thresholds for different groups")
    print("   - Example: Lower threshold for disadvantaged group")
    
    print("\nB. Calibration:")
    print("   - Adjust predicted probabilities to ensure fairness")
    print("   - Maintain accuracy while improving fairness")


# ============================================================================
# PART 5: Trade-offs and Considerations
# ============================================================================

def part5_tradeoffs():
    """
    Discuss trade-offs and important considerations.
    """
    print("\n\n" + "=" * 80)
    print("PART 5: Trade-offs and Considerations")
    print("=" * 80)
    
    print("\n⚖️  ACCURACY vs FAIRNESS")
    print("-" * 80)
    print("• Improving fairness often reduces overall accuracy")
    print("• Must decide: How much accuracy to trade for fairness?")
    print("• Consider: Is a small accuracy drop worth significant fairness gain?")
    
    print("\n\n🔄 FAIRNESS DEFINITIONS CAN CONFLICT")
    print("-" * 80)
    print("• Impossibility theorems: Can't satisfy all fairness definitions")
    print("• Example: Demographic parity vs equal opportunity")
    print("• Must choose which definition matters most for your application")
    
    print("\n\n👥 CONTEXT MATTERS")
    print("-" * 80)
    print("• Different applications need different fairness definitions")
    print("• Hiring: Maybe equal opportunity (equal chance for qualified)")
    print("• Lending: Maybe predictive parity (equal default rates)")
    print("• Criminal justice: Extremely sensitive, multiple considerations")
    
    print("\n\n📋 BEST PRACTICES")
    print("-" * 80)
    print("1. Involve stakeholders and affected communities")
    print("2. Use multiple fairness metrics, not just one")
    print("3. Document decisions and trade-offs")
    print("4. Monitor models continuously in production")
    print("5. Be transparent about limitations")
    print("6. Consider legal and ethical implications")
    
    print("\n\n⚠️  IMPORTANT CAVEATS")
    print("-" * 80)
    print("• Technical solutions alone are insufficient")
    print("• Must address root causes of bias")
    print("• Fairness is socially and culturally dependent")
    print("• No silver bullet - requires ongoing work")


# ============================================================================
# PART 6: Practical Workflow
# ============================================================================

def part6_workflow():
    """
    Provide a practical workflow for addressing bias.
    """
    print("\n\n" + "=" * 80)
    print("PART 6: Practical Workflow for Fairness")
    print("=" * 80)
    
    print("\nSTEP 1: IDENTIFY PROTECTED ATTRIBUTES")
    print("-" * 80)
    print("• What attributes should predictions NOT depend on?")
    print("• Examples: gender, race, age, religion, disability")
    print("• Consider intersectionality (multiple attributes)")
    
    print("\n\nSTEP 2: CHOOSE FAIRNESS DEFINITION(S)")
    print("-" * 80)
    print("• What does fairness mean in your context?")
    print("• Consult with stakeholders")
    print("• Document your choice and reasoning")
    
    print("\n\nSTEP 3: ANALYZE TRAINING DATA")
    print("-" * 80)
    print("• Check label distributions by group")
    print("• Look for representation issues")
    print("• Identify potential sources of bias")
    
    print("\n\nSTEP 4: TRAIN BASELINE MODEL")
    print("-" * 80)
    print("• Train without fairness constraints first")
    print("• Measure accuracy and fairness metrics")
    print("• Establish baseline for comparison")
    
    print("\n\nSTEP 5: APPLY BIAS MITIGATION")
    print("-" * 80)
    print("• Choose appropriate technique(s)")
    print("• Pre-processing, in-processing, or post-processing")
    print("• May need to try multiple approaches")
    
    print("\n\nSTEP 6: EVALUATE THOROUGHLY")
    print("-" * 80)
    print("• Test on held-out data")
    print("• Check multiple fairness metrics")
    print("• Analyze across all protected groups")
    print("• Look for unintended consequences")
    
    print("\n\nSTEP 7: MONITOR IN PRODUCTION")
    print("-" * 80)
    print("• Fairness metrics can drift over time")
    print("• Collect feedback from affected users")
    print("• Regularly re-evaluate and update")
    
    print("\n\nSTEP 8: DOCUMENT AND COMMUNICATE")
    print("-" * 80)
    print("• Create model cards documenting fairness")
    print("• Be transparent about limitations")
    print("• Communicate trade-offs clearly")


# ============================================================================
# Main Tutorial Execution
# ============================================================================

def run_tutorial():
    """Run the complete tutorial."""
    print("\n\n")
    print("*" * 80)
    print("*" + " " * 78 + "*")
    print("*" + "  BIAS AND FAIRNESS IN DEEP LEARNING - INTERACTIVE TUTORIAL  ".center(78) + "*")
    print("*" + " " * 78 + "*")
    print("*" * 80)
    
    part1_data_bias()
    part2_fairness_types()
    part3_measuring_bias()
    part4_mitigation_strategies()
    part5_tradeoffs()
    part6_workflow()
    
    print("\n\n" + "=" * 80)
    print("TUTORIAL COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Run practical_example.py for a complete workflow example")
    print("2. Explore bias_detection.py, fairness_metrics.py, and bias_mitigation.py")
    print("3. Apply these techniques to your own datasets")
    print("\nRemember: Fairness is complex and context-dependent.")
    print("Always involve domain experts and affected communities!")
    print("=" * 80)


if __name__ == "__main__":
    run_tutorial()
