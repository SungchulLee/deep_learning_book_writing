"""
Practical Example: Bias Detection and Mitigation
Demonstrates bias detection and mitigation on a synthetic loan approval dataset.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from typing import Tuple


class LoanApprovalDataset:
    """Generate synthetic loan approval dataset with built-in bias."""
    
    @staticmethod
    def generate_data(
        n_samples: int = 2000,
        bias_strength: float = 0.3,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Generate synthetic loan approval data with bias.
        
        Args:
            n_samples: Number of samples
            bias_strength: Strength of bias (0-1)
            random_state: Random seed
            
        Returns:
            DataFrame with features and target
        """
        np.random.seed(random_state)
        
        # Generate features
        data = {
            'age': np.random.randint(18, 70, n_samples),
            'income': np.random.randint(20000, 200000, n_samples),
            'credit_score': np.random.randint(300, 850, n_samples),
            'employment_years': np.random.randint(0, 40, n_samples),
            'loan_amount': np.random.randint(5000, 500000, n_samples),
            'num_credit_cards': np.random.randint(0, 10, n_samples),
            'debt_to_income': np.random.uniform(0, 1, n_samples),
        }
        
        # Protected attributes
        data['gender'] = np.random.choice(['Male', 'Female'], n_samples)
        data['race'] = np.random.choice(['White', 'Black', 'Hispanic', 'Asian'], n_samples)
        
        df = pd.DataFrame(data)
        
        # Generate target with bias
        # Base approval probability on legitimate factors
        base_score = (
            (df['credit_score'] - 300) / 550 * 0.4 +
            (df['income'] - 20000) / 180000 * 0.3 +
            (1 - df['debt_to_income']) * 0.2 +
            (df['employment_years'] / 40) * 0.1
        )
        
        # Add bias based on protected attributes
        bias_factor = np.where(
            df['gender'] == 'Male',
            1 + bias_strength,
            1 - bias_strength
        )
        
        biased_score = base_score * bias_factor
        biased_score = np.clip(biased_score, 0, 1)
        
        # Convert to binary approval
        df['approved'] = (biased_score > 0.5).astype(int)
        
        return df


def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for modeling.
    
    Args:
        df: Input dataframe
        
    Returns:
        Tuple of (X, y, gender_binary, race_binary)
    """
    # Select features
    feature_cols = ['age', 'income', 'credit_score', 'employment_years',
                    'loan_amount', 'num_credit_cards', 'debt_to_income']
    
    X = df[feature_cols].values
    y = df['approved'].values
    
    # Convert protected attributes to binary for analysis
    gender_binary = (df['gender'] == 'Male').astype(int).values
    race_binary = (df['race'] == 'White').astype(int).values
    
    return X, y, gender_binary, race_binary


def detect_bias_in_data(df: pd.DataFrame):
    """
    Detect bias in the dataset before modeling.
    
    Args:
        df: Input dataframe
    """
    print("\n" + "=" * 80)
    print("DATA-LEVEL BIAS ANALYSIS")
    print("=" * 80)
    
    # Approval rates by gender
    print("\nApproval Rates by Gender:")
    print("-" * 40)
    gender_stats = df.groupby('gender')['approved'].agg(['mean', 'count'])
    print(gender_stats)
    
    male_rate = df[df['gender'] == 'Male']['approved'].mean()
    female_rate = df[df['gender'] == 'Female']['approved'].mean()
    print(f"\nGender Approval Gap: {abs(male_rate - female_rate):.4f}")
    print(f"Disparate Impact Ratio: {min(male_rate/female_rate, female_rate/male_rate):.4f}")
    
    # Approval rates by race
    print("\n\nApproval Rates by Race:")
    print("-" * 40)
    race_stats = df.groupby('race')['approved'].agg(['mean', 'count'])
    print(race_stats)
    
    # Feature statistics by protected group
    print("\n\nFeature Statistics by Gender:")
    print("-" * 40)
    feature_cols = ['income', 'credit_score', 'debt_to_income']
    for col in feature_cols:
        male_mean = df[df['gender'] == 'Male'][col].mean()
        female_mean = df[df['gender'] == 'Female'][col].mean()
        print(f"{col}: Male={male_mean:.2f}, Female={female_mean:.2f}, "
              f"Diff={abs(male_mean - female_mean):.2f}")


def train_baseline_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> RandomForestClassifier:
    """
    Train baseline model without bias mitigation.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Trained model
    """
    print("\n" + "=" * 80)
    print("BASELINE MODEL (No Bias Mitigation)")
    print("=" * 80)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model


def analyze_model_fairness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gender: np.ndarray,
    race: np.ndarray
):
    """
    Analyze fairness of model predictions.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        gender: Gender attribute (binary)
        race: Race attribute (binary)
    """
    print("\n" + "=" * 80)
    print("MODEL FAIRNESS ANALYSIS")
    print("=" * 80)
    
    # Gender fairness
    print("\nGender Fairness Metrics:")
    print("-" * 40)
    
    # Positive prediction rates
    male_pos_rate = np.mean(y_pred[gender == 1])
    female_pos_rate = np.mean(y_pred[gender == 0])
    print(f"Positive Prediction Rate (Male): {male_pos_rate:.4f}")
    print(f"Positive Prediction Rate (Female): {female_pos_rate:.4f}")
    print(f"Statistical Parity Difference: {abs(male_pos_rate - female_pos_rate):.4f}")
    
    # True positive rates
    male_mask = (gender == 1) & (y_true == 1)
    female_mask = (gender == 0) & (y_true == 1)
    
    if np.sum(male_mask) > 0:
        male_tpr = np.sum((y_pred == 1) & male_mask) / np.sum(male_mask)
    else:
        male_tpr = 0
    
    if np.sum(female_mask) > 0:
        female_tpr = np.sum((y_pred == 1) & female_mask) / np.sum(female_mask)
    else:
        female_tpr = 0
    
    print(f"True Positive Rate (Male): {male_tpr:.4f}")
    print(f"True Positive Rate (Female): {female_tpr:.4f}")
    print(f"Equal Opportunity Difference: {abs(male_tpr - female_tpr):.4f}")
    
    # Accuracy by group
    male_acc = accuracy_score(y_true[gender == 1], y_pred[gender == 1])
    female_acc = accuracy_score(y_true[gender == 0], y_pred[gender == 0])
    print(f"\nAccuracy (Male): {male_acc:.4f}")
    print(f"Accuracy (Female): {female_acc:.4f}")
    print(f"Accuracy Difference: {abs(male_acc - female_acc):.4f}")


def train_with_reweighing(
    X_train: np.ndarray,
    y_train: np.ndarray,
    gender_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> RandomForestClassifier:
    """
    Train model with reweighing bias mitigation.
    
    Args:
        X_train: Training features
        y_train: Training labels
        gender_train: Gender attribute for training
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Trained model
    """
    print("\n" + "=" * 80)
    print("MODEL WITH REWEIGHING")
    print("=" * 80)
    
    # Compute sample weights
    weights = np.ones(len(y_train))
    
    n = len(y_train)
    for gender_val in [0, 1]:
        for label_val in [0, 1]:
            mask = (gender_train == gender_val) & (y_train == label_val)
            p_observed = np.sum(mask) / n
            
            p_gender = np.sum(gender_train == gender_val) / n
            p_label = np.sum(y_train == label_val) / n
            p_expected = p_gender * p_label
            
            if p_observed > 0:
                weights[mask] = p_expected / p_observed
    
    print(f"Sample weights computed.")
    print(f"Weight statistics: min={np.min(weights):.4f}, "
          f"max={np.max(weights):.4f}, mean={np.mean(weights):.4f}")
    
    # Train with weights
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train, sample_weight=weights)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model


def main():
    """Main execution function."""
    print("=" * 80)
    print("BIAS AND FAIRNESS IN DEEP LEARNING - PRACTICAL EXAMPLE")
    print("Loan Approval Dataset")
    print("=" * 80)
    
    # Generate dataset
    print("\nGenerating synthetic loan approval dataset...")
    df = LoanApprovalDataset.generate_data(n_samples=2000, bias_strength=0.3)
    print(f"Dataset generated: {len(df)} samples")
    print(f"Approval rate: {df['approved'].mean():.4f}")
    
    # Detect bias in data
    detect_bias_in_data(df)
    
    # Prepare data
    X, y, gender, race = prepare_data(df)
    X_train, X_test, y_train, y_test, gender_train, gender_test = train_test_split(
        X, y, gender, test_size=0.3, random_state=42
    )
    
    # Train baseline model
    baseline_model = train_baseline_model(X_train, y_train, X_test, y_test)
    y_pred_baseline = baseline_model.predict(X_test)
    
    # Analyze baseline fairness
    analyze_model_fairness(y_test, y_pred_baseline, gender_test, gender_test)
    
    # Train with bias mitigation
    fair_model = train_with_reweighing(
        X_train, y_train, gender_train, X_test, y_test
    )
    y_pred_fair = fair_model.predict(X_test)
    
    # Analyze fair model
    analyze_model_fairness(y_test, y_pred_fair, gender_test, gender_test)
    
    # Compare models
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    
    baseline_acc = accuracy_score(y_test, y_pred_baseline)
    fair_acc = accuracy_score(y_test, y_pred_fair)
    
    baseline_spd = abs(
        np.mean(y_pred_baseline[gender_test == 1]) -
        np.mean(y_pred_baseline[gender_test == 0])
    )
    fair_spd = abs(
        np.mean(y_pred_fair[gender_test == 1]) -
        np.mean(y_pred_fair[gender_test == 0])
    )
    
    print(f"\nBaseline Model:")
    print(f"  Accuracy: {baseline_acc:.4f}")
    print(f"  Statistical Parity Difference: {baseline_spd:.4f}")
    
    print(f"\nFair Model (Reweighing):")
    print(f"  Accuracy: {fair_acc:.4f}")
    print(f"  Statistical Parity Difference: {fair_spd:.4f}")
    
    print(f"\nImprovement:")
    print(f"  Accuracy Change: {fair_acc - baseline_acc:.4f}")
    print(f"  Fairness Improvement: {baseline_spd - fair_spd:.4f}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
