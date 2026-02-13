"""
================================================================================
01_breast_cancer_diagnosis.py - Complete Medical Diagnosis Pipeline
================================================================================

LEARNING OBJECTIVES:
- Build a complete end-to-end machine learning pipeline
- Handle real medical data with proper validation
- Implement comprehensive evaluation metrics
- Create production-ready inference code
- Understand clinical implications of ML predictions

PREREQUISITES:
- All basics and intermediate tutorials completed
- Understanding of classification metrics
- Knowledge of model evaluation

TIME TO COMPLETE: ~2 hours

DIFFICULTY: ⭐⭐⭐⭐☆ (Advanced Application)
================================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("BREAST CANCER DIAGNOSIS - COMPLETE CLINICAL APPLICATION")
print("="*80)

# =============================================================================
# PART 1: DATA LOADING AND EXPLORATION
# =============================================================================

class BreastCancerDataset:
    """
    Wrapper class for Breast Cancer dataset with preprocessing
    """
    
    def __init__(self, test_size=0.2, val_size=0.2, random_state=42):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.scaler = None
        self.feature_names = None
        
        self._load_data()
        self._preprocess()
    
    def _load_data(self):
        """Load raw data"""
        bc = datasets.load_breast_cancer()
        self.X = bc.data
        self.y = bc.target
        self.feature_names = bc.feature_names
        self.target_names = bc.target_names
        
        print("\n1. Dataset Information")
        print("-" * 40)
        print(f"Total samples: {len(self.X)}")
        print(f"Features: {len(self.feature_names)}")
        print(f"Target classes: {self.target_names}")
        print(f"Class distribution:")
        for i, name in enumerate(self.target_names):
            count = (self.y == i).sum()
            pct = 100 * count / len(self.y)
            print(f"  {name}: {count} ({pct:.1f}%)")
    
    def _preprocess(self):
        """Split and standardize data"""
        # Three-way split
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, 
            random_state=self.random_state, stratify=self.y
        )
        
        val_size_adjusted = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted,
            random_state=self.random_state, stratify=y_temp
        )
        
        # Standardization
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        
        # Convert to tensors
        self.X_train = torch.FloatTensor(X_train)
        self.X_val = torch.FloatTensor(X_val)
        self.X_test = torch.FloatTensor(X_test)
        self.y_train = torch.FloatTensor(y_train).reshape(-1, 1)
        self.y_val = torch.FloatTensor(y_val).reshape(-1, 1)
        self.y_test = torch.FloatTensor(y_test).reshape(-1, 1)
        
        print(f"\nData split completed:")
        print(f"  Training: {len(self.X_train)} samples")
        print(f"  Validation: {len(self.X_val)} samples")
        print(f"  Test: {len(self.X_test)} samples")

# =============================================================================
# PART 2: MODEL DEFINITION
# =============================================================================

class ClinicalLogisticRegression(nn.Module):
    """
    Logistic Regression for Clinical Diagnosis
    
    Includes methods for:
    - Training
    - Prediction
    - Feature importance analysis
    - Confidence-aware predictions
    """
    
    def __init__(self, n_features):
        super(ClinicalLogisticRegression, self).__init__()
        self.linear = nn.Linear(n_features, 1)
        
    def forward(self, x):
        """Standard forward pass"""
        return torch.sigmoid(self.linear(x))
    
    def predict_with_confidence(self, x, threshold=0.5):
        """
        Predict with confidence scores
        
        Returns:
            predictions: Binary predictions (0/1)
            probabilities: Confidence scores
            confidence_level: High/Medium/Low based on probability
        """
        self.eval()
        with torch.no_grad():
            probs = self.forward(x)
            preds = (probs >= threshold).float()
            
            # Confidence levels
            confidence = torch.zeros_like(probs)
            confidence[(probs > 0.7) | (probs < 0.3)] = 2  # High
            confidence[((probs >= 0.45) & (probs <= 0.55))] = 0  # Low
            confidence[((probs >= 0.3) & (probs < 0.45)) | 
                      ((probs > 0.55) & (probs <= 0.7))] = 1  # Medium
            
        return preds, probs, confidence
    
    def get_feature_importance(self):
        """Get absolute feature importances (weights)"""
        return torch.abs(self.linear.weight.data).flatten()

# =============================================================================
# PART 3: TRAINING WITH COMPREHENSIVE EVALUATION
# =============================================================================

def train_clinical_model(model, dataset, num_epochs=300, lr=0.01, patience=20):
    """
    Train model with early stopping and comprehensive tracking
    """
    print("\n2. Training Clinical Model")
    print("-" * 40)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': [], 'val_auc': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        y_pred = model(dataset.X_train)
        loss = criterion(y_pred, dataset.y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Training metrics
        with torch.no_grad():
            train_acc = ((y_pred >= 0.5).float() == dataset.y_train).float().mean()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(dataset.X_val)
            val_loss = criterion(val_pred, dataset.y_val)
            val_acc = ((val_pred >= 0.5).float() == dataset.y_val).float().mean()
            
            # Additional metrics
            y_val_np = dataset.y_val.numpy().flatten()
            val_pred_class = (val_pred >= 0.5).float().numpy().flatten()
            val_pred_prob = val_pred.numpy().flatten()
            
            val_precision = precision_score(y_val_np, val_pred_class, zero_division=0)
            val_recall = recall_score(y_val_np, val_pred_class, zero_division=0)
            val_f1 = f1_score(y_val_np, val_pred_class, zero_division=0)
            val_auc = roc_auc_score(y_val_np, val_pred_prob)
        
        # Store history
        history['train_loss'].append(loss.item())
        history['train_acc'].append(train_acc.item())
        history['val_loss'].append(val_loss.item())
        history['val_acc'].append(val_acc.item())
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        history['val_auc'].append(val_auc)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            model.load_state_dict(best_model_state)
            break
        
        # Print progress
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:3d}: "
                  f"Loss={loss.item():.4f} "
                  f"Val_Acc={val_acc:.4f} "
                  f"Val_AUC={val_auc:.4f}")
    
    return history

# =============================================================================
# PART 4: COMPREHENSIVE EVALUATION
# =============================================================================

def evaluate_clinical_performance(model, dataset):
    """Comprehensive clinical evaluation"""
    print("\n3. Clinical Performance Evaluation")
    print("-" * 40)
    
    model.eval()
    with torch.no_grad():
        # Test predictions
        test_pred_prob = model(dataset.X_test)
        test_pred_class = (test_pred_prob >= 0.5).float()
        
        # Convert to numpy
        y_test_np = dataset.y_test.numpy().flatten()
        test_pred_np = test_pred_class.numpy().flatten()
        test_prob_np = test_pred_prob.numpy().flatten()
    
    # Calculate all metrics
    metrics = {
        'accuracy': accuracy_score(y_test_np, test_pred_np),
        'precision': precision_score(y_test_np, test_pred_np),
        'recall': recall_score(y_test_np, test_pred_np),
        'f1': f1_score(y_test_np, test_pred_np),
        'auc': roc_auc_score(y_test_np, test_prob_np),
        'confusion_matrix': confusion_matrix(y_test_np, test_pred_np)
    }
    
    # Print results
    print("\nTest Set Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f} (Of predicted benign, how many correct?)")
    print(f"  Recall:    {metrics['recall']:.4f} (Of actual benign, how many found?)")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['auc']:.4f}")
    
    # Confusion matrix breakdown
    tn, fp, fn, tp = metrics['confusion_matrix'].ravel()
    print(f"\nConfusion Matrix Breakdown:")
    print(f"  True Negatives (TN):  {tn:3d} - Correctly identified malignant")
    print(f"  False Positives (FP): {fp:3d} - Incorrectly called benign (False alarm)")
    print(f"  False Negatives (FN): {fn:3d} - Missed cancers (CRITICAL!)")
    print(f"  True Positives (TP):  {tp:3d} - Correctly identified benign")
    
    # Clinical interpretation
    print(f"\nClinical Interpretation:")
    if fn > 0:
        print(f"  ⚠️  {fn} cancers were missed by the model")
        print(f"     This is the most critical error in medical diagnosis")
    else:
        print(f"  ✅ All cancers were detected!")
    
    if fp > 0:
        print(f"  ⚠️  {fp} patients had false alarms")
        print(f"     May lead to unnecessary procedures/anxiety")
    
    return metrics, test_prob_np

# =============================================================================
# PART 5: VISUALIZATION
# =============================================================================

def create_clinical_visualizations(history, metrics, test_probs, dataset, model):
    """Create comprehensive clinical visualizations"""
    print("\n4. Creating Clinical Visualizations")
    print("-" * 40)
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Training History
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(history['train_loss'], label='Train', alpha=0.7)
    ax1.plot(history['val_loss'], label='Validation', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Progress: Loss', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy Progress
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(history['train_acc'], label='Train', alpha=0.7)
    ax2.plot(history['val_acc'], label='Validation', alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Progress: Accuracy', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Validation Metrics
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(history['val_precision'], label='Precision', alpha=0.7)
    ax3.plot(history['val_recall'], label='Recall', alpha=0.7)
    ax3.plot(history['val_f1'], label='F1', alpha=0.7)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Score')
    ax3.set_title('Validation Metrics', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Confusion Matrix
    ax4 = plt.subplot(3, 3, 4)
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', 
                cmap='Blues', ax=ax4, cbar=False)
    ax4.set_xlabel('Predicted')
    ax4.set_ylabel('Actual')
    ax4.set_title('Confusion Matrix', fontweight='bold')
    ax4.set_xticklabels(['Malignant', 'Benign'])
    ax4.set_yticklabels(['Malignant', 'Benign'])
    
    # 5. ROC Curve
    ax5 = plt.subplot(3, 3, 5)
    y_test_np = dataset.y_test.numpy().flatten()
    fpr, tpr, thresholds = roc_curve(y_test_np, test_probs)
    ax5.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC={metrics["auc"]:.3f})')
    ax5.plot([0, 1], [0, 1], 'r--', label='Random Classifier')
    ax5.set_xlabel('False Positive Rate')
    ax5.set_ylabel('True Positive Rate')
    ax5.set_title('ROC Curve', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Prediction Distribution
    ax6 = plt.subplot(3, 3, 6)
    ax6.hist(test_probs[y_test_np==0], bins=30, alpha=0.6, label='Malignant', color='red')
    ax6.hist(test_probs[y_test_np==1], bins=30, alpha=0.6, label='Benign', color='blue')
    ax6.axvline(x=0.5, color='black', linestyle='--', label='Threshold')
    ax6.set_xlabel('Predicted Probability')
    ax6.set_ylabel('Count')
    ax6.set_title('Prediction Distribution', fontweight='bold')
    ax6.legend()
    
    # 7. Feature Importance
    ax7 = plt.subplot(3, 3, 7)
    feature_importance = model.get_feature_importance().numpy()
    top_k = 10
    top_indices = np.argsort(feature_importance)[-top_k:]
    top_features = [dataset.feature_names[i] for i in top_indices]
    top_values = feature_importance[top_indices]
    
    y_pos = np.arange(len(top_features))
    ax7.barh(y_pos, top_values, color='green', alpha=0.7)
    ax7.set_yticks(y_pos)
    ax7.set_yticklabels(top_features, fontsize=8)
    ax7.set_xlabel('Absolute Weight')
    ax7.set_title(f'Top {top_k} Most Important Features', fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='x')
    
    # 8. Metrics Comparison
    ax8 = plt.subplot(3, 3, 8)
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    metric_values = [metrics['accuracy'], metrics['precision'], 
                    metrics['recall'], metrics['f1'], metrics['auc']]
    bars = ax8.bar(metric_names, metric_values, color=['blue', 'green', 'orange', 'red', 'purple'], alpha=0.7)
    ax8.set_ylim([0, 1])
    ax8.set_ylabel('Score')
    ax8.set_title('Test Set Metrics', fontweight='bold')
    ax8.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    # 9. Summary
    ax9 = plt.subplot(3, 3, 9)
    tn, fp, fn, tp = metrics['confusion_matrix'].ravel()
    summary = f"""
CLINICAL SUMMARY
{'='*35}

Test Set: {len(dataset.X_test)} patients

Performance:
  Accuracy:  {metrics['accuracy']:.1%}
  Sensitivity (Recall): {metrics['recall']:.1%}
  Specificity: {tn/(tn+fp):.1%}
  
Critical Metrics:
  Missed Cancers (FN): {fn}
  False Alarms (FP): {fp}
  
Clinical Interpretation:
  - Model detected {tp}/{tp+fn} benign cases
  - Model detected {tn}/{tn+fp} malignant cases
  
{'✅ ACCEPTABLE' if fn == 0 else '⚠️ NEEDS IMPROVEMENT'}
Reason: {'All cancers detected' if fn == 0 else f'{fn} cancers missed'}
"""
    ax9.text(0.05, 0.5, summary, fontsize=9, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax9.axis('off')
    
    plt.tight_layout()
    save_path = '/home/claude/pytorch_logistic_regression_tutorial/04_applications/clinical_diagnosis_results.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution pipeline"""
    print("\n" + "="*80)
    print("RUNNING COMPLETE CLINICAL DIAGNOSIS PIPELINE")
    print("="*80)
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load and preprocess data
    dataset = BreastCancerDataset()
    
    # Create and train model
    model = ClinicalLogisticRegression(n_features=dataset.X_train.shape[1])
    history = train_clinical_model(model, dataset, num_epochs=300, lr=0.01, patience=20)
    
    # Evaluate
    metrics, test_probs = evaluate_clinical_performance(model, dataset)
    
    # Visualize
    create_clinical_visualizations(history, metrics, test_probs, dataset, model)
    
    print("\n" + "="*80)
    print("✅ CLINICAL PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)

if __name__ == "__main__":
    main()

print("""
================================================================================
KEY TAKEAWAYS - CLINICAL ML APPLICATIONS
================================================================================

1. MEDICAL CONTEXT MATTERS
   - False negatives (missed diseases) are often worse than false positives
   - Consider using higher recall even if precision suffers
   - Always provide confidence scores, not just binary predictions

2. COMPREHENSIVE EVALUATION
   - Never rely on accuracy alone
   - Use precision, recall, F1, AUC
   - Confusion matrix provides actionable insights

3. FEATURE IMPORTANCE
   - Understanding which features matter helps clinicians
   - Can validate model is using medically relevant features
   - Helps build trust in the model

4. PRODUCTION CONSIDERATIONS
   - Include confidence levels in predictions
   - Provide explanations when possible
   - Regular retraining with new data
   - Human oversight is essential

5. ETHICAL CONSIDERATIONS
   - Model is a tool to assist, not replace, doctors
   - Consider bias in training data
   - Transparent reporting of limitations
   - Patient privacy and data security

================================================================================
NEXT STEPS FOR DEPLOYMENT
================================================================================

1. Clinical Validation
   - Test on additional datasets from different hospitals
   - Validate with medical professionals
   - Conduct prospective clinical studies

2. Regulatory Approval
   - FDA approval process for medical devices
   - Documentation of training data and performance
   - Regular audits and updates

3. Integration
   - API for hospital systems
   - User-friendly interface for clinicians
   - Integration with electronic health records

4. Monitoring
   - Track model performance over time
   - Detect data drift
   - Regular retraining schedule

5. Continuous Improvement
   - Collect feedback from users
   - Update with latest research
   - Expand to other cancer types or conditions
================================================================================
""")
