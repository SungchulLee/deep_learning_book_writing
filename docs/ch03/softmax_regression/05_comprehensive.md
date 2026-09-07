# 종합 프로젝트

5단계: 여러 데이터셋을 아우르는 종합 분류 프로젝트

이 튜토리얼은 PyTorch에서 소프트맥스 회귀에 대한 기초적인 이해를 쌓는다. 코드를 따라가 보면 모델을 세우고, 손실 함수를 정의하고, 경사 하강법으로 학습시키고, 분류 과제에서 성능을 평가하는 법을 알 수 있다.

## 코드

```python
"""
===============================================================================
LEVEL 5: Comprehensive Multi-Dataset Classification Project
===============================================================================
Difficulty: Advanced
Prerequisites: Level 1-4
배움 목표:
  - Work with multiple real datasets
  - Build a flexible model factory
  - Implement comprehensive experiment tracking
  - Create reusable training pipelines
  - Generate detailed performance reports
  - Compare architectures systematically

Time to complete: 90-120 minutes
===============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
from collections import defaultdict

# 난수 씨앗을 설정한다
torch.manual_seed(42)
np.random.seed(42)

print("=" * 80)
print("LEVEL 5: COMPREHENSIVE MULTI-DATASET CLASSIFICATION")
print("=" * 80)


# =============================================================================
# 1부: 데이터셋 관리자
# =============================================================================
print("\n" + "=" * 80)
print("PART 1: Building a Dataset Manager")
print("=" * 80)

class DatasetManager:
    """
    Centralized dataset loading and preprocessing.
    Supports multiple datasets with consistent interface.
    """
    
    def __init__(self, dataset_name, batch_size=128, val_split=0.1):
        """
        Initialize dataset manager.
        
        Args:
            dataset_name: 'mnist', 'fashion_mnist', or 'cifar10'
            batch_size: Batch size for data loaders
            val_split: Fraction of training data for validation
        """
        self.dataset_name = dataset_name.lower()
        self.batch_size = batch_size
        self.val_split = val_split
        
        # 데이터셋을 불러온다
        self.train_loader, self.val_loader, self.test_loader = self._load_dataset()
        self.num_classes = self._get_num_classes()
        self.input_shape = self._get_input_shape()
        
    def _load_dataset(self):
        """지정한 데이터셋을 불러와 준비한다."""
        if self.dataset_name == 'mnist':
            return self._load_mnist()
        elif self.dataset_name == 'fashion_mnist':
            return self._load_fashion_mnist()
        elif self.dataset_name == 'cifar10':
            return self._load_cifar10()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _load_mnist(self):
        """MNIST 데이터셋을 불러온다."""
        print(f"Loading MNIST...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)
        
        return self._create_loaders(train_dataset, test_dataset)
    
    def _load_fashion_mnist(self):
        """Fashion-MNIST 데이터셋을 불러온다."""
        print(f"Loading Fashion-MNIST...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        train_dataset = datasets.FashionMNIST(root='./data', train=True,
                                             download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False,
                                            download=True, transform=transform)
        
        return self._create_loaders(train_dataset, test_dataset)
    
    def _load_cifar10(self):
        """CIFAR-10 데이터셋을 불러온다."""
        print(f"Loading CIFAR-10...")
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                               (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                               (0.2023, 0.1994, 0.2010))
        ])
        
        train_dataset = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
        
        return self._create_loaders(train_dataset, test_dataset)
    
    def _create_loaders(self, train_dataset, test_dataset):
        """학습, 검증, 시험 로더를 만든다."""
        # 학습 데이터를 나눈다
        val_size = int(len(train_dataset) * self.val_split)
        train_size = len(train_dataset) - val_size
        
        train_subset, val_subset = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(train_subset, batch_size=self.batch_size,
                                 shuffle=True, num_workers=2)
        val_loader = DataLoader(val_subset, batch_size=self.batch_size,
                               shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size,
                                shuffle=False, num_workers=2)
        
        return train_loader, val_loader, test_loader
    
    def _get_num_classes(self):
        """데이터셋의 클래스 개수를 얻는다."""
        if self.dataset_name in ['mnist', 'fashion_mnist', 'cifar10']:
            return 10
        return None
    
    def _get_input_shape(self):
        """데이터셋의 입력 모양을 얻는다."""
        # 표본 배치를 하나 얻는다
        sample_batch = next(iter(self.train_loader))[0]
        return sample_batch.shape[1:]  # Remove batch dimension
    
    def get_info(self):
        """데이터셋 정보를 반환한다."""
        return {
            'name': self.dataset_name,
            'num_classes': self.num_classes,
            'input_shape': self.input_shape,
            'train_samples': len(self.train_loader.dataset),
            'val_samples': len(self.val_loader.dataset),
            'test_samples': len(self.test_loader.dataset),
            'batch_size': self.batch_size
        }


# =============================================================================
# 2부: 모델 팩토리
# =============================================================================
print("\n" + "=" * 80)
print("PART 2: Building a Model Factory")
print("=" * 80)

class ModelFactory:
    """
    Factory for creating different model architectures.
    """
    
    @staticmethod
    def create_model(model_type, input_shape, num_classes, **kwargs):
        """
        Create a model based on type.
        
        Args:
            model_type: 'simple', 'medium', or 'deep'
            input_shape: Input tensor shape (C, H, W)
            num_classes: 내놓음 갈래의 수
            **kwargs: Additional model parameters
        
        Returns:
            PyTorch model
        """
        if model_type == 'simple':
            return ModelFactory._simple_model(input_shape, num_classes, **kwargs)
        elif model_type == 'medium':
            return ModelFactory._medium_model(input_shape, num_classes, **kwargs)
        elif model_type == 'deep':
            return ModelFactory._deep_model(input_shape, num_classes, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def _simple_model(input_shape, num_classes, dropout=0.2):
        """단순한 2층 완전 연결 신경망."""
        input_size = int(np.prod(input_shape))
        
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.flatten = nn.Flatten()
                self.fc1 = nn.Linear(input_size, 128)
                self.relu1 = nn.ReLU()
                self.dropout1 = nn.Dropout(dropout)
                self.fc2 = nn.Linear(128, num_classes)
            
            def forward(self, x):
                x = self.flatten(x)
                x = self.fc1(x)
                x = self.relu1(x)
                x = self.dropout1(x)
                x = self.fc2(x)
                return x
        
        return SimpleModel()
    
    @staticmethod
    def _medium_model(input_shape, num_classes, dropout=0.3):
        """배치 정규화를 넣은 중간 크기의 3층 완전 연결 신경망."""
        input_size = int(np.prod(input_shape))
        
        class MediumModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.flatten = nn.Flatten()
                self.fc1 = nn.Linear(input_size, 256)
                self.bn1 = nn.BatchNorm1d(256)
                self.relu1 = nn.ReLU()
                self.dropout1 = nn.Dropout(dropout)
                
                self.fc2 = nn.Linear(256, 128)
                self.bn2 = nn.BatchNorm1d(128)
                self.relu2 = nn.ReLU()
                self.dropout2 = nn.Dropout(dropout)
                
                self.fc3 = nn.Linear(128, num_classes)
            
            def forward(self, x):
                x = self.flatten(x)
                x = self.fc1(x)
                x = self.bn1(x)
                x = self.relu1(x)
                x = self.dropout1(x)
                
                x = self.fc2(x)
                x = self.bn2(x)
                x = self.relu2(x)
                x = self.dropout2(x)
                
                x = self.fc3(x)
                return x
        
        return MediumModel()
    
    @staticmethod
    def _deep_model(input_shape, num_classes, dropout=0.4):
        """깊은 4층 완전 연결 신경망."""
        input_size = int(np.prod(input_shape))
        
        class DeepModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.flatten = nn.Flatten()
                
                self.fc1 = nn.Linear(input_size, 512)
                self.bn1 = nn.BatchNorm1d(512)
                self.relu1 = nn.ReLU()
                self.dropout1 = nn.Dropout(dropout)
                
                self.fc2 = nn.Linear(512, 256)
                self.bn2 = nn.BatchNorm1d(256)
                self.relu2 = nn.ReLU()
                self.dropout2 = nn.Dropout(dropout)
                
                self.fc3 = nn.Linear(256, 128)
                self.bn3 = nn.BatchNorm1d(128)
                self.relu3 = nn.ReLU()
                self.dropout3 = nn.Dropout(dropout)
                
                self.fc4 = nn.Linear(128, num_classes)
            
            def forward(self, x):
                x = self.flatten(x)
                
                x = self.fc1(x)
                x = self.bn1(x)
                x = self.relu1(x)
                x = self.dropout1(x)
                
                x = self.fc2(x)
                x = self.bn2(x)
                x = self.relu2(x)
                x = self.dropout2(x)
                
                x = self.fc3(x)
                x = self.bn3(x)
                x = self.relu3(x)
                x = self.dropout3(x)
                
                x = self.fc4(x)
                return x
        
        return DeepModel()


# =============================================================================
# 3부: 종합 학습기
# =============================================================================
print("\n" + "=" * 80)
print("PART 3: Building a Comprehensive Trainer")
print("=" * 80)

class Trainer:
    """
    Comprehensive training pipeline with experiment tracking.
    """
    
    def __init__(self, model, train_loader, val_loader, test_loader,
                 criterion, optimizer, device, experiment_name="experiment"):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            train_loader, val_loader, test_loader: Data loaders
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            experiment_name: Name for saving results
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.experiment_name = experiment_name
        
        # 추적을 초기화한다
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': []
        }
        self.best_val_acc = 0.0
        self.best_model_state = None
        
    def train_epoch(self):
        """한 에폭을 학습한다."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # 순전파
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # 역전파
            loss.backward()
            self.optimizer.step()
            
            # 지표를 추적한다
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        avg_loss = running_loss / len(self.train_loader)
        accuracy = correct / total
        return avg_loss, accuracy
    
    def validate(self):
        """모델을 검증한다."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = running_loss / len(self.val_loader)
        accuracy = correct / total
        return avg_loss, accuracy
    
    def test(self):
        """모델을 시험한다."""
        self.model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        accuracy = correct / total
        return accuracy, np.array(all_predictions), np.array(all_targets)
    
    def train(self, num_epochs, scheduler=None, early_stopping_patience=None):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            scheduler: Learning rate scheduler (optional)
            early_stopping_patience: Patience for early stopping (optional)
        
        Returns:
            Dictionary with training results
        """
        print(f"\n{'='*80}")
        print(f"Training: {self.experiment_name}")
        print(f"{'='*80}")
        
        start_time = time.time()
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # 학습하고 검증한다
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            # 지표를 추적한다
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # 학습률을 추적한다
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)
            
            # 학습률을 갱신한다
            if scheduler is not None:
                scheduler.step()
            
            # 최고 성능 모델 저장
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 진행 상황 출력
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}]")
                print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
                print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}")
                print(f"  LR: {current_lr:.6f}")
            
            # 조기 종료
            if early_stopping_patience and patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        # 가장 좋은 모델을 불러온다
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # 최종 모델을 시험한다
        test_acc, predictions, targets = self.test()
        
        training_time = time.time() - start_time
        
        print(f"\n{'='*80}")
        print(f"Training Complete!")
        print(f"  Time: {training_time:.2f}s")
        print(f"  Best Val Acc: {self.best_val_acc:.4f}")
        print(f"  Test Acc: {test_acc:.4f}")
        print(f"{'='*80}\n")
        
        return {
            'history': self.history,
            'best_val_acc': self.best_val_acc,
            'test_acc': test_acc,
            'predictions': predictions,
            'targets': targets,
            'training_time': training_time,
            'experiment_name': self.experiment_name
        }


# =============================================================================
# 4부: 실험 실행기
# =============================================================================
print("\n" + "=" * 80)
print("PART 4: Running Comprehensive Experiments")
print("=" * 80)

class ExperimentRunner:
    """
    Run and compare multiple experiments.
    """
    
    def __init__(self):
        self.results = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def run_experiment(self, dataset_name, model_type, num_epochs=20,
                      lr=0.001, use_scheduler=False):
        """
        Run a single experiment.
        
        Args:
            dataset_name: Name of dataset
            model_type: Type of model
            num_epochs: Number of training epochs
            lr: Learning rate
            use_scheduler: Whether to use LR scheduling
        """
        experiment_name = f"{dataset_name}_{model_type}_lr{lr}"
        print(f"\n{'='*80}")
        print(f"Experiment: {experiment_name}")
        print(f"{'='*80}")
        
        # 데이터셋을 불러온다
        dm = DatasetManager(dataset_name, batch_size=128)
        info = dm.get_info()
        print(f"\nDataset: {info['name']}")
        print(f"  Classes: {info['num_classes']}")
        print(f"  Input shape: {info['input_shape']}")
        print(f"  Train samples: {info['train_samples']}")
        
        # 모델 생성
        model = ModelFactory.create_model(
            model_type,
            info['input_shape'],
            info['num_classes']
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nModel: {model_type}")
        print(f"  Total parameters: {total_params:,}")
        
        # 학습 준비
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        scheduler = None
        if use_scheduler:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        # 학습
        trainer = Trainer(
            model, dm.train_loader, dm.val_loader, dm.test_loader,
            criterion, optimizer, self.device, experiment_name
        )
        
        result = trainer.train(
            num_epochs=num_epochs,
            scheduler=scheduler,
            early_stopping_patience=10
        )
        
        # 결과를 저장한다
        result['dataset'] = dataset_name
        result['model_type'] = model_type
        result['num_params'] = total_params
        result['learning_rate'] = lr
        result['use_scheduler'] = use_scheduler
        
        self.results.append(result)
        
        return result
    
    def generate_report(self):
        """종합적인 비교 보고서를 만든다."""
        print(f"\n{'='*80}")
        print("EXPERIMENT COMPARISON REPORT")
        print(f"{'='*80}\n")
        
        # 시험 정확도순으로 정렬한다
        sorted_results = sorted(self.results, key=lambda x: x['test_acc'], reverse=True)
        
        print(f"{'Rank':<6} {'Experiment':<35} {'Val Acc':<10} {'Test Acc':<10} {'Time(s)':<10}")
        print("-" * 80)
        
        for i, result in enumerate(sorted_results, 1):
            name = result['experiment_name']
            val_acc = result['best_val_acc']
            test_acc = result['test_acc']
            time_taken = result['training_time']
            
            print(f"{i:<6} {name:<35} {val_acc:.4f}{'':>4} {test_acc:.4f}{'':>4} {time_taken:.2f}")
        
        # 데이터셋별 최고 성능
        print(f"\n{'='*80}")
        print("BEST MODEL PER DATASET")
        print(f"{'='*80}\n")
        
        datasets = set(r['dataset'] for r in self.results)
        for dataset in datasets:
            dataset_results = [r for r in self.results if r['dataset'] == dataset]
            best = max(dataset_results, key=lambda x: x['test_acc'])
            
            print(f"{dataset}:")
            print(f"  Best model: {best['model_type']}")
            print(f"  Test accuracy: {best['test_acc']:.4f}")
            print(f"  Parameters: {best['num_params']:,}")
            print()
        
        return sorted_results


# =============================================================================
# 5부: 실험 실행
# =============================================================================
print("\n" + "=" * 80)
print("PART 5: Running Multiple Experiments")
print("=" * 80)

# 실험 실행기를 만든다
runner = ExperimentRunner()

# 실행할 실험을 정의한다 (시연을 위해 일부만)
experiments = [
    # 데이터셋, 모델 종류, 에폭, 학습률, 스케줄러 사용 여부
    ('mnist', 'simple', 15, 0.001, False),
    ('mnist', 'medium', 15, 0.001, True),
    ('fashion_mnist', 'simple', 15, 0.001, False),
    ('fashion_mnist', 'medium', 15, 0.001, True),
]

print("\nRunning experiments...")
print("Note: Running limited experiments for demonstration.")
print("For full comparison, uncomment additional experiments below.\n")

# 실험을 실행한다
for dataset, model_type, epochs, lr, use_sched in experiments:
    try:
        runner.run_experiment(dataset, model_type, epochs, lr, use_sched)
    except Exception as e:
        print(f"Error in experiment: {e}")
        continue

# 더 폭넓게 시험하려면 주석을 푼다:
# experiments_full = [
#     ('mnist', 'simple', 20, 0.001, False),
#     ('mnist', 'medium', 20, 0.001, True),
#     ('mnist', 'deep', 20, 0.001, True),
#     ('fashion_mnist', 'simple', 20, 0.001, False),
#     ('fashion_mnist', 'medium', 20, 0.001, True),
#     ('fashion_mnist', 'deep', 20, 0.001, True),
#     ('cifar10', 'simple', 30, 0.001, False),
#     ('cifar10', 'medium', 30, 0.001, True),
#     ('cifar10', 'deep', 30, 0.001, True),
# ]

# 보고서를 만든다
if runner.results:
    final_report = runner.generate_report()


# =============================================================================
# 요약
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY - What You Accomplished")
print("=" * 80)

print("""
✅ Built a comprehensive dataset manager supporting multiple datasets
✅ Created a model factory for different architectures
✅ Implemented a full-featured training pipeline
✅ Tracked experiments with detailed metrics
✅ Compared multiple models and datasets systematically
✅ Generated performance comparison reports

Project Structure:
------------------
1. DatasetManager
   - Unified interface for multiple datasets
   - Automatic train/val/test splitting
   - Configurable batch sizes and preprocessing

2. ModelFactory
   - Simple, Medium, Deep architectures
   - Flexible parameter configuration
   - Easy to extend with new models

3. Trainer
   - Complete training loop
   - 배움 빠르기 짜기
   - 일찍 멈추기
   - Best model tracking
   - Comprehensive history

4. ExperimentRunner
   - Run multiple experiments
   - Automatic comparison
   - Report generation
   - Results tracking

고갱이 배움:
--------------
• Modular design enables easy experimentation
• Systematic comparison reveals best approaches
• Experiment tracking is crucial for reproducibility
• Different datasets need different architectures
• Hyperparameters significantly impact performance

Professional Tips:
------------------
1. Always track experiments systematically
2. Use validation set for model selection
3. Report results on held-out test set
4. Compare models fairly (same data, device, seeds)
5. Document all hyperparameters and configurations

다음 걸음:
-----------
→ Extend with more datasets (CIFAR-100, ImageNet, custom)
→ Add convolutional architectures
→ Implement more advanced techniques (mixup, cutout)
→ Create visualizations and tensorboard logging
→ Deploy best models for inference

🎉 Congratulations! You've completed the entire tutorial series!

You now have the skills to:
• Understand softmax regression from theory to practice
• Build and train deep learning classifiers
• Apply advanced techniques and best practices
• Run systematic experiments and comparisons
• Create production-ready training pipelines

Keep learning and experimenting! 🚀
""")


if __name__ == "__main__":
    pass
```

## 논의

이 구현은 4개의 클래스(`DatasetManager`, `ModelFactory`, `Trainer`, `ExperimentRunner`)를 정의하며, 이들이 함께 작동하여 완전한 소프트맥스 회귀 구조를 이룬다. 각 클래스가 서로 다른 구성 요소를 감싸므로 코드가 모듈식이 되고 확장하기 쉬워진다. `forward` 메서드들이 PyTorch가 자동 미분에 사용하는 계산 그래프를 정의한다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 확장된다. 초매개변수, 구조의 변형, 여러 데이터셋을 두루 실험해 보면 이해가 깊어지고 다중 클래스 분류 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
기본 초기화 상태의 `DatasetManager`에서 학습 가능한 매개변수의 총 개수를 계산하라. 가중치와 편향을 모두 포함하여 층별로 나누어 세어라.

??? success "연습문제 1 풀이"
    `nn.Linear(in_features, out_features)` 각각에는 `in_features * out_features`개의 가중치 매개변수와 (`bias=False`가 아닌 한) `out_features`개의 편향 매개변수가 있다. `nn.Conv2d(in_c, out_c, k)`에는 `in_c * out_c * k * k`개의 가중치와 `out_c`개의 편향이 있다. `nn.Embedding(num, dim)`에는 `num * dim`개의 매개변수가 있다. 모든 층에 대해 더하면 된다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다.

---

**연습문제 2.**
최적화기를 Adam으로 바꾸고(`torch.optim.Adam`에 `lr=0.001`을 쓴다) 원래 최적화기와 학습 수렴을 비교하라. 두 손실 곡선을 같은 그래프에 그려라.

??? success "연습문제 2 풀이"
    최적화기를 만드는 줄을 `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`으로 바꾼다. Adam은 매개변수마다 적응적인 학습률과 운동량 추정값을 유지하므로 초반 에폭에서 대체로 더 빠르게 수렴한다. Adam의 손실 곡선은 보통 처음 몇 에폭에서 더 가파르게 떨어지지만, 최적점 근처에서는 운동량을 쓴 SGD보다 조금 더 흔들릴 수 있다. 공정한 비교를 위해 둘을 같은 난수 씨앗과 같은 에폭 수로 실행하라.

---

**연습문제 3.**
이 구현에서 생길 수 있는 실패 양상 두 가지를 서술하고, 각각을 어떻게 진단하고 고칠지 설명하라.

??? success "연습문제 3 풀이"
    흔한 실패 양상은 다음과 같다. (1) **경사 소실/폭발** — 경사의 노름을 지켜보아 진단한다(`torch.nn.utils.clip_grad_norm_`을 쓰거나 층마다 `param.grad.norm()`을 기록한다). 경사 자르기, 더 나은 초기화(Xavier/Kaiming), 또는 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** — 학습 손실은 줄어드는데 검증 손실이 늘어나면 진단된다. 정칙화(드롭아웃, 가중치 감쇠, 데이터 증강)나 모델 용량 축소로 고친다. 이런 문제를 일찍 잡아내려면 언제나 학습 지표와 검증 지표를 함께 살펴라.

---

**연습문제 4.**
층이나 블록의 개수를 설정할 수 있도록 `DatasetManager`를 확장하라. `__init__`에 `num_layers` 매개변수를 추가하고 `nn.ModuleList`로 깊이가 변하는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`처럼 순회한다. (평범한 파이썬 리스트가 아니라) `nn.ModuleList`를 써야 PyTorch가 모든 매개변수를 최적화 대상으로 등록한다. 시험은 다음과 같이 한다. `for n in [2, 4, 8]: model = DatasetManager(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
