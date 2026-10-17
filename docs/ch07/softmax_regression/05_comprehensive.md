# 종합 프로젝트

여러 이미지 데이터셋과 분류 모델을 같은 실험 인터페이스로 관리한다. 데이터 준비, 모델 생성, 학습과 결과 집계를 나누어 재사용하는 방법을 살펴본다.

## 1. 코드

??? note "코드 (790줄)"

    ```python
    """
    ===============================================================================
    5단계: 여러 데이터셋을 아우르는 분류 과제
    ===============================================================================
    어려움: 앞섬
    미리 알아 둘 것: 1~4단계
    학습 목표:
      - 참 데이터셋 여럿을 다룬다
      - 두루 쓸 수 있는 모델 공장을 짓는다
      - 실험을 두루 좇는 구조를 짠다
      - 다시 쓸 수 있는 학습 흐름을 만든다
      - 자세한 성능 알림을 만든다
      - 구조를 짜임새 있게 견준다

    소요 시간: 90~120분
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
        데이터셋 불러오기와 미리 다듬기를 한곳에 모은 것.
        한결같은 창구로 여러 데이터셋을 받쳐 준다.
        """

        def __init__(self, dataset_name, batch_size=128, val_split=0.1):
            """
            데이터셋 관리자의 초기화한다.

            Args:
                dataset_name: 'mnist', 'fashion_mnist', 'cifar10' 가운데 하나
                batch_size: 데이터 로더의 배치 크기
                val_split: 학습 데이터 가운데 검증에 쓸 몫
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
                                     # num_workers 를 0으로 둔다. 1 이상이면 맥과 윈도가 일꾼 프로세스를
                                     # **새로 띄우며**(spawn) 이 각본을 처음부터 다시 읽는다. 맨 바깥에서 자료를
                                     # 돌리고 있으면 일꾼이 또 일꾼을 띄워 끝내 죽는다. 1 이상을 쓰려면 돌리는
                                     # 부분을 모두 `if __name__ == "__main__":` 안으로 넣어야 한다.
                                     shuffle=True, num_workers=0)
            val_loader = DataLoader(val_subset, batch_size=self.batch_size,
                                   shuffle=False, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size,
                                    shuffle=False, num_workers=0)

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
        여러 모델 구조를 만드는 공장.
        """

        @staticmethod
        def create_model(model_type, input_shape, num_classes, **kwargs):
            """
            클래스에 따라 모델을 만든다.

            Args:
                model_type: 'simple', 'medium', 'deep' 가운데 하나
                input_shape: 입력 텐서의 모양 (C, H, W)
                num_classes: 출력 클래스의 수
                **kwargs: 덧붙이는 모델 파라메터

            Returns:
                PyTorch 모델
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
            """단순한 2레이어 완전 연결 뉴럴 넷트웍."""
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
            """배치 정규화를 넣은 중간 크기의 3레이어 완전 연결 뉴럴 넷트웍."""
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
            """깊은 4레이어 완전 연결 뉴럴 넷트웍."""
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
        실험 좇기를 곁들인 두루 갖춘 학습 흐름.
        """

        def __init__(self, model, train_loader, val_loader, test_loader,
                     criterion, optimizer, device, experiment_name="experiment"):
            """
            학습개의 초기화한다.

            Args:
                model: PyTorch 모델
                train_loader, val_loader, test_loader: 데이터 로더
                criterion: 로스 함수
                optimizer: 최적화기
                device: 익힐 장치
                experiment_name: 결과를 저장할 이름
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
            주요 학습 루프.

            Args:
                num_epochs: 익힐 에폭 수
                scheduler: 러닝레이트 짜개(골라 씀)
                early_stopping_patience: 조기 종료의 참을성(골라 씀)

            Returns:
                학습 결과를 담은 사전
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

                # 러닝레이트를 추적한다
                current_lr = self.optimizer.param_groups[0]['lr']
                self.history['learning_rates'].append(current_lr)

                # 러닝레이트를 갱신한다
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
        여러 실험을 돌리고 견준다.
        """

        def __init__(self):
            self.results = []
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {self.device}")

        def run_experiment(self, dataset_name, model_type, num_epochs=20,
                          lr=0.001, use_scheduler=False):
            """
            실험 하나를 돌린다.

            Args:
                dataset_name: 데이터셋의 이름
                model_type: 모델의 클래스
                num_epochs: 학습 에폭 수
                lr: 러닝레이트
                use_scheduler: 러닝레이트 짜기를 쓸지 여부
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
        # 데이터셋, 모델 종류, 에폭, 러닝레이트, 스케줄러 사용 여부
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
    ✅ 여러 데이터셋을 받쳐 주는 데이터셋 관리자를 지었다
    ✅ 여러 구조를 위한 모델 공장을 만들었다
    ✅ 온전히 갖춘 학습 흐름을 짰다
    ✅ 자세한 자로 실험을 좇았다
    ✅ 여러 모델과 데이터셋을 짜임새 있게 견주었다
    ✅ 성능 견줌 알림을 만들었다

    과제 구조:
    ------------------
    1. DatasetManager
       - 여러 데이터셋을 위한 하나로 된 창구
       - 학습/검증/시험 저절로 나누기
       - 배치 크기와 미리 다듬기를 맞출 수 있다

    2. ModelFactory
       - 단순, 보통, 깊은 구조
       - 파라메터를 두루 맞출 수 있다
       - 새 모델으로 넓히기 쉽다

    3. Trainer
       - 온전한 학습 루프
       - 러닝레이트 짜기
       - 조기 종료
       - 가장 좋은 모델 좇기
       - 두루 갖춘 자취

    4. ExperimentRunner
       - 여러 실험 돌리기
       - 저절로 견주기
       - 알림 만들기
       - 결과 좇기

    핵심 학습:
    --------------
    • 조각으로 나눈 설계는 실험을 쉽게 한다
    • 짜임새 있는 견줌이 가장 좋은 길을 드러낸다
    • 실험 좇기는 다시 해내기에 매우 종요롭다
    • 데이터셋이 다르면 구조도 달라야 한다
    • 초파라메터는 성능에 크게 미친다

    익은 이의 조언:
    ------------------
    1. 늘 실험을 짜임새 있게 좇아라
    2. 모델을 고를 때는 검증 배치를 써라
    3. 따로 떼어 둔 시험 배치의 결과를 알려라
    4. 모델을 고르게 견주어라(같은 데이터, 장치, 씨앗)
    5. 모든 초파라메터와 맞춤을 적어 두어라

    다음 스텝:
    -----------
    → 데이터셋을 더 넓힌다(CIFAR-100, ImageNet, 맞춤)
    → 컨볼루션 구조를 더한다
    → 더 앞선 기법을 짠다(믹스업, 컷아웃)
    → 그림 보기와 텐서보드 적기를 만든다
    → 가장 좋은 모델을 추론에 올린다

    🎉 잘했다! 학습 클래스 모두를 마쳤다!

    이제 다음을 할 기법을 갖추었다.
    • 소프트맥스 회귀를 이론에서 실제까지 이해한다
    • 깊은 학습 분류기를 짓고 익힌다
    • 앞선 기법와 가장 좋은 버릇을 쓴다
    • 짜임새 있는 실험과 견줌을 돌린다
    • 참으로 굴릴 수 있는 학습 흐름을 만든다

    배우고 실험하기를 이어 가라! 🚀
    """)


    if __name__ == "__main__":
        pass
    ```


??? note "전체 출력 (291줄)"

    ```
    ================================================================================
    LEVEL 5: COMPREHENSIVE MULTI-DATASET CLASSIFICATION
    ================================================================================

    ================================================================================
    PART 1: Building a Dataset Manager
    ================================================================================

    ================================================================================
    PART 2: Building a Model Factory
    ================================================================================

    ================================================================================
    PART 3: Building a Comprehensive Trainer
    ================================================================================

    ================================================================================
    PART 4: Running Comprehensive Experiments
    ================================================================================

    ================================================================================
    PART 5: Running Multiple Experiments
    ================================================================================
    Using device: cpu

    Running experiments...
    Note: Running limited experiments for demonstration.
    For full comparison, uncomment additional experiments below.


    ================================================================================
    Experiment: mnist_simple_lr0.001
    ================================================================================
    Loading MNIST...

    Dataset: mnist
      Classes: 10
      Input shape: torch.Size([1, 28, 28])
      Train samples: 54000

    Model: simple
      Total parameters: 101,770

    ================================================================================
    Training: mnist_simple_lr0.001
    ================================================================================
    Epoch [1/15]
      Train: Loss=0.3425, Acc=0.9000
      Val:   Loss=0.1817, Acc=0.9452
      LR: 0.001000
    Epoch [5/15]
      Train: Loss=0.0895, Acc=0.9724
      Val:   Loss=0.0888, Acc=0.9747
      LR: 0.001000
    Epoch [10/15]
      Train: Loss=0.0551, Acc=0.9819
      Val:   Loss=0.0819, Acc=0.9768
      LR: 0.001000
    Epoch [15/15]
      Train: Loss=0.0376, Acc=0.9873
      Val:   Loss=0.0780, Acc=0.9783
      LR: 0.001000

    ================================================================================
    Training Complete!
      Time: 28.84s
      Best Val Acc: 0.9793
      Test Acc: 0.9792
    ================================================================================


    ================================================================================
    Experiment: mnist_medium_lr0.001
    ================================================================================
    Loading MNIST...

    Dataset: mnist
      Classes: 10
      Input shape: torch.Size([1, 28, 28])
      Train samples: 54000

    Model: medium
      Total parameters: 235,914

    ================================================================================
    Training: mnist_medium_lr0.001
    ================================================================================
    Epoch [1/15]
      Train: Loss=0.3427, Acc=0.9081
      Val:   Loss=0.1378, Acc=0.9602
      LR: 0.001000
    Epoch [5/15]
      Train: Loss=0.0859, Acc=0.9732
      Val:   Loss=0.0783, Acc=0.9767
      LR: 0.001000
    Epoch [10/15]
      Train: Loss=0.0528, Acc=0.9831
      Val:   Loss=0.0698, Acc=0.9780
      LR: 0.001000
    Epoch [15/15]
      Train: Loss=0.0315, Acc=0.9904
      Val:   Loss=0.0571, Acc=0.9817
      LR: 0.000100

    ================================================================================
    Training Complete!
      Time: 29.72s
      Best Val Acc: 0.9825
      Test Acc: 0.9846
    ================================================================================


    ================================================================================
    Experiment: fashion_mnist_simple_lr0.001
    ================================================================================
    Loading Fashion-MNIST...

    Dataset: fashion_mnist
      Classes: 10
      Input shape: torch.Size([1, 28, 28])
      Train samples: 54000

    Model: simple
      Total parameters: 101,770

    ================================================================================
    Training: fashion_mnist_simple_lr0.001
    ================================================================================
    Epoch [1/15]
      Train: Loss=0.5716, Acc=0.7958
      Val:   Loss=0.4475, Acc=0.8395
      LR: 0.001000
    Epoch [5/15]
      Train: Loss=0.3438, Acc=0.8746
      Val:   Loss=0.3450, Acc=0.8762
      LR: 0.001000
    Epoch [10/15]
      Train: Loss=0.2961, Acc=0.8923
      Val:   Loss=0.3240, Acc=0.8813
      LR: 0.001000
    Epoch [15/15]
      Train: Loss=0.2610, Acc=0.9053
      Val:   Loss=0.3204, Acc=0.8843
      LR: 0.001000

    ================================================================================
    Training Complete!
      Time: 28.53s
      Best Val Acc: 0.8920
      Test Acc: 0.8778
    ================================================================================


    ================================================================================
    Experiment: fashion_mnist_medium_lr0.001
    ================================================================================
    Loading Fashion-MNIST...

    Dataset: fashion_mnist
      Classes: 10
      Input shape: torch.Size([1, 28, 28])
      Train samples: 54000

    Model: medium
      Total parameters: 235,914

    ================================================================================
    Training: fashion_mnist_medium_lr0.001
    ================================================================================
    Epoch [1/15]
      Train: Loss=0.5461, Acc=0.8149
      Val:   Loss=0.3944, Acc=0.8587
      LR: 0.001000
    Epoch [5/15]
      Train: Loss=0.3218, Acc=0.8807
      Val:   Loss=0.3202, Acc=0.8822
      LR: 0.001000
    Epoch [10/15]
      Train: Loss=0.2694, Acc=0.9004
      Val:   Loss=0.3130, Acc=0.8848
      LR: 0.001000
    Epoch [15/15]
      Train: Loss=0.2169, Acc=0.9204
      Val:   Loss=0.2799, Acc=0.8993
      LR: 0.000100

    ================================================================================
    Training Complete!
      Time: 29.99s
      Best Val Acc: 0.9000
      Test Acc: 0.8965
    ================================================================================


    ================================================================================
    EXPERIMENT COMPARISON REPORT
    ================================================================================

    Rank   Experiment                          Val Acc    Test Acc   Time(s)   
    --------------------------------------------------------------------------------
    1      mnist_medium_lr0.001                0.9825     0.9846     29.72
    2      mnist_simple_lr0.001                0.9793     0.9792     28.84
    3      fashion_mnist_medium_lr0.001        0.9000     0.8965     29.99
    4      fashion_mnist_simple_lr0.001        0.8920     0.8778     28.53

    ================================================================================
    BEST MODEL PER DATASET
    ================================================================================

    mnist:
      Best model: medium
      Test accuracy: 0.9846
      Parameters: 235,914

    fashion_mnist:
      Best model: medium
      Test accuracy: 0.8965
      Parameters: 235,914


    ================================================================================
    SUMMARY - What You Accomplished
    ================================================================================

    ✅ 여러 데이터셋을 받쳐 주는 데이터셋 관리자를 지었다
    ✅ 여러 구조를 위한 모델 공장을 만들었다
    ✅ 온전히 갖춘 학습 흐름을 짰다
    ✅ 자세한 자로 실험을 좇았다
    ✅ 여러 모델과 데이터셋을 짜임새 있게 견주었다
    ✅ 성능 견줌 알림을 만들었다

    과제 구조:
    ------------------
    1. DatasetManager
       - 여러 데이터셋을 위한 하나로 된 창구
       - 학습/검증/시험 저절로 나누기
       - 배치 크기와 미리 다듬기를 맞출 수 있다

    2. ModelFactory
       - 단순, 보통, 깊은 구조
       - 파라메터를 두루 맞출 수 있다
       - 새 모델으로 넓히기 쉽다

    3. Trainer
       - 온전한 학습 루프
       - 러닝레이트 짜기
       - 조기 종료
       - 가장 좋은 모델 좇기
       - 두루 갖춘 자취

    4. ExperimentRunner
       - 여러 실험 돌리기
       - 저절로 견주기
       - 알림 만들기
       - 결과 좇기

    핵심 학습:
    --------------
    • 조각으로 나눈 설계는 실험을 쉽게 한다
    • 짜임새 있는 견줌이 가장 좋은 길을 드러낸다
    • 실험 좇기는 다시 해내기에 매우 종요롭다
    • 데이터셋이 다르면 구조도 달라야 한다
    • 초파라메터는 성능에 크게 미친다

    익은 이의 조언:
    ------------------
    1. 늘 실험을 짜임새 있게 좇아라
    2. 모델을 고를 때는 검증 배치를 써라
    3. 따로 떼어 둔 시험 배치의 결과를 알려라
    4. 모델을 고르게 견주어라(같은 데이터, 장치, 씨앗)
    5. 모든 초파라메터와 맞춤을 적어 두어라

    다음 스텝:
    -----------
    → 데이터셋을 더 넓힌다(CIFAR-100, ImageNet, 맞춤)
    → 컨볼루션 구조를 더한다
    → 더 앞선 기법을 짠다(믹스업, 컷아웃)
    → 그림 보기와 텐서보드 적기를 만든다
    → 가장 좋은 모델을 추론에 올린다

    🎉 잘했다! 학습 클래스 모두를 마쳤다!

    이제 다음을 할 기법을 갖추었다.
    • 소프트맥스 회귀를 이론에서 실제까지 이해한다
    • 깊은 학습 분류기를 짓고 익힌다
    • 앞선 기법와 가장 좋은 버릇을 쓴다
    • 짜임새 있는 실험과 견줌을 돌린다
    • 참으로 굴릴 수 있는 학습 흐름을 만든다

    배우고 실험하기를 이어 가라! 🚀

    ```
## 2. 논의

### 구성요소의 역할

`DatasetManager`는 데이터와 로더를 준비하고 `ModelFactory`는 입력 모양과 클래스 수에 맞는 모델을 만든다. `Trainer`는 학습·검증·테스트를 담당하며 `ExperimentRunner`는 여러 설정의 결과를 모은다. 관리용 클래스가 모두 `nn.Module`이거나 `forward()`로 계산 그래프를 만드는 것은 아니다.

### 비교 단위와 평가

데이터셋이 다르면 클래스의 난이도와 입력 해상도도 달라진다. 하나의 정확도 표로 알고리즘의 보편적인 우열을 정하지 않는다. 같은 데이터에서 모델을 비교할 때는 초기값, 배치 순서, 러닝레이트와 예산을 기록하고 여러 실행의 변동도 확인한다. 검증셋은 모델 선택에, 테스트셋은 최종 평가에 사용한다.

### 최고 모델과 스케줄러 주의점

이 코드의 최고 상태 저장도 `state_dict().copy()`를 사용하므로 텐서가 독립적인 과거 값으로 보존되는지 보장되지 않는다. 마지막에 `load_state_dict()`를 호출했다는 사실만으로 가장 좋은 검증 시점을 복원했다고 판단할 수 없다. 또한 인수 없이 호출하는 스케줄러와 검증 지표를 요구하는 스케줄러는 인터페이스가 다르므로 모든 종류를 그대로 교체할 수는 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
DatasetManager의 역할과 실제 학습 모델의 파라미터 수를 구분하라.

</div>

??? success "풀이"
    데이터 관리 객체 자체의 표본 수나 설정은 학습 웨이트가 아니다. 생성된 nn.Module에 대해 파라미터를 합산하고 동결 여부도 확인한다. 모델 종류와 입력·출력 모양이 달라지면 같은 데이터 관리자라도 학습 파라미터 수는 달라진다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
옵티마이저를 Adam으로 바꾸고(`torch.optim.Adam`에 `lr=0.001`을 쓴다) 원래 옵티마이저와 학습 수렴을 비교하라. 두 로스 곡선을 같은 그래프에 그려라.

</div>

??? success "풀이"
    옵티마이저를 만드는 줄을 `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`으로 바꾼다. Adam은 경사와 경사 제곱의 이동 평균으로 성분별 업데이트를 조정한다. 어느 옵티마이저가 더 빨리 수렴하는지는 문제와 러닝레이트에 달려 있다. 같은 초기 웨이트와 데이터 순서, 평가 기준을 사용하고 각 러닝레이트를 조정해 비교한다. 여러 시드의 변동과 검증 성능도 확인한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
여러 실험 결과를 하나의 표로 비교할 때 필요한 조건과 체크포인트의 한계를 설명하라.

</div>

??? success "풀이"
    같은 평가 분할과 지표 정의, 데이터 전처리와 학습 예산을 확인하고 시드 변동을 기록한다. 제공된 얕은 복사 체크포인트가 과거 웨이트를 보존하는지 점검해야 하며, 테스트셋을 선택에 재사용하면 독립적인 성능 평가가 아니다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
제시된 ModuleList 확장 패턴을 현재 코드에 적용하려면 어떤 새 모델 정의가 필요한지 설명하라. NumPy 구현 또는 데이터 관리자와 학습 모델을 구분하라.

</div>

??? success "풀이"
    아래 코드는 일반적인 PyTorch 계층 구성 패턴의 일부이며 원래 클래스의 완성된 대체 구현은 아니다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    PyTorch 모델 클래스를 별도로 정의하고 생성자·forward·출력 모양을 함께 맞춰야 한다. NumPy 클래스나 DatasetManager에 ModuleList를 붙인다고 자동으로 모델이 바뀌지 않는다. 비선형 은닉 레이어를 추가하면 원래 단일 선형 소프트맥스 회귀와 다른 가설 공간이 된다.

## 정리하며

실험 관리 구조는 비교 조건을 통일하는 도구다. 관리자와 모델의 역할을 구분하고 데이터 분할, 체크포인트와 스케줄러 호출의 실제 동작을 확인한다.
