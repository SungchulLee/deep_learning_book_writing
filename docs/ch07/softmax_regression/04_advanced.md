# 심화

NumPy 소프트맥스 회귀와 비선형 PyTorch 분류기를 비교하면서 스케줄링, 조기 종료, 경사 클리핑과 레이블 스무딩을 살펴본다. 각각은 서로 다른 목적의 도구이므로 한 가지 기법으로 묶어 해석하지 않는다.

## 1. 코드

??? note "코드 (790줄)"

    ```python
    """
    ===============================================================================
    4단계: 앞선 소프트맥스 회귀 기법
    ===============================================================================
    어려움: 가운데~앞선
    미리 알아 둘 것: 1, 2, 3단계
    학습 목표:
      - 소프트맥스 회귀를 밑바닥부터 짠다(넘파이)
      - 앞선 정칙화 기법(L2, 드롭아웃, 배치 정규화)
      - 러닝레이트 짜기
      - 조기 종료
      - 그래디언트 절단
      - 맞춤 로스 함수와 자

    소요 시간: 60~90분
    ===============================================================================
    """

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    import time

    # 난수 씨앗을 설정한다
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 80)
    print("LEVEL 4: ADVANCED SOFTMAX REGRESSION TECHNIQUES")
    print("=" * 80)


    # =============================================================================
    # 1부: 바닥부터 만드는 소프트맥스 회귀 (NumPy)
    # =============================================================================
    print("\n" + "=" * 80)
    print("PART 1: Implementing Softmax Regression from Scratch")
    print("=" * 80)

    class SoftmaxRegressionNumPy:
        """
        넘파이만으로 밑바닥부터 짠 소프트맥스 회귀.
        PyTorch가 속에서 무엇을 하는지 이해하는 데 도움이 된다.
        """

        def __init__(self, input_dim, num_classes, lr=0.01, reg_lambda=0.01):
            """
            소프트맥스 회귀 모델의 초기화한다.

            Args:
                input_dim: 입력 피처의 수
                num_classes: 출력 클래스의 수
                lr: 러닝레이트
                reg_lambda: L2 정칙화 파라메터
            """
            # 웨이트와 바이어스를 작은 무작위 값으로 초기화한다
            self.W = np.random.randn(input_dim, num_classes) * 0.01
            self.b = np.zeros((1, num_classes))
            self.lr = lr
            self.reg_lambda = reg_lambda
            self.loss_history = []

        def softmax(self, z):
            """
            z의 줄마다 소프트맥스 값을 계산한다.

            Args:
                z: 모양이 (batch_size, num_classes)인 로짓

            Returns:
                모양이 (batch_size, num_classes)인 확률
            """
            # 수치적 안정성을 위해 최댓값을 뺀다
            z_shifted = z - np.max(z, axis=1, keepdims=True)
            exp_z = np.exp(z_shifted)
            return exp_z / np.sum(exp_z, axis=1, keepdims=True)

        def forward(self, X):
            """
            순전파: 클래스 확률을 계산한다.

            Args:
                X: 모양이 (batch_size, input_dim)인 입력 피처

            Returns:
                모양이 (batch_size, num_classes)인 확률
            """
            logits = np.dot(X, self.W) + self.b
            probs = self.softmax(logits)
            return probs

        def compute_loss(self, X, y):
            """
            L2 정칙화를 곁들인 교차 엔트로피 로스를 계산한다.

            Args:
                X: 입력 피처
                y: 참 레이블(클래스 번호)

            Returns:
                평균 로스 값
            """
            m = X.shape[0]  # Number of samples
            probs = self.forward(X)

            # 교차 엔트로피 로스
            # 표본마다 참 클래스의 확률을 얻는다
            correct_logprobs = -np.log(probs[range(m), y] + 1e-10)
            data_loss = np.sum(correct_logprobs) / m

            # L2 정칙화 로스
            reg_loss = 0.5 * self.reg_lambda * np.sum(self.W * self.W)

            total_loss = data_loss + reg_loss
            return total_loss

        def backward(self, X, y):
            """
            역전파: 그래디언트를 계산한다.

            Args:
                X: 입력 피처
                y: 참 레이블
            """
            m = X.shape[0]
            probs = self.forward(X)

            # 로짓에 대한 로스의 경사를 계산한다
            # 핵심 통찰이다: d(loss)/d(logits) = (probs - one_hot_labels) / m
            dlogits = probs.copy()
            dlogits[range(m), y] -= 1  # Subtract 1 from true class probabilities
            dlogits /= m

            # 웨이트와 바이어스에 대한 경사를 계산한다
            dW = np.dot(X.T, dlogits) + self.reg_lambda * self.W  # Add L2 gradient
            db = np.sum(dlogits, axis=0, keepdims=True)

            return dW, db

        def train_step(self, X, y):
            """
            경사 하강법을 한 스텝 한다.

            Args:
                X: 학습 피처
                y: 학습 레이블
            """
            # 경사를 계산한다
            dW, db = self.backward(X, y)

            # 웨이트를 갱신한다
            self.W -= self.lr * dW
            self.b -= self.lr * db

            # 로스를 계산하여 저장한다
            loss = self.compute_loss(X, y)
            self.loss_history.append(loss)

            return loss

        def predict(self, X):
            """
            클래스 레이블을 예측한다.

            Args:
                X: 입력 피처

            Returns:
                예측한 클래스 번호
            """
            probs = self.forward(X)
            return np.argmax(probs, axis=1)

        def accuracy(self, X, y):
            """
            정확도를 계산한다.

            Args:
                X: 입력 피처
                y: 참 레이블

            Returns:
                정확도 값
            """
            predictions = self.predict(X)
            return np.mean(predictions == y)


    # 시험용 합성 데이터를 생성한다
    print("Generating synthetic dataset...")
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                              n_redundant=5, n_classes=3, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Classes: {len(np.unique(y))}")

    # NumPy 모델을 학습시킨다
    print("\nTraining NumPy implementation...")
    model_numpy = SoftmaxRegressionNumPy(
        input_dim=X_train.shape[1],
        num_classes=3,
        lr=0.1,
        reg_lambda=0.01
    )

    num_epochs = 200
    for epoch in range(num_epochs):
        loss = model_numpy.train_step(X_train, y_train)

        if (epoch + 1) % 50 == 0:
            train_acc = model_numpy.accuracy(X_train, y_train)
            test_acc = model_numpy.accuracy(X_test, y_test)
            print(f"Epoch {epoch+1}: Loss = {loss:.4f}, "
                  f"Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}")

    final_test_acc = model_numpy.accuracy(X_test, y_test)
    print(f"\n✅ NumPy implementation final test accuracy: {final_test_acc:.2%}")


    # =============================================================================
    # 2부: 배치 정규화를 넣은 고급 모델
    # =============================================================================
    print("\n" + "=" * 80)
    print("PART 2: Batch Normalization")
    print("=" * 80)

    """
    배치 정규화:
    -------------------
    - 레이어마다 입력의 정규화한다
    - 학습을 든든하게 한다
    - 러닝레이트를 더 크게 쓸 수 있다
    - 정칙화 노릇을 한다
    - 안쪽 함께 바뀜의 옮겨감을 줄인다
    """

    class AdvancedClassifier(nn.Module):
        """
        배치 정규화를 곁들인 뉴럴 넷트웍.

        배치 정규화는 레이어의 입력을 맞추어 다음을 이룬다.
        1. 학습을 빠르게 한다
        2. 초기화에 덜 흔들린다
        3. 정칙화 노릇을 한다
        """

        def __init__(self, input_size, hidden_size1, hidden_size2, num_classes,
                     use_batchnorm=True, dropout_rate=0.3):
            super(AdvancedClassifier, self).__init__()

            self.use_batchnorm = use_batchnorm

            # 첫 번째 레이어
            self.fc1 = nn.Linear(input_size, hidden_size1)
            self.bn1 = nn.BatchNorm1d(hidden_size1) if use_batchnorm else nn.Identity()
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(dropout_rate)

            # 두 번째 레이어
            self.fc2 = nn.Linear(hidden_size1, hidden_size2)
            self.bn2 = nn.BatchNorm1d(hidden_size2) if use_batchnorm else nn.Identity()
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(dropout_rate)

            # 출력 레이어
            self.fc3 = nn.Linear(hidden_size2, num_classes)

        def forward(self, x):
            x = self.fc1(x)
            x = self.bn1(x)      # Batch normalization here
            x = self.relu1(x)
            x = self.dropout1(x)

            x = self.fc2(x)
            x = self.bn2(x)      # Batch normalization here
            x = self.relu2(x)
            x = self.dropout2(x)

            x = self.fc3(x)
            return x


    # 데이터셋들을 만든다
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 배치 정규화가 있는 모델과 없는 모델을 비교한다
    print("\nComparing models with/without batch normalization...")

    models_config = [
        ("Without BatchNorm", False),
        ("With BatchNorm", True)
    ]

    for name, use_bn in models_config:
        print(f"\nTraining: {name}")
        print("-" * 40)

        model = AdvancedClassifier(
            input_size=20,
            hidden_size1=64,
            hidden_size2=32,
            num_classes=3,
            use_batchnorm=use_bn,
            dropout_rate=0.3
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 빠른 학습
        num_epochs = 50
        for epoch in range(num_epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

        # 평가한다
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_t)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y_test_t).float().mean()

        print(f"Final test accuracy: {accuracy:.4f}")


    # =============================================================================
    # 3부: 러닝레이트 일정 조절
    # =============================================================================
    print("\n" + "=" * 80)
    print("PART 3: Learning Rate Scheduling")
    print("=" * 80)

    """
    러닝레이트 짜기:
    ------------------------
    붙박인 러닝레이트를 쓰는 대신 다음을 할 수 있다.
    - 처음에 빠르게 배우도록 큰 러닝레이트로 비롯한다
    - 곱게 다듬으려고 차츰 줄인다
    - 여러 전략을 쓴다: 계단 잦아들기, 지수, 코사인 따위
    """

    class LRSchedulerDemo:
        """여러 러닝레이트 일정을 보여준다."""

        @staticmethod
        def step_lr_schedule(initial_lr, epoch, step_size=30, gamma=0.1):
            """
            계단 잦아들기: step_size 에폭마다 러닝레이트를 gamma배로 줄인다.

            보기: 러닝레이트 = 0.1 → 0.01 → 0.001(gamma=0.1, step_size=30이면)
            """
            return initial_lr * (gamma ** (epoch // step_size))

        @staticmethod
        def exponential_schedule(initial_lr, epoch, gamma=0.95):
            """
            지수 잦아들기: 러닝레이트 = initial_lr * gamma^epoch

            매끄럽고 이어지게 잦아든다.
            """
            return initial_lr * (gamma ** epoch)

        @staticmethod
        def cosine_schedule(initial_lr, epoch, total_epochs):
            """
            코사인 달구기: 러닝레이트가 코사인 굽이를 따른다.

            크게 비롯해 매끄럽게 줄어든다. 요즘 학습에서 널리 쓴다.
            """
            return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))


    # 여러 일정을 시각화한다
    print("Visualizing learning rate schedules...")
    epochs = range(100)
    initial_lr = 0.1

    schedules = {
        'Constant': [initial_lr] * 100,
        'Step Decay': [LRSchedulerDemo.step_lr_schedule(initial_lr, e) for e in epochs],
        'Exponential': [LRSchedulerDemo.exponential_schedule(initial_lr, e) for e in epochs],
        'Cosine': [LRSchedulerDemo.cosine_schedule(initial_lr, e, 100) for e in epochs]
    }

    # 그리려면 주석을 푼다
    # plt.figure(figsize=(10, 6))
    # for name, lrs in schedules.items():
    #     plt.plot(epochs, lrs, label=name, linewidth=2)
    # plt.xlabel('Epoch')
    # plt.ylabel('Learning Rate')
    # plt.title('Learning Rate Schedules')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.show()


    # 러닝레이트 일정을 적용해 학습한다
    print("\nTraining with StepLR scheduler...")
    model_scheduled = AdvancedClassifier(20, 64, 32, 3, use_batchnorm=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_scheduled.parameters(), lr=0.01)

    # 스케줄러를 만든다: 20 에폭마다 러닝레이트에 0.1을 곱한다
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    num_epochs = 60
    lr_history = []

    for epoch in range(num_epochs):
        model_scheduled.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model_scheduled(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        # 스케줄러를 한 단계 진행시킨다
        scheduler.step()

        # 러닝레이트를 추적한다
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: Learning Rate = {current_lr:.6f}")

    print("✅ Training with LR scheduling complete")


    # =============================================================================
    # 4부: 조기 종료
    # =============================================================================
    print("\n" + "=" * 80)
    print("PART 4: Early Stopping")
    print("=" * 80)

    """
    조기 종료:
    --------------
    검증 로스가 더 나아지지 않으면 학습을 멈춘다.
    과적합을 막고 때를 아낀다.
    """

    class EarlyStopping:
        """
        검증 로스가 더 나아지지 않으면 학습을 멈추는 조기 종료.
        """

        def __init__(self, patience=5, min_delta=0.001, verbose=True):
            """
            Args:
                patience: 멈추기 앞에 몇 판을 기다릴지
                min_delta: 나아졌다고 볼 가장 작은 바뀜
                verbose: 알림을 찍는다
            """
            self.patience = patience
            self.min_delta = min_delta
            self.verbose = verbose
            self.counter = 0
            self.best_loss = None
            self.early_stop = False
            self.best_model = None

        def __call__(self, val_loss, model):
            """
            학습을 멈춰야 할지 살핀다.

            Args:
                val_loss: 이제 검증 로스
                model: 이제 모델(가장 좋은 판을 저장하려고)

            Returns:
                멈춰야 하면 True, 아니면 False
            """
            if self.best_loss is None:
                self.best_loss = val_loss
                self.best_model = model.state_dict().copy()
            elif val_loss > self.best_loss - self.min_delta:
                self.counter += 1
                if self.verbose:
                    print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                if self.verbose:
                    print(f"  Validation loss improved: {self.best_loss:.4f} → {val_loss:.4f}")
                self.best_loss = val_loss
                self.best_model = model.state_dict().copy()
                self.counter = 0

            return self.early_stop


    # 조기 종료를 보여준다
    print("Training with early stopping...")

    # 데이터를 학습과 검증으로 나눈다
    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    train_loader_es = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader_es = DataLoader(val_subset, batch_size=32, shuffle=False)

    model_es = AdvancedClassifier(20, 64, 32, 3, use_batchnorm=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_es.parameters(), lr=0.001)

    early_stopping = EarlyStopping(patience=10, min_delta=0.001, verbose=True)

    max_epochs = 100
    for epoch in range(max_epochs):
        # 학습
        model_es.train()
        train_loss = 0
        for X_batch, y_batch in train_loader_es:
            optimizer.zero_grad()
            outputs = model_es(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 검증
        model_es.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader_es:
                outputs = model_es(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader_es)
        avg_val_loss = val_loss / len(val_loader_es)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, "
              f"Val Loss = {avg_val_loss:.4f}")

        # 조기 종료 여부를 확인한다
        if early_stopping(avg_val_loss, model_es):
            print(f"\n✅ Early stopping triggered at epoch {epoch+1}")
            # 가장 좋은 모델을 불러온다
            model_es.load_state_dict(early_stopping.best_model)
            break

    if epoch == max_epochs - 1:
        print(f"\n✅ Training completed all {max_epochs} epochs")


    # =============================================================================
    # 5부: 경사 자르기
    # =============================================================================
    print("\n" + "=" * 80)
    print("PART 5: Gradient Clipping")
    print("=" * 80)

    """
    그래디언트 절단:
    -----------------
    그래디언트의 크기를 제한하어 터지는 것을 막는다.
    학습을 든든하게 하며 특히 되도는 뉴럴 넷트웍과 LSTM에 쓸모 있다.
    """

    def train_with_gradient_clipping(model, train_loader, criterion, optimizer,
                                    max_norm=1.0, num_epochs=10):
        """
        그래디언트 절단을 곁들여 모델을 익힌다.

        Args:
            max_norm: 그래디언트의 가장 큰 노름
        """
        print(f"Training with gradient clipping (max_norm={max_norm})...")

        for epoch in range(num_epochs):
            model.train()
            total_grad_norm = 0
            num_batches = 0

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()

                # 경사를 자른다
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                total_grad_norm += grad_norm
                num_batches += 1

                optimizer.step()

            avg_grad_norm = total_grad_norm / num_batches
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}: Avg gradient norm = {avg_grad_norm:.4f}")

        print("✅ Training with gradient clipping complete")


    model_clip = AdvancedClassifier(20, 64, 32, 3, use_batchnorm=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_clip.parameters(), lr=0.001)

    train_with_gradient_clipping(model_clip, train_loader, criterion, optimizer,
                                max_norm=1.0, num_epochs=20)


    # =============================================================================
    # 6부: 사용자 정의 로스 함수 - 레이블 평활화
    # =============================================================================
    print("\n" + "=" * 80)
    print("PART 6: Label Smoothing")
    print("=" * 80)

    """
    레이블 스무딩:
    ---------------
    딱딱한 타깃(0이나 1) 대신 부드러운 타깃을 쓴다.
    보기: [0, 1, 0] 대신 [0.05, 0.9, 0.05]을 쓴다

    Benefits:
    - 지나친 자신을 막는다
    - 정칙화 노릇을 한다
    - 일반화이 나아지는 일이 잦다
    """

    class LabelSmoothingCrossEntropy(nn.Module):
        """
        레이블 스무딩을 곁들인 교차 엔트로피 로스.
        """

        def __init__(self, num_classes, smoothing=0.1):
            """
            Args:
                num_classes: 클래스의 수
                smoothing: 매끄럽게 하기 파라메터(0이면 안 함, 1이면 고르게)
            """
            super().__init__()
            self.num_classes = num_classes
            self.smoothing = smoothing
            self.confidence = 1.0 - smoothing

        def forward(self, predictions, targets):
            """
            레이블 스무딩 로스를 계산한다.

            Args:
                predictions: 모델 로짓 (batch_size, num_classes)
                targets: 참 클래스 번호 (batch_size,)
            """
            # 로그 소프트맥스를 적용한다
            log_probs = torch.nn.functional.log_softmax(predictions, dim=1)

            # 평활화된 목표를 만든다
            with torch.no_grad():
                # 균등분포로 시작한다
                smooth_targets = torch.zeros_like(log_probs)
                smooth_targets.fill_(self.smoothing / (self.num_classes - 1))
                # 참 클래스에 더 높은 확률을 준다
                smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)

            # 로스를 계산한다
            loss = torch.sum(-smooth_targets * log_probs, dim=1)
            return loss.mean()


    # 표준 교차 엔트로피와 레이블 평활화를 비교한다
    print("Comparing standard CrossEntropy vs Label Smoothing...")

    configs = [
        ("Standard CrossEntropy", nn.CrossEntropyLoss(), 0.0),
        ("Label Smoothing (0.1)", LabelSmoothingCrossEntropy(3, smoothing=0.1), 0.1)
    ]

    for name, criterion, smoothing in configs:
        print(f"\n{name}:")
        print("-" * 40)

        model = AdvancedClassifier(20, 64, 32, 3, use_batchnorm=True)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 몇 에폭 동안 학습한다
        for epoch in range(30):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

        # 평가한다
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_t)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y_test_t).float().mean()

        print(f"Test accuracy: {accuracy:.4f}")


    # =============================================================================
    # 요약
    # =============================================================================
    print("\n" + "=" * 80)
    print("SUMMARY - What You Learned")
    print("=" * 80)

    print("""
    ✅ 넘파이로 소프트맥스 회귀를 밑바닥부터 짰다
    ✅ 학습을 든든하게 하려고 배치 정규화를 썼다
    ✅ 러닝레이트 짜기 전략을 걸었다
    ✅ 과적합을 막으려고 조기 종료를 짰다
    ✅ 학습을 든든하게 하려고 그래디언트 절단을 썼다
    ✅ 정칙화 기법로 레이블 스무딩을 살펴보았다

    앞선 기법 간추림:
    ---------------------------
    1. 배치 정규화
       - 레이어의 입력을 맞춘다
       - 학습을 빠르게 한다
       - 정칙화 노릇을 한다

    2. 러닝레이트 짜기
       - 계단 잦아들기: 사이를 두고 러닝레이트를 떨어뜨린다
       - 지수: 매끄럽게 이어지는 잦아들기
       - 코사인: 코사인 굽이를 따른다

    3. 조기 종료
       - 검증 로스를 지켜보아라
       - 나아짐이 없으면 멈춘다
       - 가장 좋은 모델을 저장한다

    4. 그래디언트 절단
       - 그래디언트가 터지는 것을 막는다
       - 그래디언트의 크기를 옭아맨다
       - 든든함을 높인다

    5. 레이블 스무딩
       - 딱딱한 타깃 대신 부드러운 타깃
       - 지나친 자신을 막는다
       - 일반화이 더 낫다

    가장 좋은 버릇:
    --------------
    • 더 깊은 망에는 배치 정규화를 쓴다
    • 러닝레이트를 크게 비롯해 차츰 낮춘다
    • 검증 배치와 함께 늘 조기 종료를 쓴다
    • 되도는 뉴럴 넷트웍이나 흔들리는 학습에서는 그래디언트를 자른다
    • 일반화를 높이려면 레이블 스무딩을 살펴본다

    다음 스텝:
    -----------
    → 5단계: 여러 데이터셋과 구조를 견주기
    → 여러 기법을 뒤섞어 실험해 보기
    → 제 데이터셋에 써 보기

    🎉 잘했다! 앞선 기법을 익혔다!
    """)


    if __name__ == "__main__":
        pass
    ```


??? note "전체 출력 (243줄)"

    ```
    ================================================================================
    LEVEL 4: ADVANCED SOFTMAX REGRESSION TECHNIQUES
    ================================================================================

    ================================================================================
    PART 1: Implementing Softmax Regression from Scratch
    ================================================================================
    Generating synthetic dataset...
    Training samples: 800
    Test samples: 200
    Features: 20
    Classes: 3

    Training NumPy implementation...
    Epoch 50: Loss = 0.7205, Train Acc = 0.7125, Test Acc = 0.6650
    Epoch 100: Loss = 0.7185, Train Acc = 0.7113, Test Acc = 0.6550
    Epoch 150: Loss = 0.7181, Train Acc = 0.7087, Test Acc = 0.6550
    Epoch 200: Loss = 0.7179, Train Acc = 0.7087, Test Acc = 0.6600

    ✅ NumPy implementation final test accuracy: 66.00%

    ================================================================================
    PART 2: Batch Normalization
    ================================================================================

    Comparing models with/without batch normalization...

    Training: Without BatchNorm
    ----------------------------------------
    Final test accuracy: 0.8650

    Training: With BatchNorm
    ----------------------------------------
    Final test accuracy: 0.8800

    ================================================================================
    PART 3: Learning Rate Scheduling
    ================================================================================
    Visualizing learning rate schedules...

    Training with StepLR scheduler...
    Epoch 20: Learning Rate = 0.001000
    Epoch 40: Learning Rate = 0.000100
    Epoch 60: Learning Rate = 0.000010
    ✅ Training with LR scheduling complete

    ================================================================================
    PART 4: Early Stopping
    ================================================================================
    Training with early stopping...
    Epoch 1: Train Loss = 1.1228, Val Loss = 1.0128
    Epoch 2: Train Loss = 1.0015, Val Loss = 0.9222
      Validation loss improved: 1.0128 → 0.9222
    Epoch 3: Train Loss = 0.9420, Val Loss = 0.8578
      Validation loss improved: 0.9222 → 0.8578
    Epoch 4: Train Loss = 0.8788, Val Loss = 0.8164
      Validation loss improved: 0.8578 → 0.8164
    Epoch 5: Train Loss = 0.8527, Val Loss = 0.7746
      Validation loss improved: 0.8164 → 0.7746
    Epoch 6: Train Loss = 0.8028, Val Loss = 0.7472
      Validation loss improved: 0.7746 → 0.7472
    Epoch 7: Train Loss = 0.7777, Val Loss = 0.7149
      Validation loss improved: 0.7472 → 0.7149
    Epoch 8: Train Loss = 0.7812, Val Loss = 0.6939
      Validation loss improved: 0.7149 → 0.6939
    Epoch 9: Train Loss = 0.7302, Val Loss = 0.6690
      Validation loss improved: 0.6939 → 0.6690
    Epoch 10: Train Loss = 0.7394, Val Loss = 0.6502
      Validation loss improved: 0.6690 → 0.6502
    Epoch 11: Train Loss = 0.6925, Val Loss = 0.6257
      Validation loss improved: 0.6502 → 0.6257
    Epoch 12: Train Loss = 0.6588, Val Loss = 0.6027
      Validation loss improved: 0.6257 → 0.6027
    Epoch 13: Train Loss = 0.6741, Val Loss = 0.5875
      Validation loss improved: 0.6027 → 0.5875
    Epoch 14: Train Loss = 0.6474, Val Loss = 0.5770
      Validation loss improved: 0.5875 → 0.5770
    Epoch 15: Train Loss = 0.6280, Val Loss = 0.5573
      Validation loss improved: 0.5770 → 0.5573
    Epoch 16: Train Loss = 0.5973, Val Loss = 0.5414
      Validation loss improved: 0.5573 → 0.5414
    Epoch 17: Train Loss = 0.5824, Val Loss = 0.5389
      Validation loss improved: 0.5414 → 0.5389
    Epoch 18: Train Loss = 0.5838, Val Loss = 0.5233
      Validation loss improved: 0.5389 → 0.5233
    Epoch 19: Train Loss = 0.5735, Val Loss = 0.5114
      Validation loss improved: 0.5233 → 0.5114
    Epoch 20: Train Loss = 0.5445, Val Loss = 0.4930
      Validation loss improved: 0.5114 → 0.4930
    Epoch 21: Train Loss = 0.5755, Val Loss = 0.4886
      Validation loss improved: 0.4930 → 0.4886
    Epoch 22: Train Loss = 0.5454, Val Loss = 0.4876
      Validation loss improved: 0.4886 → 0.4876
    Epoch 23: Train Loss = 0.5165, Val Loss = 0.4830
      Validation loss improved: 0.4876 → 0.4830
    Epoch 24: Train Loss = 0.5055, Val Loss = 0.4734
      Validation loss improved: 0.4830 → 0.4734
    Epoch 25: Train Loss = 0.4835, Val Loss = 0.4698
      Validation loss improved: 0.4734 → 0.4698
    Epoch 26: Train Loss = 0.4946, Val Loss = 0.4502
      Validation loss improved: 0.4698 → 0.4502
    Epoch 27: Train Loss = 0.4959, Val Loss = 0.4443
      Validation loss improved: 0.4502 → 0.4443
    Epoch 28: Train Loss = 0.4719, Val Loss = 0.4408
      Validation loss improved: 0.4443 → 0.4408
    Epoch 29: Train Loss = 0.5177, Val Loss = 0.4312
      Validation loss improved: 0.4408 → 0.4312
    Epoch 30: Train Loss = 0.4732, Val Loss = 0.4200
      Validation loss improved: 0.4312 → 0.4200
    Epoch 31: Train Loss = 0.4607, Val Loss = 0.4247
      EarlyStopping counter: 1/10
    Epoch 32: Train Loss = 0.4736, Val Loss = 0.4171
      Validation loss improved: 0.4200 → 0.4171
    Epoch 33: Train Loss = 0.4456, Val Loss = 0.4203
      EarlyStopping counter: 1/10
    Epoch 34: Train Loss = 0.4490, Val Loss = 0.4088
      Validation loss improved: 0.4171 → 0.4088
    Epoch 35: Train Loss = 0.4241, Val Loss = 0.3966
      Validation loss improved: 0.4088 → 0.3966
    Epoch 36: Train Loss = 0.4233, Val Loss = 0.3902
      Validation loss improved: 0.3966 → 0.3902
    Epoch 37: Train Loss = 0.4037, Val Loss = 0.3777
      Validation loss improved: 0.3902 → 0.3777
    Epoch 38: Train Loss = 0.4656, Val Loss = 0.3676
      Validation loss improved: 0.3777 → 0.3676
    Epoch 39: Train Loss = 0.4031, Val Loss = 0.3719
      EarlyStopping counter: 1/10
    Epoch 40: Train Loss = 0.4229, Val Loss = 0.3701
      EarlyStopping counter: 2/10
    Epoch 41: Train Loss = 0.3966, Val Loss = 0.3792
      EarlyStopping counter: 3/10
    Epoch 42: Train Loss = 0.4098, Val Loss = 0.3713
      EarlyStopping counter: 4/10
    Epoch 43: Train Loss = 0.3976, Val Loss = 0.3554
      Validation loss improved: 0.3676 → 0.3554
    Epoch 44: Train Loss = 0.4156, Val Loss = 0.3478
      Validation loss improved: 0.3554 → 0.3478
    Epoch 45: Train Loss = 0.3831, Val Loss = 0.3456
      Validation loss improved: 0.3478 → 0.3456
    Epoch 46: Train Loss = 0.3563, Val Loss = 0.3408
      Validation loss improved: 0.3456 → 0.3408
    Epoch 47: Train Loss = 0.3967, Val Loss = 0.3317
      Validation loss improved: 0.3408 → 0.3317
    Epoch 48: Train Loss = 0.3757, Val Loss = 0.3447
      EarlyStopping counter: 1/10
    Epoch 49: Train Loss = 0.3973, Val Loss = 0.3419
      EarlyStopping counter: 2/10
    Epoch 50: Train Loss = 0.3442, Val Loss = 0.3343
      EarlyStopping counter: 3/10
    Epoch 51: Train Loss = 0.3917, Val Loss = 0.3449
      EarlyStopping counter: 4/10
    Epoch 52: Train Loss = 0.3618, Val Loss = 0.3467
      EarlyStopping counter: 5/10
    Epoch 53: Train Loss = 0.3652, Val Loss = 0.3495
      EarlyStopping counter: 6/10
    Epoch 54: Train Loss = 0.3646, Val Loss = 0.3325
      EarlyStopping counter: 7/10
    Epoch 55: Train Loss = 0.3483, Val Loss = 0.3351
      EarlyStopping counter: 8/10
    Epoch 56: Train Loss = 0.3356, Val Loss = 0.3487
      EarlyStopping counter: 9/10
    Epoch 57: Train Loss = 0.3611, Val Loss = 0.3313
      EarlyStopping counter: 10/10

    ✅ Early stopping triggered at epoch 57

    ================================================================================
    PART 5: Gradient Clipping
    ================================================================================
    Training with gradient clipping (max_norm=1.0)...
    Epoch 5: Avg gradient norm = 1.0583
    Epoch 10: Avg gradient norm = 1.1695
    Epoch 15: Avg gradient norm = 1.3154
    Epoch 20: Avg gradient norm = 1.5507
    ✅ Training with gradient clipping complete

    ================================================================================
    PART 6: Label Smoothing
    ================================================================================
    Comparing standard CrossEntropy vs Label Smoothing...

    Standard CrossEntropy:
    ----------------------------------------
    Test accuracy: 0.8750

    Label Smoothing (0.1):
    ----------------------------------------
    Test accuracy: 0.8250

    ================================================================================
    SUMMARY - What You Learned
    ================================================================================

    ✅ 넘파이로 소프트맥스 회귀를 밑바닥부터 짰다
    ✅ 학습을 든든하게 하려고 배치 정규화를 썼다
    ✅ 러닝레이트 짜기 전략을 걸었다
    ✅ 과적합을 막으려고 조기 종료를 짰다
    ✅ 학습을 든든하게 하려고 그래디언트 절단을 썼다
    ✅ 정칙화 기법로 레이블 스무딩을 살펴보았다

    앞선 기법 간추림:
    ---------------------------
    1. 배치 정규화
       - 레이어의 입력을 맞춘다
       - 학습을 빠르게 한다
       - 정칙화 노릇을 한다

    2. 러닝레이트 짜기
       - 계단 잦아들기: 사이를 두고 러닝레이트를 떨어뜨린다
       - 지수: 매끄럽게 이어지는 잦아들기
       - 코사인: 코사인 굽이를 따른다

    3. 조기 종료
       - 검증 로스를 지켜보아라
       - 나아짐이 없으면 멈춘다
       - 가장 좋은 모델을 저장한다

    4. 그래디언트 절단
       - 그래디언트가 터지는 것을 막는다
       - 그래디언트의 크기를 옭아맨다
       - 든든함을 높인다

    5. 레이블 스무딩
       - 딱딱한 타깃 대신 부드러운 타깃
       - 지나친 자신을 막는다
       - 일반화이 더 낫다

    가장 좋은 버릇:
    --------------
    • 더 깊은 망에는 배치 정규화를 쓴다
    • 러닝레이트를 크게 비롯해 차츰 낮춘다
    • 검증 배치와 함께 늘 조기 종료를 쓴다
    • 되도는 뉴럴 넷트웍이나 흔들리는 학습에서는 그래디언트를 자른다
    • 일반화를 높이려면 레이블 스무딩을 살펴본다

    다음 스텝:
    -----------
    → 5단계: 여러 데이터셋과 구조를 견주기
    → 여러 기법을 뒤섞어 실험해 보기
    → 제 데이터셋에 써 보기

    🎉 잘했다! 앞선 기법을 익혔다!

    ```


## 2. 논의

### 수동 미분과 비선형 모델

`SoftmaxRegressionNumPy`는 웨이트와 바이어스를 NumPy 배열로 관리하며 경사를 직접 계산한다. `AdvancedClassifier`는 은닉 레이어를 가진 PyTorch 모델이다. 두 클래스는 같은 모델의 필수 부품이 아니라 서로 다른 구현과 표현력을 보여주는 예제다.

### 학습 제어의 구분

스케줄러는 러닝레이트를 시간이나 지표에 따라 바꾸고, 클리핑은 계산된 경사의 노름을 제한한다. 조기 종료는 검증 기준으로 반복을 중단하며, 레이블 스무딩은 목표 분포 자체를 바꾼다. 어느 방법도 수렴이나 일반화 개선을 무조건 보장하지 않는다.

### 체크포인트의 실제 보존

제공 코드의 `model.state_dict().copy()`는 사전의 얕은 복사다. 값 텐서의 저장 공간이 모델과 공유될 수 있으므로 이후 학습 중 “최고 모델” 값도 바뀔 수 있다. 독립 텐서 복제나 깊은 복사, 당시 파일 저장이 필요하다. 원래 코드는 유지했으며, 그 코드가 최적 검증 시점을 확실히 복원한다고 해석하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
NumPy 선형 소프트맥스 모델의 파라미터 수를 어떻게 계산하고, PyTorch의 `.parameters()`와 어떻게 구분하는가?

</div>

??? success "풀이"
    입력 $D$개, 클래스 $K$개이면 웨이트 $DK$개와 바이어스 $K$개, 총 $K(D+1)$개다. NumPy 클래스는 nn.Module이 아니므로 `.parameters()`로 자동 등록된 목록을 얻는 구조가 아니다. 배열의 모양과 원소 수로 직접 확인한다.

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
`state_dict().copy()`가 독립적인 최고 모델 스냅숏이 되지 않을 수 있는 이유를 설명하라.

</div>

??? success "풀이"
    사전의 매핑만 복사하고 값 텐서는 원래 저장 공간을 공유할 수 있다. 학습이 값을 수정하면 복사 사전이 가리키는 값도 바뀔 수 있으므로 각 텐서 복제나 깊은 복사·당시 파일 저장이 필요하다.

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

러닝레이트 조절, 클리핑, 중단과 목표 스무딩은 서로 다른 기능이다. 비교 목적과 검증 기준을 정하고 체크포인트가 실제로 독립 저장되는지 확인한다.
