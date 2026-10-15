# 고급 기법

혼합 정밀도, 경사 누적, 학습률 스케줄과 단계적 동결 해제·추론 증강을 결합한 전이 학습 예제다. 여러 기법을 켠다고 최고 성능이나 망각 방지가 보장되지는 않는다. 실제 코드에서 갱신·검증·스케줄이 어떻게 연결되는지 확인한다.

## 1. 코드

??? note "코드 (481줄)"

    ```python
    """
    예제 4: 고급 전이 학습 기법
    =================================================

    이 스크립트는 최신 전이 학습 기법을 보여 준다.
    - 섞인 정밀도 학습 (FP16)
    - 기울기 모으기
    - 고급 학습률 일정
    - 층을 차츰 녹이기
    - 모델 앙상블
    - 여러 구조

    이 기법들은 최고의 성능을 얻으려고 연구와 실무에서
    쓰인다.

    지은이: PyTorch Transfer Learning Tutorial
    날짜: 2025
    """

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch.cuda.amp import autocast, GradScaler
    import torchvision
    import torchvision.transforms as transforms
    from torchvision.models import (resnet18, resnet50,
                                    ResNet18_Weights, ResNet50_Weights)
    import time
    import copy
    import numpy as np

    # 난수 씨앗 고정
    torch.manual_seed(42)
    np.random.seed(42)

    # ============================================================================
    # 설정
    # ============================================================================
    """
    이 깃발을 켜고 끄며 여러 고급 기법을 쓰거나 끈다.
    여러 조합을 시험해 보라!
    """

    # 학습 설정
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    BASE_LR = 0.001

    # 고급 기법 깃발
    USE_MIXED_PRECISION = True        # FP16 학습을 켠다 (GPU가 필요하다)
    GRADIENT_ACCUMULATION_STEPS = 2   # 실효 배치 크기 = BATCH_SIZE × 이 값
    USE_COSINE_SCHEDULE = True        # 코사인 담금질을 쓸지 정체 감지를 쓸지
    GRADUAL_UNFREEZING = True         # 학습 중에 층을 차츰 녹인다
    USE_ENSEMBLE = False              # 앙상블을 위해 모델을 여럿 학습한다 (느리다)

    print("Advanced Transfer Learning Configuration:")
    print("=" * 70)
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Effective Batch Size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Mixed Precision: {USE_MIXED_PRECISION}")
    print(f"Gradient Accumulation: {GRADIENT_ACCUMULATION_STEPS} steps")
    print(f"Cosine Annealing Schedule: {USE_COSINE_SCHEDULE}")
    print(f"Gradual Unfreezing: {GRADUAL_UNFREEZING}")
    print(f"Model Ensemble: {USE_ENSEMBLE}")
    print("=" * 70 + "\n")

    # ============================================================================
    # 1단계: 장치 설정
    # ============================================================================

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 섞인 정밀도를 쓸 수 있는지 살핀다
    if USE_MIXED_PRECISION and not torch.cuda.is_available():
        print("Warning: Mixed precision requires CUDA. Disabling mixed precision.")
        USE_MIXED_PRECISION = False

    if USE_MIXED_PRECISION:
        print("Mixed precision (FP16) training enabled")

    # ============================================================================
    # 2단계: 고급 데이터 불리기
    # ============================================================================

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # 학습 불리기 파이프라인
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    # 시험 때 불리기 변환
    tta_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.FiveCrop(224),
        transforms.Lambda(lambda crops: torch.stack([
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(
                transforms.ToTensor()(crop)
            ) for crop in crops
        ]))
    ])

    print("\nLoading CIFAR-10 dataset with advanced augmentation...")

    # 데이터셋 불러오기
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=val_transform
    )

    # 학습 자료를 학습과 검증으로 나눈다
    from torch.utils.data import random_split
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True if torch.cuda.is_available() else False
    )

    classes = ['plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    print(f"Dataset loaded: Train={len(train_dataset)}, "
          f"Val={len(val_dataset)}, Test={len(test_dataset)}")

    # ============================================================================
    # 3단계: 구조를 골라 모델 만들기
    # ============================================================================

    def create_model(architecture='resnet18', num_classes=10, pretrained=True):
        """
        정한 구조로 전이 학습 모델을 만든다.
        """
        print(f"\nCreating {architecture} model...")

        if architecture == 'resnet18':
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            model = resnet18(weights=weights)
            num_features = model.fc.in_features
        elif architecture == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            model = resnet50(weights=weights)
            num_features = model.fc.in_features
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        # 처음에는 모든 층을 얼린다
        for param in model.parameters():
            param.requires_grad = False

        # 마지막 층을 바꾼다
        model.fc = nn.Linear(num_features, num_classes)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        return model

    # 주 모델(ResNet18)을 만든다
    model = create_model('resnet18', num_classes=len(classes))
    model = model.to(device)

    # ============================================================================
    # 4단계: 차츰 녹이기 도우미
    # ============================================================================

    def unfreeze_layers(model, epoch, total_epochs):
        """
        학습이 나아가는 정도에 따라 층을 차츰 녹인다.
        """
        if not GRADUAL_UNFREEZING:
            for name, param in model.named_parameters():
                if any(layer in name for layer in ['layer3', 'layer4']):
                    param.requires_grad = True
            return

        progress = epoch / total_epochs

        if progress < 0.25:
            pass
        elif progress < 0.5:
            for name, param in model.named_parameters():
                if 'layer4' in name:
                    param.requires_grad = True
        elif progress < 0.75:
            for name, param in model.named_parameters():
                if any(layer in name for layer in ['layer3', 'layer4']):
                    param.requires_grad = True
        else:
            for name, param in model.named_parameters():
                if any(layer in name for layer in ['layer2', 'layer3', 'layer4']):
                    param.requires_grad = True

    # ============================================================================
    # 5단계: 최적화기와 일정 조정기 설정
    # ============================================================================

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=BASE_LR,
        weight_decay=0.01
    )

    if USE_COSINE_SCHEDULE:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2, eta_min=1e-6
        )
        print("\nUsing CosineAnnealingWarmRestarts scheduler")
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        print("\nUsing ReduceLROnPlateau scheduler")

    # ============================================================================
    # 6단계: 섞인 정밀도 설정
    # ============================================================================

    if USE_MIXED_PRECISION:
        scaler = GradScaler()
        print("Mixed precision scaler initialized")

    # ============================================================================
    # 7단계: 고급 기능을 갖춘 학습 함수
    # ============================================================================

    def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
        """섞인 정밀도와 기울기 모으기로 한 세대를 학습한다."""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        optimizer.zero_grad()

        for batch_idx, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if USE_MIXED_PRECISION:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss = loss / GRADIENT_ACCUMULATION_STEPS
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss = loss / GRADIENT_ACCUMULATION_STEPS

            if USE_MIXED_PRECISION:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                if USE_MIXED_PRECISION:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'  Batch {batch_idx + 1}/{len(loader)}: '
                      f'Loss: {running_loss / (batch_idx + 1):.3f}, '
                      f'Acc: {100. * correct / total:.2f}%, '
                      f'LR: {current_lr:.6f}')

        epoch_loss = running_loss / len(loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc


    def evaluate(model, loader, criterion, device):
        """표준 평가 함수."""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                if USE_MIXED_PRECISION:
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = running_loss / len(loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    # ============================================================================
    # 8단계: 주 학습 고리
    # ============================================================================

    print(f"\n{'='*70}")
    print(f"Starting advanced training for {NUM_EPOCHS} epochs...")
    print(f"{'='*70}\n")

    best_val_acc = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 70)

        prev_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        unfreeze_layers(model, epoch, NUM_EPOCHS)
        curr_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        if curr_trainable != prev_trainable:
            print(f"Unfroze additional layers: {curr_trainable:,} trainable parameters")
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=optimizer.param_groups[0]['lr'],
                weight_decay=0.01
            )

        if USE_MIXED_PRECISION:
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, device, epoch
            )
        else:
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, None, device, epoch
            )

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")

        if USE_COSINE_SCHEDULE:
            scheduler.step()
        else:
            scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = copy.deepcopy(model.state_dict())
            print(f"  New best model saved")

        print()

    total_time = time.time() - start_time
    print(f"{'='*70}")
    print(f"Training completed in {total_time // 60:.0f}m {total_time % 60:.0f}s")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"{'='*70}\n")

    model.load_state_dict(best_model_weights)

    # ============================================================================
    # 9단계: 고급 평가
    # ============================================================================

    print("Final Test Evaluation:")
    print("=" * 70)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Single Model Test Accuracy: {test_acc:.2f}%")

    # ============================================================================
    # 10단계: 시험 때 불리기 (TTA)
    # ============================================================================

    print("\nTest-Time Augmentation Evaluation:")
    print("-" * 70)

    def evaluate_with_tta(model, loader, device):
        """시험 때 불리기로 평가한다."""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in loader:
                batch_size = inputs.size(0)

                flipped = torch.flip(inputs, dims=[3])
                all_inputs = torch.cat([inputs, flipped], dim=0)
                all_inputs = all_inputs.to(device)

                if USE_MIXED_PRECISION:
                    with autocast():
                        outputs = model(all_inputs)
                else:
                    outputs = model(all_inputs)

                outputs1 = outputs[:batch_size]
                outputs2 = outputs[batch_size:]
                avg_outputs = (outputs1 + outputs2) / 2

                labels = labels.to(device)
                _, predicted = avg_outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total
        return accuracy

    tta_acc = evaluate_with_tta(model, test_loader, device)
    print(f"With TTA Test Accuracy: {tta_acc:.2f}%")
    print(f"TTA Improvement: +{tta_acc - test_acc:.2f}%")

    # ============================================================================
    # 마무리 요약
    # ============================================================================

    print("\n" + "="*70)
    print("ADVANCED TRANSFER LEARNING COMPLETE!")
    print("="*70)

    # 마지막 모델을 저장한다
    torch.save(model.state_dict(), 'advanced_transfer_model.pth')
    print("\nModel saved as 'advanced_transfer_model.pth'")


    if __name__ == "__main__":
        pass
    ```


## 2. 논의

**검증 전처리:** random_split이 같은 train_dataset을 참조하는 Subset 두 개를 만든다. 따라서 val_dataset도 무작위 자르기·색 변화·지우기 등 train_transform을 사용한다. 검증셋에서 고정 val_transform을 사용하려면 별도 데이터셋 객체에 같은 인덱스를 적용하는 등 전처리를 분리해야 한다. 공식 테스트 분할 자체는 따로 있다.

**동결 해제·스케줄:** 학습 가능한 계수가 늘 때마다 AdamW를 새로 만들어 기존 계수의 적률 상태를 잃는다. scheduler는 처음 옵티마이저를 계속 참조해 이후 현재 옵티마이저의 학습률을 조절하지 않는다. 새 계수를 param_group에 추가하거나 스케줄·상태를 함께 재구성하는 정책이 필요하다. 이 구현은 층별 서로 다른 학습률도 실제로 설정하지 않는다. model.train은 동결 부분의 BatchNorm 버퍼를 계속 갱신한다.

**경사 누적:** 같은 크기 마이크로배치 A개에 손실을 1/A로 나누고 한 번 갱신하는 것이 기본이다. 현재 코드는 마지막 배치 수가 A로 나누어떨어지지 않으면 남은 경사를 갱신하지 않고 다음 에포크에서 버린다. 크기가 다른 마지막 배치도 정확한 표본 가중 평균과 다를 수 있다. BatchNorm·드롭아웃 때문에 하나의 큰 배치와 완전히 동일하지 않다.

**AMP:** CUDA가 없으면 이 예제는 혼합 정밀도를 끈다. FP16의 작은 경사에 GradScaler를 사용하지만 모든 가중치·Adam 상태가 절반 크기가 되거나 속도가 2~3배인 것은 아니다. 코드의 torch.cuda.amp 표기는 버전별 API도 확인한다. 스케일러가 갱신을 건너뛰는 경우도 실제 갱신 횟수와 구분한다.

**TTA·앙상블:** FiveCrop tta_transform을 정의하지만 사용하지 않는다. 실제 TTA는 중앙 크롭과 그 좌우 반전 두 로짓을 평균하며, 소프트맥스 확률 평균과 다르다. USE_ENSEMBLE도 설정·출력만 하고 모델 학습·평균에 연결하지 않는다. ensemble_evaluate는 별도 조각이며 각 모델·장치·클래스 순서를 호출자가 맞춰야 한다. 어떤 방법도 테스트 개선을 보장하지 않는다.

가중치는 검증 정확도 최고 시점을 깊게 복사해 복원한다. 마지막 저장 파일에는 모델 state_dict만 있어 옵티마이저·스케줄러·스케일러·분할·전처리 설정을 포함한 완전한 재시작 파일은 아니다. 상단의 데이터 생성·학습과 num_workers=2는 적절한 main 보호 없이 실행되어 프로세스 생성 환경에도 주의해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
마이크로배치 32, 누적 4단계의 실효 배치와 큰 단일 배치와의 차이를 설명하라.

</div>

??? success "연습문제 1 풀이"
    동일 크기 배치 네 개를 한 갱신에 반영하면 128표본이다. 각 경계에서 정확히 평균·초기화해야 하며 Integrated Gradients 설명 기법과는 관계없다. BatchNorm 통계·드롭아웃·마지막 그룹·연산 순서 차이가 있고 실제 처리 시간도 장치·커널에 따라 달라진다. 8GB에서 배치 32가 반드시 들어간다는 보장도 없다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
CosineAnnealingWarmRestarts와 ReduceLROnPlateau의 진행 기준을 비교하라.

</div>

??? success "연습문제 2 풀이"
    전자는 호출에 대응한 주기에서 학습률을 코사인으로 바꾸며 T_0는 첫 주기, T_mult는 다음 주기의 배수다. 후자는 전달한 검증 지표의 개선 규칙에 따라 감소한다. 주기 단위는 step 호출 방식에 달려 있고 재시작은 가중치·옵티마이저 상태 초기화를 뜻하지 않는다. 어느 쪽의 품질이 항상 좋거나 국소 최솟값 탈출이 보장되는 것은 아니다. 현재 코드에서는 옵티마이저 교체 후 스케줄 연결부터 수정해야 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 앙상블 함수의 평균 대상과 사전 조건·비교 방법을 설명하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    def ensemble_evaluate(models, loader, device):
        """소프트맥스 출력을 평균 내어 모델 앙상블을 평가한다."""
        for m in models:
            m.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 모든 모델의 소프트맥스 예측을 평균 낸다
                avg_probs = None
                for m in models:
                    outputs = m(inputs)
                    probs = torch.softmax(outputs, dim=1)
                    if avg_probs is None:
                        avg_probs = probs
                    else:
                        avg_probs += probs
                avg_probs /= len(models)

                _, predicted = avg_probs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return 100. * correct / total

    # 쓰는 법:
    # model_r18 = create_model('resnet18', num_classes=10).to(device)
    # model_r50 = create_model('resnet50', num_classes=10).to(device)
    # ... 두 모델을 학습시킨다 ...
    # ensemble_acc = ensemble_evaluate([model_r18, model_r50], test_loader, device)
    ```

    각 모델의 소프트맥스 확률을 같은 클래스 순서로 평균한다. 모델 목록이 비어 있지 않고 모두 같은 장치·정답 체계를 사용해야 한다. 다른 오류를 내는 모델이 도움이 될 수 있지만 한 모델의 큰 오류·보정 차이가 성능을 해칠 수도 있다. 정확도가 항상 1~3% 오르거나 구조가 다르면 유용한 다양성이 자동 생긴다는 보장은 없다. 실제 지연·메모리도 함께 비교한다.

## 정리하며

고급 기법은 실제 옵티마이저·스케줄 연결과 검증 조건이 맞아야 효과를 해석할 수 있다. 공유된 무작위 검증 변환, 누적 마지막 그룹과 동결 버퍼를 확인하고 TTA·앙상블의 코드 연결 여부와 추가 비용을 구분한다.
