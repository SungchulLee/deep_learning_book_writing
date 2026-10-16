# MNIST 확산 모델 학습 실행 예제

이 페이지는 [MNIST 확산 모델](mnist_diffusion.md)의 학습 클래스를 설정 사전으로 호출하는 진입점이다. 모델이나 로스를 새로 정의하지 않고 시간 단계 수, 배치 크기, 러닝레이트와 저장 간격을 한곳에 모은다.

## 1. 코드

```python
"""
익히기 대본 보기

기본 자리매김으로 MNIST 확산 모델을 익히는 단순한 대본.
실험에 맞게 웃파라메터를 조정하라.
"""

from mnist_diffusion import MNISTDiffusion

# ========================================================================
# 메인
# ========================================================================


def main():
    # 설정
    config = {
        'timesteps': 1000,       # 확산 걸음 수
        'batch_size': 128,       # 익히기 배치 크기
        'learning_rate': 2e-4,   # 러닝 비율
        'epochs': 100,           # 익히기 바퀴 수
        'save_interval': 10,     # N바퀴마다 표본을 갈무리한다
    }
    
    print("Training Configuration:")
    print("-" * 40)
    for key, value in config.items():
        print(f"{key:20s}: {value}")
    print("-" * 40)
    print()
    
    # 모형을 시작한다
    diffusion = MNISTDiffusion(
        timesteps=config['timesteps'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate']
    )
    
    # 학습
    diffusion.train(
        epochs=config['epochs'],
        save_interval=config['save_interval']
    )
    
    # 마지막 모델을 저장한다
    diffusion.save_checkpoint('mnist_diffusion_model.pt')
    
    print("\nTraining complete! Model saved to 'mnist_diffusion_model.pt'")
    print("Check the 'samples' folder for generated images during training.")


if __name__ == "__main__":
    main()
```

## 2. 설정이 전달되는 경로

생성자에는 timesteps=1000, batch_size=128, learning_rate=0.0002를 전달한다. epochs=100과 save_interval=10은 이후 train 메서드에 전달한다. 각 인수가 소비되는 위치를 구분하면 생성자의 모델 설정과 학습 반복 설정을 혼동하지 않을 수 있다.

MNISTDiffusion은 코사인 스케줄, base_channels=64의 SimpleUNet, Adam과 EMA 모델을 준비한다. 별도 장치를 지정하지 않아 CUDA가 사용 가능하면 CUDA를, 그렇지 않으면 CPU를 선택한다. 이 파일은 장치나 난수 시드, 스케줄 종류를 직접 지정하는 인터페이스를 제공하지 않는다.

## 3. 실행 시 수행하는 작업

실제로 main을 실행하면 학습 클래스가 MNIST 로더를 만들며 필요한 경우 데이터를 내려받는다. 픽셀을 [−1,1]로 바꾸고 균등하게 뽑은 시간별 노이즈 예측 MSE를 최소화한다. 라벨은 사용하지 않으며 별도 테스트 평가는 없다.

60,000개 데이터를 배치 128개씩 마지막 배치까지 사용하므로 한 에포크는 469회, 100에포크는 46,900회의 optimizer 업데이트다. 생성에 쓰는 1,000단계는 **배치당 1,000회 학습한다는 뜻이 아니다**. 각 학습 이미지에는 무작위로 고른 한 단계의 노이즈를 직접 섞는다.

train은 에포크 1과 이후 10에포크마다 EMA 샘플을 저장하고, 50에포크마다 체크포인트를 남긴다. 종료 시 training_loss.png를 만들며, 이 진입점은 추가로 mnist_diffusion_model.pt를 저장한다. 실행 시간이나 로스 감소, 생성 품질을 확인한 결과가 이 페이지에 제시된 것은 아니다.

## 4. 재개와 비교 실험의 조건

설정 사전을 출력하는 것은 실행 조건을 살피는 데 도움이 되지만 완전한 실험 기록은 아니다. 난수 상태와 소프트웨어 버전, 데이터 처리 조건을 함께 기록해야 재현성을 판단할 수 있다. 기존 파일과 같은 이름으로 저장하므로 반복 실행 결과를 구분할 별도 경로도 필요하다.

이 예제에는 체크포인트를 불러오는 분기가 없어 호출할 때 새 모델로 학습을 시작한다. 클래스의 load_checkpoint를 사용하더라도 저장된 확산 계수와 에포크·난수 상태는 복원하지 않는다. 스케줄 변경 실험은 생성 조건까지 일치시키고 같은 계산 예산에서 별도로 비교해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
config의 다섯 항목이 각각 어디로 전달되는지 설명하라.

</div>

??? success "연습문제 1 풀이"

    timesteps, batch_size, learning_rate는 MNISTDiffusion 생성자에 전달한다. epochs와 save_interval은 train 호출에 전달한다. 전자는 확산 계수·로더·optimizer를 구성하고 후자는 반복 횟수와 중간 샘플 저장 간격을 정한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
1,000단계와 100에포크가 학습 비용에 미치는 의미를 구분하라.

</div>

??? success "연습문제 2 풀이"

    한 배치의 각 이미지에 시간 하나를 뽑아 노이즈 입력을 직접 만들고 예측기를 한 번 호출한다. 한 에포크는 469번, 전체는 46,900번 업데이트한다. 반면 중간 샘플 생성은 1,000단계의 순차 모델 호출을 수행하므로 별도 생성 비용이 든다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 진입점이 자동으로 중단 학습을 재개하는가? 공정한 설정 비교에 추가로 필요한 것은 무엇인가?

</div>

??? success "연습문제 3 풀이"

    load_checkpoint를 호출하지 않아 새 모델에서 시작한다. 재개하려면 상태 복원 경로와 동일한 확산 설정을 갖추어야 한다. 비교에는 같은 데이터·계산 예산·평가 절차와 난수 조건, 결과 파일의 구분이 필요하며 어느 스케줄이 항상 우수하다고 미리 결론낼 수 없다.



## 정리하며

이 파일은 설정 사전과 학습 클래스 호출을 연결하는 진입점이다. 학습 단계와 생성 단계를 구분하고, 저장 파일·검증 범위·재개 기능의 유무를 확인한다.
