# ONNX 기초

ONNX는 여러 프레임워크에 걸쳐 딥러닝 모델을 변환하고 검증하고 실행하는 표준화된 방법을 제공한다. 이 모듈은 ONNX 작업 흐름 전체를 다룬다. PyTorch 모델 변환하기, 모델의 유효성 검증하기, ONNX Runtime으로 추론하기, 출력이 일치하는지 견주기, 추론을 빠르게 하는 그래프 수준의 최적화를 적용하기이다.

## 1. 코드

```python
"""
ONNX 기초 - 모델 변환과 추론

이 모듈은 다음을 보인다:
- PyTorch/TensorFlow 모델을 ONNX 형식으로 바꾸기
- ONNX 모델 불러와 실행하기
- 모델의 유효성 확인하기
- ONNX 모델 최적화하기
"""

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
from typing import Dict, List, Tuple

# ========================================================================
# 메인
# ========================================================================


class SimpleModel(nn.Module):
    """시연을 위한 예시 PyTorch 모델"""
    def __init__(self, input_size: int = 10, hidden_size: int = 20, output_size: int = 5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def convert_pytorch_to_onnx(
    model: nn.Module,
    dummy_input: torch.Tensor,
    onnx_path: str,
    input_names: List[str] = ['input'],
    output_names: List[str] = ['output'],
    dynamic_axes: Dict = None
) -> None:
    """
    PyTorch 모델을 ONNX 형식으로 바꾼다
    
    인수:
        model: PyTorch 모델
        dummy_input: 추적에 쓸 예시 입력 텐서
        onnx_path: ONNX 모델을 저장할 경로
        input_names: 입력 마디의 이름
        output_names: 출력 마디의 이름
        dynamic_axes: 입력의 모양이 변할 수 있는 동적 축
    """
    model.eval()
    
    # 동적 축은 배치 크기와 순차열 길이가 변할 수 있게 해 준다
    if dynamic_axes is None:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    
    # 모델 내보내기
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,  # 알맞은 ONNX opset 판본 쓰기
        do_constant_folding=True,  # 상수를 접어 최적화
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    
    print(f"Model exported to {onnx_path}")


def verify_onnx_model(onnx_path: str) -> bool:
    """
    ONNX 모델이 유효한지 확인한다
    
    인수:
        onnx_path: ONNX 모델의 경로
    
    반환값:
        모델이 유효하면 True
    """
    try:
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print("✓ ONNX model is valid")
        return True
    except Exception as e:
        print(f"✗ ONNX model validation failed: {e}")
        return False


def print_onnx_model_info(onnx_path: str) -> None:
    """ONNX 모델의 정보를 출력한다"""
    model = onnx.load(onnx_path)
    
    print("\n=== Model Information ===")
    print(f"IR Version: {model.ir_version}")
    print(f"Producer: {model.producer_name}")
    print(f"Opset Version: {model.opset_import[0].version}")
    
    print("\n=== Inputs ===")
    for input_tensor in model.graph.input:
        print(f"Name: {input_tensor.name}")
        print(f"Shape: {[d.dim_value for d in input_tensor.type.tensor_type.shape.dim]}")
        print(f"Type: {input_tensor.type.tensor_type.elem_type}")
    
    print("\n=== Outputs ===")
    for output_tensor in model.graph.output:
        print(f"Name: {output_tensor.name}")
        print(f"Shape: {[d.dim_value for d in output_tensor.type.tensor_type.shape.dim]}")


class ONNXInferenceSession:
    """ONNX Runtime 추론을 감싸는 클래스"""
    
    def __init__(self, onnx_path: str, providers: List[str] = None):
        """
        ONNX Runtime 세션 초기화
        
        인수:
            onnx_path: ONNX 모델의 경로
            providers: 실행 제공자의 목록 (예: ['CUDAExecutionProvider', 'CPUExecutionProvider'])
        """
        if providers is None:
            providers = ['CPUExecutionProvider']
        
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"Loaded model with providers: {self.session.get_providers()}")
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        추론 실행
        
        인수:
            input_data: 입력 numpy 배열
        
        반환값:
            모델의 출력
        """
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_data}
        )
        return outputs[0]
    
    def get_model_metadata(self) -> Dict:
        """모델의 메타데이터를 얻는다"""
        return {
            'inputs': [(i.name, i.shape, i.type) for i in self.session.get_inputs()],
            'outputs': [(o.name, o.shape, o.type) for o in self.session.get_outputs()],
            'providers': self.session.get_providers()
        }


def compare_pytorch_onnx_outputs(
    pytorch_model: nn.Module,
    onnx_path: str,
    test_input: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-5
) -> Tuple[bool, float]:
    """
    PyTorch 모델과 ONNX 모델의 출력을 견준다
    
    인수:
        pytorch_model: 원래 PyTorch 모델
        onnx_path: ONNX 모델의 경로
        test_input: 시험 입력 텐서
        rtol: 상대 허용 오차
        atol: 절대 허용 오차
    
    반환값:
        (일치 여부, 최대 차이)의 튜플
    """
    # PyTorch 추론
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input).numpy()
    
    # ONNX 추론
    onnx_session = ONNXInferenceSession(onnx_path)
    onnx_output = onnx_session.predict(test_input.numpy())
    
    # 비교
    max_diff = np.max(np.abs(pytorch_output - onnx_output))
    match = np.allclose(pytorch_output, onnx_output, rtol=rtol, atol=atol)
    
    print(f"\n=== Output Comparison ===")
    print(f"Max difference: {max_diff}")
    print(f"Outputs match (rtol={rtol}, atol={atol}): {match}")
    
    return match, max_diff


def optimize_onnx_model(input_path: str, output_path: str) -> None:
    """
    ONNX 최적화기로 ONNX 모델을 최적화한다
    
    인수:
        input_path: 입력 ONNX 모델의 경로
        output_path: 최적화한 모델을 저장할 경로
    """
    import onnx
    from onnx import optimizer
    
    # 모델을 불러온다
    model = onnx.load(input_path)
    
    # 최적화 적용
    passes = [
        'eliminate_identity',
        'eliminate_nop_transpose',
        'eliminate_nop_pad',
        'eliminate_unused_initializer',
        'fuse_consecutive_transposes',
        'fuse_transpose_into_gemm',
        'fuse_bn_into_conv',
    ]
    
    optimized_model = optimizer.optimize(model, passes)
    
    # 최적화한 모델 저장
    onnx.save(optimized_model, output_path)
    print(f"Optimized model saved to {output_path}")


def demo_onnx_workflow():
    """ONNX 작업 흐름 전체를 보인다"""
    print("=== ONNX Workflow Demo ===\n")
    
    # 1. PyTorch 모델 만들기
    print("1. Creating PyTorch model...")
    model = SimpleModel(input_size=10, hidden_size=20, output_size=5)
    
    # 2. ONNX로 바꾸기
    print("\n2. Converting to ONNX...")
    dummy_input = torch.randn(1, 10)
    onnx_path = "simple_model.onnx"
    convert_pytorch_to_onnx(model, dummy_input, onnx_path)
    
    # 3. 모델 검증
    print("\n3. Verifying ONNX model...")
    verify_onnx_model(onnx_path)
    
    # 4. 모델 정보 출력
    print_onnx_model_info(onnx_path)
    
    # 5. 추론 실행
    print("\n4. Running ONNX inference...")
    onnx_session = ONNXInferenceSession(onnx_path)
    test_input = torch.randn(5, 10)  # 크기 5인 배치
    output = onnx_session.predict(test_input.numpy())
    print(f"Output shape: {output.shape}")
    
    # 6. 출력 비교
    print("\n5. Comparing PyTorch vs ONNX outputs...")
    compare_pytorch_onnx_outputs(model, onnx_path, test_input)
    
    # 7. 모델 최적화
    print("\n6. Optimizing ONNX model...")
    optimized_path = "simple_model_optimized.onnx"
    optimize_onnx_model(onnx_path, optimized_path)


if __name__ == "__main__":
    demo_onnx_workflow()```

## 2. 논의

ONNX 작업 흐름은 네 단계로 이루어진다. 변환(학습된 모델 내보내기), 검증(구조의 유효성 확인), 추론(ONNX Runtime으로 모델 실행), 최적화(속도를 위한 그래프 변환 적용)이다. 각 단계를 따로 수행할 수 있으므로 모델을 한 번 변환해 여러 플랫폼에 배포할 수 있다.

ONNX Runtime은 하드웨어마다 실행 제공자를 제공한다. CPU, CUDA(NVIDIA GPU), TensorRT, OpenVINO(인텔), CoreML(애플) 등이다. 제공자만 바꾸어 지정하면 같은 ONNX 모델을 코드 수정 없이 여러 하드웨어에서 돌릴 수 있다.

변환이 제대로 되었는지 확인하려면 PyTorch와 ONNX의 출력을 견주어 보아야 한다. 프레임워크마다 부동소수점 연산의 순서가 달라 작은 수치 차이($10^{-3}$ 아래)는 으레 생긴다. 차이가 그보다 크면 변환 오류이거나 지원되지 않는 연산이 있다는 뜻이다.

## 연습문제

**연습문제 1.**
코드를 따라가며 쓰인 주요 자료 구조를 찾아라. 각각에 대해 자료형, (해당한다면) 모양, 파이프라인에서의 구실을 적어라.

??? success "연습문제 1 풀이"
    코드를 꼼꼼히 읽으며 변수 대입마다 살펴본다. 텐서는 `.shape`과 `.dtype`을 확인하고, 클래스는 `__init__`의 매개변수와 `forward`/`__call__`의 서명을 확인한다. 이름, 자료형, 모양, 구실을 열로 하는 표에 정리한다.

---


**연습문제 2.**
오류 처리와 입력 검증을 넣도록 코드를 고쳐라. 이 코드를 실전에 쓸 수 있게 하려면 어떤 검사를 더하겠는가?

??? success "연습문제 2 풀이"
    입력에 자료형 검사(`isinstance`), 모양 검증(`assert tensor.dim() == expected`), 값 범위 검사(예: 확률이 [0,1] 안인지)를 넣고, 입출력 연산은 try-except로 감싼다. 빈 배치나 NaN 같은 경계 상황에는 경고를 남긴다. 매개변수와 반환값의 자료형을 적은 독스트링을 붙인다.

---


**연습문제 3.**
직접 고른 새로운 쓰임새를 지원하도록 코드를 확장하라. 무엇을 왜 바꿀지 설명하라.

??? success "연습문제 3 풀이"
    알맞은 확장을 하나 고른다(예: 다른 데이터셋, 지표 추가, 새 모델 변형). 필요한 변경을 설명한다. 새 임포트, 클래스 정의 수정, 초매개변수 갱신, 새로운 시각화나 기록 등이다. 핵심 변경을 구현하고 간단한 시험으로 올바름을 확인한다.

## 정리하며

**다룬 것** — ONNX 기초

ONNX 작업 흐름은 네 단계로 이루어진다.

핵심 클래스는 `SimpleModel`, `ONNXInferenceSession`이며 앞의 연습문제 3개로 직접 확인할 수 있다.
