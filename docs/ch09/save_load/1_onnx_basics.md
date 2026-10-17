# ONNX 기초

ONNX 변환, 형식 검사, 추론 세션과 원본 출력 비교를 함수별로 나누어 살펴본다. 그래프 최적화와 실행 제공자의 역할도 함께 다룬다.

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
    demo_onnx_workflow()
```

## 2. 논의

ONNXInferenceSession은 기본적으로 CPUExecutionProvider를 사용하고 첫 입력·첫 출력에 초점을 둔 단순한 래퍼이다. 여러 입력이나 출력이 있는 모델에는 이름, 자료형과 반환 구조를 확장해야 한다. 실행 제공자를 지정한다고 모든 연산이 그 하드웨어에서 실행되는 것은 아니다. 설치된 제공자와 연산 지원, 장치와 드라이버 조건을 확인한다. [실행 제공자 문서](https://onnxruntime.ai/docs/execution-providers/index.html)를 참고한다.

형식 검사는 그래프의 구조적 유효성을, 출력 비교는 주어진 입력에서의 수치 차이를 확인한다. allclose는 절대·상대 허용 오차를 함께 사용한다. 어떤 절대 오차보다 크다는 이유만으로 변환 오류를 확정하거나, 작다는 이유만으로 모든 입력에서 같은 계산이라고 결론 내리지 않는다.

내보내기의 opset과 exporter는 실행 환경에 맞춰야 한다. 특히 optimize_onnx_model의 from onnx import optimizer는 이전 API이다. 최적화 패키지는 [onnxoptimizer](https://github.com/onnx/optimizer)로 분리되어 있으므로 최신 환경에서 이 함수가 그대로 동작한다고 가정할 수 없다. 최적화 후에는 형식·수치 결과를 다시 비교하고 목표 장치에서 성능을 측정해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
ONNXInferenceSession의 기본 실행 장치는 무엇인가?

</div>

??? success "풀이"
    providers를 주지 않으면 CPUExecutionProvider를 사용한다. CUDA 등은 해당 제공자를 지원하는 설치와 장치가 필요하며 세션에서 실제 선택된 제공자를 확인해야 한다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 래퍼가 여러 입력을 받는 모델에 그대로 적합한가?

</div>

??? success "풀이"
    첫 입력 이름에 하나의 배열만 전달하므로 일반적인 다중 입력 모델에는 충분하지 않다. 각 입력의 이름·모양·자료형을 맞춘 사전과 필요한 출력 처리로 확장해야 한다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
그래프 최적화 후 무엇을 다시 확인해야 하는가?

</div>

??? success "풀이"
    형식 검사, 대표 입력의 원본 대비 출력 오차와 허용 입력 모양을 확인한다. 실제 지연·처리량도 목표 제공자에서 측정해야 하며 노드 수 감소만으로 속도를 판단하지 않는다.

## 정리하며

변환·형식 검사·수치 비교·성능 측정은 각기 다른 단계이다. API 버전과 런타임 지원 범위를 확인하고 단일 입력 예제의 범위를 넘겨 일반화하지 않는다.
