# 유틸리티

배포 유틸리티는 실전 모델 서비스에 필요한 기반을 제공한다. PyTorch 모델과 ONNX 모델의 검증, 지연 백분위수를 포함한 성능 감시, 판올림을 추적하는 모델 레지스트리, 판본 사이의 출력 비교, 그리고 배포에 필요한 산출물을 모두 만들어 내는 완전한 내보내기 파이프라인이다.

## 코드

```python
"""
모델 배포를 위한 유틸리티 함수

이 모듈은 다음을 위한 도우미 함수와 유틸리티를 제공한다:
- 모델 변환
- 성능 감시
- 배포 검증
- 흔한 연산
"""

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
import time
import psutil
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import hashlib

# ========================================================================
# 메인
# ========================================================================


@dataclass
class DeploymentConfig:
    """모델 배포의 설정"""
    model_name: str
    version: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    quantization: bool = False
    batch_size: int = 1
    device: str = 'cpu'
    precision: str = 'fp32'  # fp32, fp16, int8


class ModelValidator:
    """배포 전에 모델을 검증한다"""
    
    @staticmethod
    def validate_pytorch_model(
        model: nn.Module,
        input_shape: Tuple[int, ...],
        num_classes: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        PyTorch 모델을 검증한다
        
        검증 결과를 담은 사전을 돌려준다
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        try:
            # 순전파 시험
            dummy_input = torch.randn(1, *input_shape)
            model.eval()
            with torch.no_grad():
                output = model(dummy_input)
            
            results['info']['output_shape'] = list(output.shape)
            results['info']['num_parameters'] = sum(p.numel() for p in model.parameters())
            
            # 출력의 모양 확인
            if num_classes and output.shape[-1] != num_classes:
                results['warnings'].append(
                    f"Output shape {output.shape} doesn't match num_classes {num_classes}"
                )
            
            # NaN/Inf 확인
            if torch.isnan(output).any():
                results['errors'].append("Model output contains NaN values")
                results['valid'] = False
            
            if torch.isinf(output).any():
                results['errors'].append("Model output contains Inf values")
                results['valid'] = False
                
        except Exception as e:
            results['valid'] = False
            results['errors'].append(f"Forward pass failed: {str(e)}")
        
        return results
    
    @staticmethod
    def validate_onnx_model(
        onnx_path: str,
        input_shape: Tuple[int, ...]
    ) -> Dict[str, Any]:
        """ONNX 모델을 검증한다"""
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        try:
            # 모델을 불러와 확인
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            
            # 추론을 시험한다
            session = ort.InferenceSession(onnx_path)
            input_name = session.get_inputs()[0].name
            dummy_input = np.random.randn(1, *input_shape).astype(np.float32)
            
            output = session.run(None, {input_name: dummy_input})
            
            results['info']['output_shape'] = list(output[0].shape)
            results['info']['providers'] = session.get_providers()
            
            # NaN/Inf 확인
            if np.isnan(output[0]).any():
                results['errors'].append("ONNX output contains NaN values")
                results['valid'] = False
            
        except Exception as e:
            results['valid'] = False
            results['errors'].append(f"ONNX validation failed: {str(e)}")
        
        return results


class PerformanceMonitor:
    """실전에서 모델의 성능을 감시한다"""
    
    def __init__(self):
        self.metrics = {
            'latencies': [],
            'throughputs': [],
            'memory_usage': [],
            'errors': 0,
            'total_requests': 0
        }
    
    def record_inference(
        self,
        latency_ms: float,
        batch_size: int = 1,
        success: bool = True
    ):
        """추론 한 번의 지표를 기록한다"""
        self.metrics['latencies'].append(latency_ms)
        self.metrics['throughputs'].append(batch_size / (latency_ms / 1000))
        self.metrics['total_requests'] += 1
        
        if not success:
            self.metrics['errors'] += 1
        
        # 메모리 사용량
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        self.metrics['memory_usage'].append(memory_mb)
    
    def get_statistics(self) -> Dict[str, Any]:
        """성능 통계를 얻는다"""
        if not self.metrics['latencies']:
            return {}
        
        latencies = np.array(self.metrics['latencies'])
        throughputs = np.array(self.metrics['throughputs'])
        memory = np.array(self.metrics['memory_usage'])
        
        return {
            'latency': {
                'mean': float(np.mean(latencies)),
                'median': float(np.median(latencies)),
                'p95': float(np.percentile(latencies, 95)),
                'p99': float(np.percentile(latencies, 99)),
                'min': float(np.min(latencies)),
                'max': float(np.max(latencies))
            },
            'throughput': {
                'mean': float(np.mean(throughputs)),
                'max': float(np.max(throughputs))
            },
            'memory_mb': {
                'mean': float(np.mean(memory)),
                'peak': float(np.max(memory))
            },
            'reliability': {
                'total_requests': self.metrics['total_requests'],
                'errors': self.metrics['errors'],
                'error_rate': self.metrics['errors'] / max(self.metrics['total_requests'], 1)
            }
        }
    
    def print_report(self):
        """성능 보고서를 정리해 출력한다"""
        stats = self.get_statistics()
        
        if not stats:
            print("No data collected yet")
            return
        
        print("\n" + "="*70)
        print("PERFORMANCE REPORT")
        print("="*70)
        
        print(f"\n{'LATENCY (ms)':<20}")
        print(f"  Mean:     {stats['latency']['mean']:>10.2f}")
        print(f"  Median:   {stats['latency']['median']:>10.2f}")
        print(f"  P95:      {stats['latency']['p95']:>10.2f}")
        print(f"  P99:      {stats['latency']['p99']:>10.2f}")
        
        print(f"\n{'THROUGHPUT (samples/s)':<20}")
        print(f"  Mean:     {stats['throughput']['mean']:>10.1f}")
        print(f"  Peak:     {stats['throughput']['max']:>10.1f}")
        
        print(f"\n{'MEMORY (MB)':<20}")
        print(f"  Average:  {stats['memory_mb']['mean']:>10.1f}")
        print(f"  Peak:     {stats['memory_mb']['peak']:>10.1f}")
        
        print(f"\n{'RELIABILITY':<20}")
        print(f"  Requests: {stats['reliability']['total_requests']:>10}")
        print(f"  Errors:   {stats['reliability']['errors']:>10}")
        print(f"  Error %:  {stats['reliability']['error_rate']*100:>10.2f}")


class ModelRegistry:
    """배포된 모델을 추적하는 간단한 모델 레지스트리"""
    
    def __init__(self, registry_path: str = "./model_registry.json"):
        self.registry_path = Path(registry_path)
        self.models = self._load_registry()
    
    def _load_registry(self) -> Dict:
        """디스크에서 레지스트리를 불러온다"""
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                return json.load(f)
        return {}
    
    def _save_registry(self):
        """레지스트리를 디스크에 저장한다"""
        with open(self.registry_path, 'w') as f:
            json.dump(self.models, f, indent=2)
    
    def register_model(
        self,
        model_name: str,
        version: str,
        model_path: str,
        config: DeploymentConfig,
        metrics: Optional[Dict] = None
    ):
        """새 모델을 등록한다"""
        model_id = f"{model_name}:{version}"
        
        # 무결성 확인을 위한 파일 해시 계산
        file_hash = self._calculate_file_hash(model_path)
        
        self.models[model_id] = {
            'name': model_name,
            'version': version,
            'path': model_path,
            'config': asdict(config),
            'metrics': metrics or {},
            'file_hash': file_hash,
            'registered_at': time.time()
        }
        
        self._save_registry()
        print(f"✓ Registered model: {model_id}")
    
    def get_model(self, model_name: str, version: str = 'latest') -> Optional[Dict]:
        """레지스트리에서 모델의 정보를 얻는다"""
        if version == 'latest':
            # 최신 판본 찾기
            versions = [k for k in self.models.keys() if k.startswith(model_name + ':')]
            if not versions:
                return None
            version = max(versions).split(':')[1]
        
        model_id = f"{model_name}:{version}"
        return self.models.get(model_id)
    
    def list_models(self) -> List[Dict]:
        """등록된 모델을 모두 열거한다"""
        return list(self.models.values())
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """파일의 SHA256 해시를 계산한다"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()


def compare_models(
    model1_path: str,
    model2_path: str,
    test_inputs: List[np.ndarray],
    model1_type: str = 'onnx',
    model2_type: str = 'onnx'
) -> Dict[str, Any]:
    """
    두 모델의 출력을 견준다
    
    변환이나 최적화를 검증하는 데 쓸모 있다
    """
    print("\n=== Model Comparison ===")
    
    # 모델 불러오기
    if model1_type == 'onnx':
        session1 = ort.InferenceSession(model1_path)
        input_name1 = session1.get_inputs()[0].name
    
    if model2_type == 'onnx':
        session2 = ort.InferenceSession(model2_path)
        input_name2 = session2.get_inputs()[0].name
    
    differences = []
    max_diff = 0.0
    
    for i, test_input in enumerate(test_inputs):
        # 추론 실행
        out1 = session1.run(None, {input_name1: test_input})[0]
        out2 = session2.run(None, {input_name2: test_input})[0]
        
        # 차이 계산
        diff = np.abs(out1 - out2)
        differences.append(np.mean(diff))
        max_diff = max(max_diff, np.max(diff))
    
    avg_diff = np.mean(differences)
    
    results = {
        'average_difference': float(avg_diff),
        'max_difference': float(max_diff),
        'outputs_match': max_diff < 1e-5,
        'num_comparisons': len(test_inputs)
    }
    
    print(f"Average difference: {avg_diff:.6f}")
    print(f"Max difference: {max_diff:.6f}")
    print(f"Outputs match: {'✓' if results['outputs_match'] else '✗'}")
    
    return results


def create_model_card(
    model_name: str,
    description: str,
    metrics: Dict[str, Any],
    output_path: str
):
    """
    설명을 담은 모델 카드를 만든다
    
    Following https://arxiv.org/abs/1810.03993
    """
    card = {
        'model_details': {
            'name': model_name,
            'description': description,
            'version': '1.0',
            'date': time.strftime('%Y-%m-%d')
        },
        'model_performance': metrics,
        'intended_use': {
            'primary_uses': 'TODO: Describe primary use cases',
            'out_of_scope': 'TODO: Describe out-of-scope uses'
        },
        'training_data': {
            'description': 'TODO: Describe training data',
            'preprocessing': 'TODO: Describe preprocessing'
        },
        'evaluation_data': {
            'description': 'TODO: Describe evaluation data',
            'metrics': 'TODO: List evaluation metrics'
        },
        'ethical_considerations': {
            'risks': 'TODO: Describe potential risks',
            'mitigations': 'TODO: Describe mitigations'
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(card, f, indent=2)
    
    print(f"✓ Model card saved to {output_path}")


def export_for_production(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    output_dir: str,
    model_name: str = "model",
    quantize: bool = True,
    optimize: bool = True
):
    """
    실전을 위한 완전한 내보내기 파이프라인
    
    필요한 산출물을 모두 만든다:
    - 원래 PyTorch 모델
    - ONNX 모델
    - 양자화한 ONNX 모델 (요청되면)
    - 모델의 설정
    - 검증 보고서
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"EXPORTING MODEL: {model_name}")
    print(f"{'='*70}")
    
    # 1. PyTorch 모델 저장
    print("\n1. Saving PyTorch model...")
    torch_path = output_dir / f"{model_name}.pth"
    torch.save(model.state_dict(), torch_path)
    print(f"   ✓ Saved to {torch_path}")
    
    # 2. ONNX로 내보내기
    print("\n2. Exporting to ONNX...")
    onnx_path = output_dir / f"{model_name}.onnx"
    dummy_input = torch.randn(1, *input_shape)
    
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"   ✓ Saved to {onnx_path}")
    
    # 3. 요청되면 양자화
    if quantize:
        print("\n3. Quantizing ONNX model...")
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        quantized_path = output_dir / f"{model_name}_quantized.onnx"
        quantize_dynamic(
            str(onnx_path),
            str(quantized_path),
            weight_type=QuantType.QInt8
        )
        print(f"   ✓ Saved to {quantized_path}")
    
    # 4. 검증
    print("\n4. Validating models...")
    validator = ModelValidator()
    
    pytorch_validation = validator.validate_pytorch_model(model, input_shape)
    onnx_validation = validator.validate_onnx_model(str(onnx_path), input_shape)
    
    # 5. 설정 저장
    print("\n5. Saving configuration...")
    config = {
        'model_name': model_name,
        'input_shape': list(input_shape),
        'pytorch_model': str(torch_path),
        'onnx_model': str(onnx_path),
        'quantized': quantize,
        'validation': {
            'pytorch': pytorch_validation,
            'onnx': onnx_validation
        }
    }
    
    config_path = output_dir / f"{model_name}_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"   ✓ Saved to {config_path}")
    
    print(f"\n{'='*70}")
    print("EXPORT COMPLETE!")
    print(f"{'='*70}")
    print(f"\nFiles created in {output_dir}:")
    for file in output_dir.iterdir():
        size_mb = file.stat().st_size / (1024*1024)
        print(f"  - {file.name:<40} {size_mb:>8.2f} MB")


if __name__ == "__main__":
    print("Model Deployment Utilities")
    print("="*70)
    print("\nAvailable utilities:")
    print("  - ModelValidator: Validate PyTorch and ONNX models")
    print("  - PerformanceMonitor: Track inference metrics")
    print("  - ModelRegistry: Manage deployed models")
    print("  - compare_models(): Compare model outputs")
    print("  - export_for_production(): Complete export pipeline")
    print("\nImport this module to use these utilities in your code.")```

## 논의

`ModelValidator` 클래스는 PyTorch 모델과 ONNX 모델 모두에 자동 점검을 수행한다. 순전파가 성공하는지 확인하고, 출력에 NaN이나 Inf가 있는지 살피고, 출력의 모양이 기대와 맞는지 확인한다. 이러한 점검은 실전에서 잘못된 예측을 낼 수 있는 조용한 실패를 잡아낸다.

`PerformanceMonitor`은 실전에서 추론 지표를 기록하며 백분위 지연(P95, P99)과 오류율을 계산한다. 이러한 통계는 서비스 수준 목표를 지키고 시간이 갈수록 성능이 떨어지는 현상을 진단하는 데 꼭 필요하다.

`ModelRegistry`은 배포된 모델의 판본을 관리하며, 판본마다 메타데이터, 무결성 확인을 위한 파일 해시, 성능 지표를 함께 저장한다. 이로써 배포를 재현할 수 있고 문제가 생기면 이전 판본으로 되돌릴 수 있다.

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

