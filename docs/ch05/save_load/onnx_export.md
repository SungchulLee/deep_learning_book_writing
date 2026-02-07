# ONNX Export

## Overview

ONNX (Open Neural Network Exchange) is a cross-framework model format that enables models trained in PyTorch to be deployed in other runtimes (ONNX Runtime, TensorRT, CoreML, OpenVINO). This is the standard path for production deployment.

## Exporting a Model

```python
import torch.onnx

model.eval()
dummy_input = torch.randn(1, 3, 224, 224, device=device)

torch.onnx.export(
    model,
    dummy_input,
    'model.onnx',
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
```

## Key Parameters

- **`dummy_input`**: A tensor with the expected input shape. Used for tracing the computation graph.
- **`opset_version`**: ONNX operator set version. Higher versions support more operations.
- **`dynamic_axes`**: Specifies which dimensions can vary (e.g., batch size).
- **`do_constant_folding`**: Optimizes the graph by folding constant operations.

## Inference with ONNX Runtime

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('model.onnx')
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: np.random.randn(1, 3, 224, 224).astype(np.float32)})
```

## Validation

Always validate that the ONNX model produces the same output as the PyTorch model:

```python
import onnx

# Structural validation
onnx_model = onnx.load('model.onnx')
onnx.checker.check_model(onnx_model)

# Numerical validation
torch_out = model(dummy_input).detach().numpy()
ort_session = ort.InferenceSession('model.onnx')
ort_out = ort_session.run(None, {'input': dummy_input.numpy()})[0]
np.testing.assert_allclose(torch_out, ort_out, rtol=1e-5, atol=1e-5)
```

## Key Takeaways

- ONNX provides cross-framework portability for deployment.
- Use dynamic axes for variable batch sizes.
- Always validate numerical equivalence between PyTorch and ONNX outputs.
