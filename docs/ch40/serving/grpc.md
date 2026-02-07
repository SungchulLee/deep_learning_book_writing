# gRPC Serving

## Overview

gRPC (Google Remote Procedure Call) provides high-performance, low-latency model serving using Protocol Buffers for serialization and HTTP/2 for transport. For latency-sensitive applications like algorithmic trading, gRPC offers significant advantages over REST APIs: binary serialization (smaller payloads), streaming support, and strong typing.

## Why gRPC for Model Serving?

| Feature | REST (JSON) | gRPC (Protobuf) |
|---------|-------------|-----------------|
| Serialization | Text (JSON) | Binary (Protobuf) |
| Payload size | Larger | 3-10× smaller |
| Latency | Higher | Lower |
| Streaming | Limited | Bidirectional |
| Type safety | Runtime | Compile-time |
| Code generation | Manual | Automatic |

## Protocol Buffer Definition

```protobuf
// model_service.proto
syntax = "proto3";

package inference;

service ModelService {
    // Unary prediction
    rpc Predict(PredictRequest) returns (PredictResponse);
    
    // Streaming predictions (for real-time data)
    rpc StreamPredict(stream PredictRequest) returns (stream PredictResponse);
    
    // Health check
    rpc HealthCheck(Empty) returns (HealthResponse);
}

message PredictRequest {
    repeated float features = 1;
    int32 batch_size = 2;
    string model_version = 3;
}

message PredictResponse {
    repeated float predictions = 1;
    float latency_ms = 2;
    string model_version = 3;
}

message Empty {}

message HealthResponse {
    bool healthy = 1;
    string status = 2;
}
```

## Python gRPC Server

```python
import grpc
from concurrent import futures
import torch
import time

class ModelServicer:
    def __init__(self, model, device='cpu'):
        self.model = model.eval().to(device)
        self.device = device
    
    def Predict(self, request, context):
        features = torch.tensor(request.features).unsqueeze(0).to(self.device)
        
        start = time.perf_counter()
        with torch.no_grad():
            prediction = self.model(features)
        latency = (time.perf_counter() - start) * 1000
        
        return PredictResponse(
            predictions=prediction.squeeze().tolist(),
            latency_ms=latency
        )

def serve(model, port=50051, max_workers=4):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    # Add servicer to server
    # model_service_pb2_grpc.add_ModelServiceServicer_to_server(
    #     ModelServicer(model), server
    # )
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    server.wait_for_termination()
```

## Best Practices

- **Use streaming RPCs** for continuous prediction (market data feeds)
- **Implement deadlines** to prevent slow requests from consuming resources
- **Enable compression** for large payloads (`grpc.Compression.Gzip`)
- **Use connection pooling** on the client side for high-throughput scenarios
- **Monitor with interceptors** for latency tracking and error rates

## References

1. gRPC Documentation: https://grpc.io/docs/
2. gRPC Python Tutorial: https://grpc.io/docs/languages/python/basics/
