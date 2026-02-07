# Production Monitoring

## Introduction

Production monitoring tracks model performance, health, and behavior in real-world deployment. Effective monitoring enables early detection of issues, performance optimization, and data-driven decision making.

## Monitoring Architecture

A comprehensive monitoring system includes:

```
Metrics Collection → Aggregation → Visualization → Alerting
```

**Metrics Collection** — Gather performance data from inference endpoints.

**Aggregation** — Store and compute statistics over time windows.

**Visualization** — Dashboards for real-time and historical analysis.

**Alerting** — Notifications for anomalies and threshold violations.

## Performance Metrics

### Core Inference Metrics

```python
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any
from collections import deque
import threading

@dataclass
class InferenceMetrics:
    """Container for inference performance metrics."""
    latencies: List[float] = field(default_factory=list)
    throughputs: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    errors: int = 0
    total_requests: int = 0

class PerformanceMonitor:
    """Monitor model performance in production."""
    
    def __init__(self, window_size: int = 1000):
        """
        Initialize monitor.
        
        Args:
            window_size: Number of recent measurements to keep
        """
        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)
        self.throughputs = deque(maxlen=window_size)
        self.memory_usage = deque(maxlen=window_size)
        self.errors = 0
        self.total_requests = 0
        self.lock = threading.Lock()
    
    def record_inference(
        self,
        latency_ms: float,
        batch_size: int = 1,
        success: bool = True
    ):
        """
        Record metrics for a single inference.
        
        Args:
            latency_ms: Inference latency in milliseconds
            batch_size: Number of samples in batch
            success: Whether inference succeeded
        """
        with self.lock:
            self.latencies.append(latency_ms)
            self.throughputs.append(batch_size / (latency_ms / 1000))
            self.total_requests += 1
            
            if not success:
                self.errors += 1
            
            # Record memory usage
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / (1024 * 1024)
                self.memory_usage.append(memory_mb)
            except ImportError:
                pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        with self.lock:
            if not self.latencies:
                return {}
            
            latencies = np.array(self.latencies)
            throughputs = np.array(self.throughputs)
            
            stats = {
                'latency': {
                    'mean': float(np.mean(latencies)),
                    'median': float(np.median(latencies)),
                    'std': float(np.std(latencies)),
                    'p50': float(np.percentile(latencies, 50)),
                    'p95': float(np.percentile(latencies, 95)),
                    'p99': float(np.percentile(latencies, 99)),
                    'min': float(np.min(latencies)),
                    'max': float(np.max(latencies))
                },
                'throughput': {
                    'mean': float(np.mean(throughputs)),
                    'max': float(np.max(throughputs)),
                    'current': float(throughputs[-1]) if throughputs else 0
                },
                'reliability': {
                    'total_requests': self.total_requests,
                    'errors': self.errors,
                    'error_rate': self.errors / max(self.total_requests, 1),
                    'success_rate': 1 - self.errors / max(self.total_requests, 1)
                }
            }
            
            if self.memory_usage:
                memory = np.array(self.memory_usage)
                stats['memory_mb'] = {
                    'mean': float(np.mean(memory)),
                    'peak': float(np.max(memory)),
                    'current': float(memory[-1])
                }
            
            return stats
    
    def print_report(self):
        """Print formatted performance report."""
        stats = self.get_statistics()
        
        if not stats:
            print("No data collected yet")
            return
        
        print("\n" + "="*60)
        print("PERFORMANCE REPORT")
        print("="*60)
        
        print(f"\n{'LATENCY (ms)':<20}")
        print(f"  Mean:     {stats['latency']['mean']:>10.2f}")
        print(f"  Median:   {stats['latency']['median']:>10.2f}")
        print(f"  P95:      {stats['latency']['p95']:>10.2f}")
        print(f"  P99:      {stats['latency']['p99']:>10.2f}")
        
        print(f"\n{'THROUGHPUT (samples/s)':<20}")
        print(f"  Mean:     {stats['throughput']['mean']:>10.1f}")
        print(f"  Peak:     {stats['throughput']['max']:>10.1f}")
        
        if 'memory_mb' in stats:
            print(f"\n{'MEMORY (MB)':<20}")
            print(f"  Mean:     {stats['memory_mb']['mean']:>10.1f}")
            print(f"  Peak:     {stats['memory_mb']['peak']:>10.1f}")
        
        print(f"\n{'RELIABILITY':<20}")
        print(f"  Requests: {stats['reliability']['total_requests']:>10}")
        print(f"  Errors:   {stats['reliability']['errors']:>10}")
        print(f"  Success:  {stats['reliability']['success_rate']*100:>10.2f}%")
    
    def reset(self):
        """Reset all metrics."""
        with self.lock:
            self.latencies.clear()
            self.throughputs.clear()
            self.memory_usage.clear()
            self.errors = 0
            self.total_requests = 0

# Usage example
monitor = PerformanceMonitor()

# During inference
start = time.time()
output = model(input_data)
latency_ms = (time.time() - start) * 1000

monitor.record_inference(latency_ms, batch_size=1, success=True)

# Get statistics
stats = monitor.get_statistics()
monitor.print_report()
```

### SLA Monitoring

Track Service Level Agreement (SLA) compliance:

```python
class SLAMonitor:
    """Monitor SLA compliance."""
    
    def __init__(
        self,
        latency_p99_ms: float = 100,
        error_rate_threshold: float = 0.01,
        availability_target: float = 0.999
    ):
        """
        Initialize SLA monitor.
        
        Args:
            latency_p99_ms: P99 latency target in milliseconds
            error_rate_threshold: Maximum acceptable error rate
            availability_target: Minimum availability (0-1)
        """
        self.latency_target = latency_p99_ms
        self.error_threshold = error_rate_threshold
        self.availability_target = availability_target
        
        self.latencies = []
        self.errors = 0
        self.total_requests = 0
        self.downtime_seconds = 0
        self.uptime_seconds = 0
    
    def record(self, latency_ms: float, success: bool):
        """Record inference result."""
        self.latencies.append(latency_ms)
        self.total_requests += 1
        if not success:
            self.errors += 1
    
    def record_health(self, healthy: bool, interval_seconds: float):
        """Record health check result."""
        if healthy:
            self.uptime_seconds += interval_seconds
        else:
            self.downtime_seconds += interval_seconds
    
    def get_compliance(self) -> Dict[str, Any]:
        """Check SLA compliance."""
        if not self.latencies:
            return {'compliant': True, 'metrics': {}}
        
        p99_latency = np.percentile(self.latencies, 99)
        error_rate = self.errors / self.total_requests
        total_time = self.uptime_seconds + self.downtime_seconds
        availability = self.uptime_seconds / total_time if total_time > 0 else 1.0
        
        latency_compliant = p99_latency <= self.latency_target
        error_compliant = error_rate <= self.error_threshold
        availability_compliant = availability >= self.availability_target
        
        return {
            'compliant': latency_compliant and error_compliant and availability_compliant,
            'metrics': {
                'latency_p99_ms': p99_latency,
                'latency_target_ms': self.latency_target,
                'latency_compliant': latency_compliant,
                'error_rate': error_rate,
                'error_threshold': self.error_threshold,
                'error_compliant': error_compliant,
                'availability': availability,
                'availability_target': self.availability_target,
                'availability_compliant': availability_compliant
            }
        }
```

## Prometheus Integration

### Metric Definitions

```python
from prometheus_client import Counter, Histogram, Gauge, Summary

# Request metrics
REQUESTS_TOTAL = Counter(
    'inference_requests_total',
    'Total number of inference requests',
    ['model_name', 'model_version', 'status']
)

REQUESTS_IN_PROGRESS = Gauge(
    'inference_requests_in_progress',
    'Number of inference requests currently being processed',
    ['model_name']
)

# Latency metrics
LATENCY_HISTOGRAM = Histogram(
    'inference_latency_seconds',
    'Inference latency in seconds',
    ['model_name', 'model_version'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

LATENCY_SUMMARY = Summary(
    'inference_latency_summary_seconds',
    'Inference latency summary',
    ['model_name']
)

# Throughput metrics
BATCH_SIZE = Histogram(
    'inference_batch_size',
    'Inference batch size',
    ['model_name'],
    buckets=[1, 2, 4, 8, 16, 32, 64, 128]
)

# Resource metrics
GPU_UTILIZATION = Gauge(
    'gpu_utilization_percent',
    'GPU utilization percentage',
    ['device']
)

GPU_MEMORY_USED = Gauge(
    'gpu_memory_used_bytes',
    'GPU memory used in bytes',
    ['device']
)

MODEL_LOADED = Gauge(
    'model_loaded',
    'Whether model is loaded',
    ['model_name', 'model_version']
)
```

### Instrumented Inference

```python
import time
from contextlib import contextmanager

@contextmanager
def monitored_inference(model_name: str, model_version: str):
    """Context manager for monitored inference."""
    REQUESTS_IN_PROGRESS.labels(model_name=model_name).inc()
    start_time = time.time()
    status = 'success'
    
    try:
        yield
    except Exception:
        status = 'error'
        raise
    finally:
        latency = time.time() - start_time
        
        REQUESTS_TOTAL.labels(
            model_name=model_name,
            model_version=model_version,
            status=status
        ).inc()
        
        LATENCY_HISTOGRAM.labels(
            model_name=model_name,
            model_version=model_version
        ).observe(latency)
        
        LATENCY_SUMMARY.labels(model_name=model_name).observe(latency)
        
        REQUESTS_IN_PROGRESS.labels(model_name=model_name).dec()

# Usage
with monitored_inference("resnet50", "1.0"):
    output = model(input_data)
    BATCH_SIZE.labels(model_name="resnet50").observe(input_data.shape[0])
```

### Expose Metrics Endpoint

```python
from prometheus_client import start_http_server, generate_latest
from fastapi import FastAPI, Response

app = FastAPI()

@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )

# Or standalone
start_http_server(9090)
```

## GPU Monitoring

### NVIDIA GPU Metrics

```python
def get_gpu_metrics() -> Dict[str, Any]:
    """Get NVIDIA GPU metrics using pynvml."""
    try:
        import pynvml
        pynvml.nvmlInit()
        
        device_count = pynvml.nvmlDeviceGetCount()
        metrics = []
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # Utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            # Memory
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # Temperature
            temp = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
            
            # Power
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW to W
            
            metrics.append({
                'device_id': i,
                'name': pynvml.nvmlDeviceGetName(handle).decode(),
                'gpu_utilization': util.gpu,
                'memory_utilization': util.memory,
                'memory_used_mb': mem.used / (1024**2),
                'memory_total_mb': mem.total / (1024**2),
                'temperature_c': temp,
                'power_w': power
            })
        
        pynvml.nvmlShutdown()
        return {'devices': metrics}
        
    except Exception as e:
        return {'error': str(e)}

# Update Prometheus gauges
def update_gpu_metrics():
    """Update GPU metrics in Prometheus."""
    metrics = get_gpu_metrics()
    
    if 'devices' in metrics:
        for device in metrics['devices']:
            device_label = str(device['device_id'])
            GPU_UTILIZATION.labels(device=device_label).set(
                device['gpu_utilization']
            )
            GPU_MEMORY_USED.labels(device=device_label).set(
                device['memory_used_mb'] * 1024 * 1024
            )
```

### PyTorch CUDA Monitoring

```python
import torch

def get_pytorch_cuda_metrics() -> Dict[str, Any]:
    """Get CUDA metrics from PyTorch."""
    if not torch.cuda.is_available():
        return {'cuda_available': False}
    
    metrics = {
        'cuda_available': True,
        'device_count': torch.cuda.device_count(),
        'devices': []
    }
    
    for i in range(torch.cuda.device_count()):
        device_metrics = {
            'device_id': i,
            'name': torch.cuda.get_device_name(i),
            'memory_allocated_mb': torch.cuda.memory_allocated(i) / (1024**2),
            'memory_reserved_mb': torch.cuda.memory_reserved(i) / (1024**2),
            'max_memory_allocated_mb': torch.cuda.max_memory_allocated(i) / (1024**2)
        }
        
        # Memory stats
        if hasattr(torch.cuda, 'memory_stats'):
            stats = torch.cuda.memory_stats(i)
            device_metrics['num_alloc_retries'] = stats.get('num_alloc_retries', 0)
        
        metrics['devices'].append(device_metrics)
    
    return metrics
```

## Model Registry

### Track Deployed Models

```python
import json
import hashlib
from pathlib import Path
from datetime import datetime

class ModelRegistry:
    """Track deployed models."""
    
    def __init__(self, registry_path: str = "./model_registry.json"):
        self.registry_path = Path(registry_path)
        self.models = self._load()
    
    def _load(self) -> Dict:
        """Load registry from disk."""
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                return json.load(f)
        return {}
    
    def _save(self):
        """Save registry to disk."""
        with open(self.registry_path, 'w') as f:
            json.dump(self.models, f, indent=2, default=str)
    
    def register(
        self,
        model_name: str,
        version: str,
        model_path: str,
        metrics: Dict = None,
        metadata: Dict = None
    ):
        """Register a model."""
        model_id = f"{model_name}:{version}"
        
        # Calculate file hash
        file_hash = self._calculate_hash(model_path)
        
        self.models[model_id] = {
            'name': model_name,
            'version': version,
            'path': str(model_path),
            'file_hash': file_hash,
            'metrics': metrics or {},
            'metadata': metadata or {},
            'registered_at': datetime.utcnow().isoformat(),
            'status': 'registered'
        }
        
        self._save()
        print(f"✓ Registered: {model_id}")
    
    def deploy(self, model_name: str, version: str):
        """Mark model as deployed."""
        model_id = f"{model_name}:{version}"
        if model_id in self.models:
            self.models[model_id]['status'] = 'deployed'
            self.models[model_id]['deployed_at'] = datetime.utcnow().isoformat()
            self._save()
    
    def get(self, model_name: str, version: str = 'latest') -> Dict:
        """Get model info."""
        if version == 'latest':
            versions = [
                k for k in self.models.keys() 
                if k.startswith(f"{model_name}:")
            ]
            if not versions:
                return None
            version = max(versions).split(':')[1]
        
        return self.models.get(f"{model_name}:{version}")
    
    def list_models(self) -> List[Dict]:
        """List all models."""
        return list(self.models.values())
    
    def _calculate_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

# Usage
registry = ModelRegistry()
registry.register(
    model_name="resnet50",
    version="1.0.0",
    model_path="model.onnx",
    metrics={'accuracy': 0.76, 'latency_p99_ms': 45},
    metadata={'framework': 'pytorch', 'input_shape': [3, 224, 224]}
)
```

## Alerting

### Threshold-Based Alerts

```python
from enum import Enum
from typing import Callable, Optional
import smtplib
from email.message import EmailMessage

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class Alert:
    """Alert definition."""
    
    def __init__(
        self,
        name: str,
        condition: Callable[[], bool],
        message: str,
        severity: AlertSeverity = AlertSeverity.WARNING
    ):
        self.name = name
        self.condition = condition
        self.message = message
        self.severity = severity
        self.last_triggered = None
        self.trigger_count = 0

class AlertManager:
    """Manage and trigger alerts."""
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.handlers: List[Callable] = []
    
    def add_alert(self, alert: Alert):
        """Add an alert definition."""
        self.alerts.append(alert)
    
    def add_handler(self, handler: Callable):
        """Add alert handler (e.g., email, Slack, PagerDuty)."""
        self.handlers.append(handler)
    
    def check_alerts(self, metrics: Dict[str, Any]):
        """Check all alerts against current metrics."""
        triggered = []
        
        for alert in self.alerts:
            if alert.condition(metrics):
                alert.trigger_count += 1
                alert.last_triggered = datetime.utcnow()
                triggered.append(alert)
                
                # Notify handlers
                for handler in self.handlers:
                    handler(alert, metrics)
        
        return triggered

# Example alerts
def create_standard_alerts(monitor: PerformanceMonitor) -> AlertManager:
    """Create standard monitoring alerts."""
    manager = AlertManager()
    
    # High latency alert
    manager.add_alert(Alert(
        name="high_latency",
        condition=lambda m: m.get('latency', {}).get('p99', 0) > 100,
        message="P99 latency exceeds 100ms",
        severity=AlertSeverity.WARNING
    ))
    
    # Error rate alert
    manager.add_alert(Alert(
        name="high_error_rate",
        condition=lambda m: m.get('reliability', {}).get('error_rate', 0) > 0.01,
        message="Error rate exceeds 1%",
        severity=AlertSeverity.CRITICAL
    ))
    
    # Memory alert
    manager.add_alert(Alert(
        name="high_memory",
        condition=lambda m: m.get('memory_mb', {}).get('current', 0) > 4096,
        message="Memory usage exceeds 4GB",
        severity=AlertSeverity.WARNING
    ))
    
    return manager
```

## Dashboard Example

### Grafana Dashboard JSON

```json
{
  "title": "ML Model Monitoring",
  "panels": [
    {
      "title": "Request Rate",
      "type": "graph",
      "targets": [
        {
          "expr": "rate(inference_requests_total[5m])",
          "legendFormat": "{{model_name}}"
        }
      ]
    },
    {
      "title": "Latency P99",
      "type": "graph",
      "targets": [
        {
          "expr": "histogram_quantile(0.99, rate(inference_latency_seconds_bucket[5m]))",
          "legendFormat": "P99"
        }
      ]
    },
    {
      "title": "Error Rate",
      "type": "stat",
      "targets": [
        {
          "expr": "sum(rate(inference_requests_total{status='error'}[5m])) / sum(rate(inference_requests_total[5m]))"
        }
      ]
    },
    {
      "title": "GPU Utilization",
      "type": "gauge",
      "targets": [
        {
          "expr": "gpu_utilization_percent",
          "legendFormat": "GPU {{device}}"
        }
      ]
    }
  ]
}
```

## Best Practices

1. **Define SLOs first** — Establish latency, throughput, and availability targets before deployment.

2. **Monitor at multiple levels** — Track application metrics, infrastructure metrics, and business metrics.

3. **Use percentiles** — P95/P99 latency reveals tail behavior that means/medians hide.

4. **Set up alerting** — Configure alerts for SLO violations with appropriate thresholds.

5. **Retain historical data** — Keep metrics for trend analysis and capacity planning.

6. **Correlate metrics** — Link model performance to business outcomes.

7. **Monitor data quality** — Track input distribution shifts and data quality issues.

8. **Document runbooks** — Create incident response procedures for common alerts.

## References

1. Prometheus Documentation: https://prometheus.io/docs/
2. Grafana Documentation: https://grafana.com/docs/
3. Google SRE Book: https://sre.google/sre-book/table-of-contents/
4. ML Model Monitoring: https://christophergs.com/machine%20learning/2020/03/14/how-to-monitor-machine-learning-models/
