# MLflow

MLflow is an open-source platform for managing the complete machine learning lifecycle, including experimentation, reproducibility, deployment, and model registry. Unlike TensorBoard (focused on visualization) and W&B (focused on experiment tracking), MLflow provides a comprehensive framework that bridges the gap between model development and production deployment.

## Motivation

The machine learning lifecycle extends far beyond training:

- **Experimentation**: Tracking parameters, metrics, and artifacts across runs
- **Reproducibility**: Packaging code and dependencies for exact reproduction
- **Deployment**: Serving models through REST APIs or batch inference
- **Model Management**: Versioning, staging, and promoting models through environments

MLflow addresses these challenges through four main components: Tracking, Projects, Models, and Model Registry.

## Installation

```bash
pip install mlflow
```

For additional backends:

```bash
# PostgreSQL backend
pip install mlflow[extras]

# AWS S3 artifact storage
pip install mlflow boto3

# Google Cloud Storage
pip install mlflow google-cloud-storage
```

## MLflow Components Overview

### MLflow Tracking

Records experiments: parameters, metrics, artifacts, and source code.

### MLflow Projects

Packages ML code in a reusable, reproducible format.

### MLflow Models

Packages models for deployment in diverse serving environments.

### MLflow Model Registry

Centralized model store for versioning and lifecycle management.

## MLflow Tracking

### Basic Logging

```python
import mlflow

# Set experiment (creates if doesn't exist)
mlflow.set_experiment("mnist-classification")

# Start a run
with mlflow.start_run(run_name="baseline-mlp"):
    # Log parameters
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 64)
    mlflow.log_param("hidden_size", 500)
    
    # Log multiple parameters at once
    mlflow.log_params({
        "optimizer": "adam",
        "epochs": 10,
        "dropout": 0.2
    })
    
    # Training loop
    for epoch in range(10):
        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss, val_acc = evaluate(model, val_loader)
        
        # Log metrics (step parameter for x-axis)
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)
    
    # Log final metrics
    mlflow.log_metrics({
        "final_accuracy": val_acc,
        "final_loss": val_loss
    })
```

### Logging Artifacts

```python
import mlflow
import matplotlib.pyplot as plt
import json

with mlflow.start_run():
    # Log a file
    with open("config.json", "w") as f:
        json.dump(config, f)
    mlflow.log_artifact("config.json")
    
    # Log a directory
    mlflow.log_artifacts("./outputs", artifact_path="model_outputs")
    
    # Log matplotlib figure
    fig, ax = plt.subplots()
    ax.plot(train_losses)
    ax.set_title("Training Loss")
    fig.savefig("loss_curve.png")
    mlflow.log_artifact("loss_curve.png")
    plt.close(fig)
    
    # Log model
    mlflow.pytorch.log_model(model, "model")
```

### Autologging

MLflow provides automatic logging for popular frameworks:

```python
import mlflow
import mlflow.pytorch

# Enable autologging for PyTorch
mlflow.pytorch.autolog()

# Training code - MLflow automatically logs:
# - Parameters from optimizer
# - Metrics (loss)
# - Model architecture
# - Model weights
model = train_model(train_loader, val_loader)
```

For scikit-learn:

```python
import mlflow
mlflow.sklearn.autolog()

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
# Automatically logs parameters, metrics, and model
```

### Run Management

```python
import mlflow

# Get current run
run = mlflow.active_run()
print(f"Run ID: {run.info.run_id}")

# Set tags
mlflow.set_tag("model_type", "mlp")
mlflow.set_tags({
    "team": "research",
    "priority": "high"
})

# End run explicitly
mlflow.end_run()

# Nested runs
with mlflow.start_run(run_name="parent"):
    mlflow.log_param("experiment_type", "hyperparameter_search")
    
    for lr in [0.001, 0.01, 0.1]:
        with mlflow.start_run(run_name=f"lr_{lr}", nested=True):
            mlflow.log_param("learning_rate", lr)
            accuracy = train_and_evaluate(lr)
            mlflow.log_metric("accuracy", accuracy)
```

### Querying Runs

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get experiment by name
experiment = client.get_experiment_by_name("mnist-classification")

# Search runs
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="metrics.val_accuracy > 0.95",
    order_by=["metrics.val_accuracy DESC"],
    max_results=10
)

for run in runs:
    print(f"Run ID: {run.info.run_id}")
    print(f"Accuracy: {run.data.metrics['val_accuracy']}")
    print(f"Parameters: {run.data.params}")
    print()

# Load best model
best_run = runs[0]
model = mlflow.pytorch.load_model(f"runs:/{best_run.info.run_id}/model")
```

## MLflow Projects

MLflow Projects package ML code for reproducible execution.

### Project Structure

```
my_project/
├── MLproject                 # Project configuration
├── conda.yaml               # Environment specification
├── train.py                 # Training script
├── model.py                 # Model definition
└── utils.py                 # Utility functions
```

### MLproject File

```yaml
name: mnist-classification

conda_env: conda.yaml

entry_points:
  main:
    parameters:
      learning_rate: {type: float, default: 0.001}
      batch_size: {type: int, default: 64}
      epochs: {type: int, default: 10}
    command: "python train.py --lr {learning_rate} --batch-size {batch_size} --epochs {epochs}"
  
  evaluate:
    parameters:
      model_path: {type: string}
    command: "python evaluate.py --model-path {model_path}"
```

### Conda Environment

```yaml
# conda.yaml
name: mnist-env
channels:
  - pytorch
  - conda-forge
dependencies:
  - python=3.9
  - pytorch
  - torchvision
  - numpy
  - pip:
    - mlflow
    - matplotlib
```

### Running Projects

```bash
# Run locally
mlflow run . -P learning_rate=0.01 -P epochs=20

# Run from GitHub
mlflow run https://github.com/user/project -P learning_rate=0.01

# Run with specific entry point
mlflow run . -e evaluate -P model_path=./models/best_model.pth
```

### Training Script for Projects

```python
# train.py
import argparse
import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    return parser.parse_args()

def main():
    args = parse_args()
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            'learning_rate': args.lr,
            'batch_size': args.batch_size,
            'epochs': args.epochs
        })
        
        # Data loading
        transform = transforms.ToTensor()
        train_dataset = torchvision.datasets.MNIST(
            './data', train=True, download=True, transform=transform
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        
        # Model
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        # Training
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            mlflow.log_metric('train_loss', avg_loss, step=epoch)
            print(f'Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}')
        
        # Log model
        mlflow.pytorch.log_model(model, 'model')
        
        print('Training complete!')

if __name__ == '__main__':
    main()
```

## MLflow Models

MLflow Models provides a standard format for packaging models.

### Model Flavors

MLflow supports multiple "flavors" for model serialization:

```python
import mlflow
import mlflow.pytorch
import mlflow.sklearn
import mlflow.tensorflow

# PyTorch
mlflow.pytorch.log_model(pytorch_model, "pytorch_model")

# Scikit-learn
mlflow.sklearn.log_model(sklearn_model, "sklearn_model")

# Custom model with signature
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

input_schema = Schema([
    ColSpec("double", "feature_1"),
    ColSpec("double", "feature_2"),
])
output_schema = Schema([ColSpec("long", "prediction")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

mlflow.pytorch.log_model(
    model, 
    "model",
    signature=signature,
    input_example=sample_input
)
```

### Custom Python Models

For complex inference logic:

```python
import mlflow.pyfunc

class CustomModel(mlflow.pyfunc.PythonModel):
    def __init__(self, preprocessor, model, postprocessor):
        self.preprocessor = preprocessor
        self.model = model
        self.postprocessor = postprocessor
    
    def predict(self, context, model_input):
        # Preprocess
        processed = self.preprocessor.transform(model_input)
        
        # Predict
        raw_predictions = self.model(processed)
        
        # Postprocess
        final_predictions = self.postprocessor.process(raw_predictions)
        
        return final_predictions

# Save custom model
custom_model = CustomModel(preprocessor, model, postprocessor)
mlflow.pyfunc.log_model(
    artifact_path="custom_model",
    python_model=custom_model,
    artifacts={"preprocessor": "preprocessor.pkl"}
)
```

### Model Serving

Deploy models as REST APIs:

```bash
# Serve model from run
mlflow models serve -m runs:/<run_id>/model -p 5000

# Serve model from Model Registry
mlflow models serve -m "models:/my-model/Production" -p 5000
```

Query the served model:

```python
import requests
import json

# Prepare data
data = {
    "inputs": [[0.1, 0.2, 0.3, ...]]  # Feature values
}

# Make prediction
response = requests.post(
    "http://localhost:5000/invocations",
    headers={"Content-Type": "application/json"},
    data=json.dumps(data)
)

predictions = response.json()
print(predictions)
```

### Batch Scoring

```python
import mlflow

# Load model
model = mlflow.pyfunc.load_model("runs:/<run_id>/model")

# Score batch data
import pandas as pd
data = pd.read_csv("test_data.csv")
predictions = model.predict(data)
```

## MLflow Model Registry

Centralized model management with versioning and staging.

### Registering Models

```python
import mlflow

with mlflow.start_run():
    # Train model
    model = train_model()
    
    # Log and register in one step
    mlflow.pytorch.log_model(
        model,
        "model",
        registered_model_name="mnist-classifier"
    )

# Or register from existing run
result = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name="mnist-classifier"
)
print(f"Model version: {result.version}")
```

### Managing Model Versions

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get model versions
versions = client.search_model_versions("name='mnist-classifier'")
for v in versions:
    print(f"Version {v.version}: Stage={v.current_stage}")

# Transition model stage
client.transition_model_version_stage(
    name="mnist-classifier",
    version=1,
    stage="Staging"  # None, Staging, Production, Archived
)

# Add description
client.update_model_version(
    name="mnist-classifier",
    version=1,
    description="Initial baseline model with MLP architecture"
)

# Load model by stage
model = mlflow.pyfunc.load_model("models:/mnist-classifier/Production")
```

### Model Lifecycle Example

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

def promote_model(model_name, from_stage, to_stage, metric_threshold):
    """Promote model if it meets performance threshold."""
    
    # Get models in source stage
    versions = client.get_latest_versions(model_name, stages=[from_stage])
    
    for version in versions:
        # Load model and evaluate
        model_uri = f"models:/{model_name}/{version.version}"
        model = mlflow.pyfunc.load_model(model_uri)
        
        # Get metrics from the run
        run = client.get_run(version.run_id)
        accuracy = run.data.metrics.get('val_accuracy', 0)
        
        if accuracy >= metric_threshold:
            # Archive current production model
            prod_versions = client.get_latest_versions(
                model_name, stages=[to_stage]
            )
            for pv in prod_versions:
                client.transition_model_version_stage(
                    name=model_name,
                    version=pv.version,
                    stage="Archived"
                )
            
            # Promote new model
            client.transition_model_version_stage(
                name=model_name,
                version=version.version,
                stage=to_stage
            )
            
            print(f"Promoted version {version.version} to {to_stage}")
            return True
    
    return False

# Usage
promote_model(
    model_name="mnist-classifier",
    from_stage="Staging",
    to_stage="Production",
    metric_threshold=0.95
)
```

## Tracking Server Configuration

### Local Server

```bash
# Start tracking server with SQLite backend
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlartifacts \
    --host 0.0.0.0 \
    --port 5000
```

### Remote Server

```bash
# PostgreSQL backend with S3 artifacts
mlflow server \
    --backend-store-uri postgresql://user:password@host:5432/mlflow \
    --default-artifact-root s3://my-bucket/mlflow-artifacts \
    --host 0.0.0.0 \
    --port 5000
```

### Connecting to Server

```python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("http://mlflow-server:5000")

# Or via environment variable
# export MLFLOW_TRACKING_URI=http://mlflow-server:5000

# Verify connection
print(mlflow.get_tracking_uri())
```

## Complete Training Example

```python
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Configuration
config = {
    'learning_rate': 0.001,
    'batch_size': 64,
    'hidden_size': 500,
    'num_epochs': 10,
    'dropout': 0.2
}

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train():
    # Set experiment
    mlflow.set_experiment("mnist-classification")
    
    # Set tracking URI (use local SQLite for this example)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with mlflow.start_run(run_name="mlp-baseline"):
        # Log parameters
        mlflow.log_params(config)
        mlflow.set_tag("model_type", "mlp")
        
        # Data loading
        transform = transforms.ToTensor()
        train_dataset = torchvision.datasets.MNIST(
            './data', train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            './data', train=False, transform=transform
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=config['batch_size'], shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=config['batch_size']
        )
        
        # Model
        model = MLP(784, config['hidden_size'], 10, config['dropout']).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config['learning_rate']
        )
        
        # Log model architecture
        mlflow.log_text(str(model), "model_architecture.txt")
        
        train_losses = []
        val_accuracies = []
        
        for epoch in range(config['num_epochs']):
            # Training
            model.train()
            total_loss = 0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Evaluation
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            val_accuracy = correct / total
            val_accuracies.append(val_accuracy)
            
            # Log metrics
            mlflow.log_metrics({
                'train_loss': avg_train_loss,
                'val_accuracy': val_accuracy
            }, step=epoch)
            
            print(f'Epoch {epoch+1}/{config["num_epochs"]} - '
                  f'Loss: {avg_train_loss:.4f}, Accuracy: {val_accuracy:.4f}')
        
        # Log final metrics
        mlflow.log_metric('final_accuracy', val_accuracies[-1])
        
        # Log training curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(train_losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        
        ax2.plot(val_accuracies)
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        
        plt.tight_layout()
        plt.savefig('training_curves.png')
        mlflow.log_artifact('training_curves.png')
        plt.close(fig)
        
        # Log model with signature
        from mlflow.models.signature import infer_signature
        
        sample_input = torch.randn(1, 1, 28, 28)
        model.eval()
        with torch.no_grad():
            sample_output = model(sample_input)
        
        signature = infer_signature(
            sample_input.numpy(),
            sample_output.numpy()
        )
        
        mlflow.pytorch.log_model(
            model,
            "model",
            signature=signature,
            registered_model_name="mnist-classifier"
        )
        
        print(f"\nRun ID: {mlflow.active_run().info.run_id}")
        print(f"Final Accuracy: {val_accuracies[-1]:.4f}")

if __name__ == '__main__':
    train()
```

## MLflow UI

Launch the UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Access at `http://localhost:5000`. Features include:

- **Experiment list**: Browse all experiments
- **Run comparison**: Compare metrics across runs
- **Artifact viewer**: Browse logged files and models
- **Search**: Query runs by parameters and metrics
- **Model Registry**: Manage registered models and stages

## Comparison with Other Tools

| Feature | MLflow | TensorBoard | W&B |
|---------|--------|-------------|-----|
| Metric Tracking | ✓ | ✓ | ✓ |
| Model Registry | ✓ | ✗ | ✓ |
| Model Serving | ✓ | ✗ | ✓ |
| Reproducibility | ✓ (Projects) | ✗ | ✗ |
| Open Source | ✓ | ✓ | Partial |
| Self-hosted | ✓ | ✓ | ✓ (Enterprise) |
| Cloud Offering | Databricks | ✗ | ✓ |

## Best Practices

### Experiment Organization

```python
# Use meaningful experiment names
mlflow.set_experiment("project/task/approach")

# Use tags for filtering
mlflow.set_tags({
    "team": "ml-research",
    "task": "classification",
    "dataset_version": "v2"
})
```

### Artifact Management

```python
# Structure artifacts logically
mlflow.log_artifacts("./configs", artifact_path="configs")
mlflow.log_artifacts("./plots", artifact_path="visualizations")
mlflow.log_model(model, "models/classifier")
```

### Model Signatures

```python
# Always include signatures for deployment
from mlflow.models.signature import infer_signature

signature = infer_signature(X_train, model.predict(X_train))
mlflow.sklearn.log_model(model, "model", signature=signature)
```

### CI/CD Integration

```yaml
# GitHub Actions example
name: MLflow Training
on: push
jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Train model
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_URI }}
        run: |
          pip install mlflow
          mlflow run . -P epochs=10
```

## Summary

MLflow provides an end-to-end platform for the ML lifecycle:

1. **Tracking**: Log and query experiments with parameters, metrics, and artifacts
2. **Projects**: Package code for reproducible runs across environments
3. **Models**: Standardized model format supporting multiple deployment targets
4. **Model Registry**: Centralized model management with versioning and staging

For organizations requiring full lifecycle management from experimentation through production deployment, MLflow offers the necessary infrastructure while remaining open source and vendor-agnostic.
